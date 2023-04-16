"""
Experiment for checking that PPO is working. Also a primer on PPO!

Proximal Policy Optimization (PPO) is a popular deep reinforcement learning
algorithm. At OpenAI and a lot of other places, it's used as a baseline, since
you can get pretty good performance without having to fiddle with the
hyperparameters too much.

Background:

PPO is best understood through its improvements over its predecessors. Its
evolutionary line looks like this:

DQN--
    |-->A2C-->TRPO-->PPO
VPG--

There are mini primers for each below, but if you want to learn more, there's a
bunch of resources online you can use, especially for the first three algorithms
introduced.

Deep Q-Networks (DQN) are value based RL algorithms. You train a neural net to
predict the return (sum of all rewards from a given step) from the current
environment state for each posslbe action. Then, you just choose the action with
the highest predicted reward! There's a ton of resources on DQNs since they're
easy to understand, so feel free to read up on them before going further.

Vanilla Policy Gradients (VPG, also called REINFORCE) are based on policy
gradients, which are another way to approach deep RL. Instead of outputting the
predicted value of each action, it outputs the probability of taking each action
instead. When training, instead of using mean squared error as the loss and
performing gradient descent, you instead multiply the (log) probability of the
action taken with the return as the "loss" and perform gradient ascent! This
incentivizes the net to increase the probability of actions that lead to high
returns, and lower the probability of actions that lead to low returns.

Since policy gradients output probabilities, they work for both discrete actions
(just do softmax over each action and sample) and continuous actions (have the
network output the mean and variance of each continuous action, then sample
actions under a normal distribution). They also can act nondeterministically,
unlike DQNs. Unfortunately, they also tend to be very unstable and need tons of
samples to get convergence. This is because raw returns have a lot of variance
(the same actions can lead to wildly different final results), leading to
instability during training.

Advantage Actor Critic (A2C) is an algorithm that rectifies some of these
problems. It's a policy gradient method that also learns a value function. The
value function learns the predicted return of a state (like DQNs do), and the
predicted values are used instead of raw returns when training the network. The
predicted values form the "advantage estimate", which is just how good that
action is relative to the other actions you could take. Advantages have way less
variance, so training is much more stable. Mission accomplished!

Except, not really. One other advantage of DQNs is that they're off policy. That
means you can use experience that wasn't generated by your target neural net to
train it. On policy methods, which most policy gradient algorithms are, can only
use experience generated by the network you're training. In practice, what that
means is as soon as an on policy algorithm updates its weights, you have to
throw out all the experience you've gathered and get more. So, you can't do
multiple training iterations after a sampling pass, or even use minibatches. If
you try to do it anyway, at best, you'll get suboptimal training performance,
and at worst, your network will suddenly catastrophically fail.

Trust Region Policy Optimization (TRPO) tries to fix this by correcting off
policy samples. It uses importance sampling and a complicated second order
method to ensure the network doesn't try to learn from experience that's no
longer applicable. It works, but unfortunately, it's too complicated and
computationally expensive to recommend.

PPO builds on the lessons from TRPO and encapsulates its requirements in an
elegant loss function. It replaces the log probability with an importance
sampling term, and also stops training if it detects experience is no longer
applicable, through clipping. Complexity wise, it's just A2C with 2-3 lines
changed!
"""
from functools import reduce
from typing import Any

import envpool  # type: ignore
import torch
import torch.nn as nn
import wandb
from gym.envs.box2d.lunar_lander import LunarLander
from torch.distributions import Categorical
from tqdm import tqdm
from mdrc_pacbot_rl.algorithms.ppo import train_ppo

from mdrc_pacbot_rl.algorithms.rollout_buffer import RolloutBuffer
from mdrc_pacbot_rl.utils import copy_params, init_orthogonal

_: Any

# Hyperparameters
num_envs = 256
train_steps = 250
iterations = 300
train_iters = 2
train_batch_size = 512
discount = 0.98
lambda_ = 0.95
epsilon = 0.2
max_eval_steps = 500
v_lr = 0.01
p_lr = 0.001
device = torch.device("cuda")

wandb.init(
    project="tests",
    entity="mdrc-pacbot",
    config={
        "experiment": "ppo",
        "num_envs": num_envs,
        "train_steps": train_steps,
        "train_iters": train_iters,
        "train_batch_size": train_batch_size,
        "discount": discount,
        "lambda": lambda_,
        "epsilon": epsilon,
        "max_eval_steps": max_eval_steps,
        "v_lr": v_lr,
        "p_lr": p_lr,
    },
)


# The value network takes in an observation and returns a single value, the
# predicted return
class ValueNet(nn.Module):
    def __init__(self, obs_shape: torch.Size):
        nn.Module.__init__(self)
        flat_obs_dim = reduce(lambda e1, e2: e1 * e2, obs_shape, 1)
        self.v_layer1 = nn.Linear(flat_obs_dim, 256)
        self.v_layer2 = nn.Linear(256, 256)
        self.v_layer3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x = self.v_layer1(input.flatten(1))
        x = self.relu(x)
        x = self.v_layer2(x)
        x = self.relu(x)
        x = self.v_layer3(x)
        return x


# The policy network takes in an observation and returns the log probability of
# taking each action
class PolicyNet(nn.Module):
    def __init__(self, obs_shape: torch.Size, action_count: int):
        nn.Module.__init__(self)
        flat_obs_dim = reduce(lambda e1, e2: e1 * e2, obs_shape, 1)
        self.a_layer1 = nn.Linear(flat_obs_dim, 256)
        self.a_layer2 = nn.Linear(256, 256)
        self.a_layer3 = nn.Linear(256, action_count)
        self.relu = nn.ReLU()
        self.logits = nn.LogSoftmax(1)
        init_orthogonal(self)

    def forward(self, input: torch.Tensor, masks=None):
        x = self.a_layer1(input.flatten(1))
        x = self.relu(x)
        x = self.a_layer2(x)
        x = self.relu(x)
        x = self.a_layer3(x)
        x = self.logits(x)
        return x


env = envpool.make("LunarLander-v2", "gym", num_envs=num_envs)
test_env = LunarLander(render_mode="human")

# Initialize policy and value networks
obs_space = env.observation_space
act_space = env.action_space
v_net = ValueNet(obs_space.shape)
p_net = PolicyNet(obs_space.shape, act_space.n)
v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)

# A rollout buffer stores experience collected during a sampling run
buffer = RolloutBuffer(
    obs_space.shape,
    torch.Size((1,)),
    torch.Size((int(act_space.n),)),
    torch.int,
    num_envs,
    train_steps,
    device,
)

obs = torch.Tensor(env.reset()[0])
done = False
for step in tqdm(range(iterations), position=0):
    # Collect experience for a number of steps and store it in the buffer
    with torch.no_grad():
        for _ in tqdm(range(train_steps), position=1):
            action_probs = p_net(obs)
            actions = Categorical(logits=action_probs).sample().numpy()
            obs_, rewards, dones, truncs, _ = env.step(actions)
            buffer.insert_step(
                obs,
                torch.from_numpy(actions).unsqueeze(-1),
                action_probs,
                rewards,
                dones,
                truncs,
                None,
            )
            obs = torch.from_numpy(obs_)
            if done:
                obs = torch.Tensor(env.reset()[0])
                done = False
        buffer.insert_final_step(obs)

    # Train
    total_p_loss, total_v_loss = train_ppo(
        p_net,
        v_net,
        p_opt,
        v_opt,
        buffer,
        device,
        train_iters,
        train_batch_size,
        discount,
        lambda_,
        epsilon,
        gradient_steps=1,
    )

    """
    p_net.train()
    v_net.train()
    total_v_loss = 0.0
    total_p_loss = 0.0
    for _ in range(train_iters):
        # The rollout buffer provides randomized minibatches of samples
        batches = buffer.samples(train_batch_size, discount, lambda_, v_net)
        for prev_states, actions, action_probs, returns, advantages, _ in batches:
            # Train policy network.
            #
            # First, we get the log probabilities of taking the actions we took
            # when we took them.
            with torch.no_grad():
                old_act_probs = Categorical(logits=action_probs).log_prob(
                    actions.squeeze()
                )
            # Next, we get the log probabilities of taking the actions with our
            # current network. During the first iteration, when we sample our
            # first minibatch, this should give us the exact same probabilities
            # as the step above, since we didn't update the network yet.
            p_opt.zero_grad()
            new_log_probs = p_net(prev_states)
            new_act_probs = Categorical(logits=new_log_probs).log_prob(
                actions.squeeze()
            )
            # Then, we run PPO's loss function, which is sometimes called the
            # surrogate loss. Written out explicitly, it's
            # min(current_probs/prev_probs * advantages,
            # clamp(current_probs/prev_probs, 1 - epsilon, 1 + epsilon) *
            # advantages). The actual code written is just a more optimized way
            # of writing that.
            #
            # Basically, we only want to update our network if the probability
            # of taking the actions with our current net is slightly less or
            # slightly more than the probability of taking the actions under the
            # old net. If that ratio is too high or too low, then the clipping
            # kicks in, and the gradient goes to 0 since we're differentiating a
            # constant (1 - epsilon or 1 + epsilon).
            #
            # Note that the only major difference is the importance sampling
            # term (the ratio) and the clipping; the loss function for A2C is
            # current_log_probs * advantages, which is very similar. Also, we're
            # using a negative loss function because we're trying to maximize
            # this instead of minimizing it.
            term1: torch.Tensor = (new_act_probs - old_act_probs).exp() * advantages
            term2: torch.Tensor = (1.0 + epsilon * advantages.sign()) * advantages
            p_loss = -term1.min(term2).mean()
            p_loss.backward()
            p_opt.step()
            total_p_loss += p_loss.item()

            # Train value network. Hopefully, this part is much easier to
            # understand.
            v_opt.zero_grad()
            diff: torch.Tensor = v_net(prev_states) - returns.unsqueeze(1)
            v_loss = (diff * diff).mean()
            v_loss.backward()
            v_opt.step()
            total_v_loss += v_loss.item()

    p_net.eval()
    v_net.eval()
    """
    buffer.clear()

    # Evaluate the network's performance after this training iteration. The
    # reward per episode and entropy are both recorded here. Entropy is useful
    # because as the agent learns, unless there really *is* a benefit to
    # learning a policy with randomness (and usually there isn't), the agent
    # should act more deterministically as time goes on. So, the entropy should
    # decrease.
    #
    # No, you don't need to understand the code here.
    test_env.render_mode = None
    if step % 10 == 0:
        test_env.render_mode = "human"
    eval_obs = torch.Tensor(test_env.reset()[0])
    done = False
    with torch.no_grad():
        # Visualize
        reward_total = 0
        entropy_total = 0.0
        eval_obs = torch.Tensor(test_env.reset()[0])
        eval_steps = 8
        for _ in range(eval_steps):
            avg_entropy = 0.0
            steps_taken = 0
            for _ in range(max_eval_steps):
                distr = Categorical(logits=p_net(eval_obs.unsqueeze(0)).squeeze())
                action = distr.sample().item()
                obs_, reward, done, _, _ = test_env.step(action)
                eval_obs = torch.Tensor(obs_)
                steps_taken += 1
                if done:
                    eval_obs = torch.Tensor(test_env.reset()[0])
                    break
                reward_total += reward
                avg_entropy += distr.entropy()
            avg_entropy /= steps_taken
            entropy_total += avg_entropy

    wandb.log(
        {
            "avg_eval_episode_reward": reward_total / eval_steps,
            "avg_eval_entropy": entropy_total / eval_steps,
            "avg_v_loss": total_v_loss / train_iters,
            "avg_p_loss": total_p_loss / train_iters,
        }
    )

# Congrats! You now know how to write the same exact algorithm OpenAI uses in
# their own work! Next stop: https://openai.com/careers
