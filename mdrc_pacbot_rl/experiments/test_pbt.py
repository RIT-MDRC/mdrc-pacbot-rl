"""
Experiment for testing population based training.
"""
from functools import reduce
from typing import Any

import envpool  # type: ignore
import torch
import torch.nn as nn
from gym.envs.classic_control.cartpole import CartPoleEnv
from ray.air import Checkpoint, session
from ray import tune
from torch.distributions import Categorical
from ray.tune.schedulers import PopulationBasedTraining
from mdrc_pacbot_rl.algorithms.rollout_buffer import RolloutBuffer
from mdrc_pacbot_rl.utils import copy_params, init_orthogonal

_: Any

# Hyperparameters.
# Any hyperparams in the `config` dict will be changed during population based
# training.
num_envs = 128
train_steps = 500
iterations = 20
max_eval_steps = 500
perturbation_interval = 10
report_interval = 10
trials_at_a_time = 2  # Number of trials to keep track of at a time
config = {
    "v_lr": tune.uniform(0.001, 0.1),
    "p_lr": tune.uniform(0.001, 0.1),
    "epsilon": tune.uniform(0.1, 0.4),
    "discount": tune.uniform(0.9, 1.0),
    "lambda": tune.uniform(0.9, 1.0),
    "train_iters": list(range(10)),
    "train_batch_size": [2**x for x in range(5, 10)],
}
resume = False  # Setting this to True resumes where you left off
device = torch.device("cpu")


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

    def forward(self, input: torch.Tensor):
        x = self.a_layer1(input.flatten(1))
        x = self.relu(x)
        x = self.a_layer2(x)
        x = self.relu(x)
        x = self.a_layer3(x)
        x = self.logits(x)
        return x


def run(config: dict):
    env = envpool.make("CartPole-v1", "gym", num_envs=num_envs)
    test_env = CartPoleEnv()

    # Load hyperparams from config
    v_lr = config["v_lr"]
    p_lr = config["p_lr"]
    epsilon = config["epsilon"]
    discount = config["discount"]
    lambda_ = config["lambda"]
    train_iters = config["train_iters"]
    train_batch_size = config["train_batch_size"]

    # Initialize policy and value networks
    obs_space = env.observation_space
    act_space = env.action_space
    v_net = ValueNet(obs_space.shape)
    p_net = PolicyNet(obs_space.shape, act_space.n)
    p_net_old = PolicyNet(obs_space.shape, act_space.n)
    p_net_old.eval()
    v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
    p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)

    buffer = RolloutBuffer(
        obs_space.shape,
        torch.Size((1,)),
        torch.int,
        num_envs,
        train_steps,
        device,
    )

    # Load from checkpoint if applicable
    start_step = 0
    if session.get_checkpoint():
        checkpoint = session.get_checkpoint().to_dict()
        start_step = checkpoint["step"]
        v_net.load_state_dict(checkpoint["v_net_state_dict"])
        p_net.load_state_dict(checkpoint["p_net_state_dict"])
        v_opt.load_state_dict(checkpoint["v_opt_state_dict"])
        p_opt.load_state_dict(checkpoint["p_opt_state_dict"])
        for param_group in v_opt.param_groups:
            param_group["lr"] = v_lr
        for param_group in p_opt.param_groups:
            param_group["lr"] = p_lr

    obs = torch.Tensor(env.reset()[0])
    done = False
    for step in range(start_step, iterations):
        # Collect experience for a number of steps and store it in the buffer
        with torch.no_grad():
            for _ in range(train_steps):
                actions = Categorical(logits=p_net(obs)).sample().numpy()
                obs_, rewards, dones, _, _ = env.step(actions)
                buffer.insert_step(
                    obs, torch.from_numpy(actions).unsqueeze(-1), rewards, dones
                )
                obs = torch.from_numpy(obs_)
                if done:
                    obs = torch.Tensor(env.reset()[0])
                    done = False
            buffer.insert_final_step(obs)

        # Train
        p_net.train()
        v_net.train()
        copy_params(p_net, p_net_old)

        for _ in range(train_iters):
            # The rollout buffer provides randomized minibatches of samples
            batches = buffer.samples(train_batch_size, discount, lambda_, v_net)
            for prev_states, _, actions, _, rewards_to_go, advantages, _ in batches:
                # Train policy network
                with torch.no_grad():
                    old_log_probs = p_net_old(prev_states)
                    old_act_probs = Categorical(logits=old_log_probs).log_prob(
                        actions.squeeze()
                    )
                p_opt.zero_grad()
                new_log_probs = p_net(prev_states)
                new_act_probs = Categorical(logits=new_log_probs).log_prob(
                    actions.squeeze()
                )
                term1: torch.Tensor = (new_act_probs - old_act_probs).exp() * advantages
                term2: torch.Tensor = (1.0 + epsilon * advantages.sign()) * advantages
                p_loss = -term1.min(term2).mean()
                p_loss.backward()
                p_opt.step()

                # Train value network
                v_opt.zero_grad()
                diff: torch.Tensor = v_net(prev_states) - rewards_to_go.unsqueeze(1)
                v_loss = (diff * diff).mean()
                v_loss.backward()
                v_opt.step()

        p_net.eval()
        v_net.eval()
        buffer.clear()

        # Evaluate
        if step % report_interval == 0:
            done = False
            with torch.no_grad():
                reward_total = 0
                entropy_total = 0.0
                obs = torch.Tensor(test_env.reset()[0])
                eval_steps = 8
                for _ in range(eval_steps):
                    avg_entropy = 0.0
                    steps_taken = 0
                    for _ in range(max_eval_steps):
                        distr = Categorical(logits=p_net(obs.unsqueeze(0)).squeeze())
                        action = distr.sample().item()
                        obs_, reward, done, _, _ = test_env.step(action)
                        obs = torch.Tensor(obs_)
                        steps_taken += 1
                        if done:
                            obs = torch.Tensor(test_env.reset()[0])
                            break
                        reward_total += reward
                        avg_entropy += distr.entropy().mean().item()
                    avg_entropy /= steps_taken
                    entropy_total += avg_entropy

            # Perform checkpointing and report
            checkpoint = Checkpoint.from_dict(
                {
                    "step": step,
                    "v_net_state_dict": v_net.state_dict(),
                    "p_net_state_dict": p_net.state_dict(),
                    "v_opt_state_dict": v_opt.state_dict(),
                    "p_opt_state_dict": p_opt.state_dict(),
                }
            )
            session.report(
                {
                    "step": step,
                    "avg_reward": reward_total / eval_steps,
                    "avg_entropy": entropy_total / eval_steps,
                },
                checkpoint=checkpoint,
            )

        obs = torch.Tensor(env.reset()[0])
        done = False


# Use PBT to find hyperparameters
pbt_scheduler = PopulationBasedTraining(
    time_attr="step",
    metric="avg_reward",
    mode="max",
    perturbation_interval=perturbation_interval,
    hyperparam_mutations=config,
)
analysis = tune.run(
    run,
    scheduler=pbt_scheduler,
    resources_per_trial={"cpu": 1},
    num_samples=trials_at_a_time,
    resume=resume,
)
