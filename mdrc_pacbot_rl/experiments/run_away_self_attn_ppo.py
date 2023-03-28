"""
Experiment for testing if the agent can learn to run from a ghost using self
attention.

CLI Args:
    --eval: Run the last saved policy in the test environment, with
    visualization.
"""
import sys
from typing import Any
import numpy as np

import torch
import torch.nn as nn
import wandb
from gymnasium.spaces.discrete import Discrete
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from torch.distributions import Categorical
from tqdm import tqdm

from mdrc_pacbot_rl.algorithms.rollout_buffer import RolloutBuffer
from mdrc_pacbot_rl.micro_envs import RunAwayEnv
from mdrc_pacbot_rl.utils import copy_params, get_img_size, init_orthogonal

_: Any

# Hyperparameters
num_envs = 256  # Number of environments to step through at once during sampling.
train_steps = 200  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs
iterations = 1000  # Number of sample/train iterations.
train_iters = 1  # Number of passes over the samples collected.
train_batch_size = 2048  # Minibatch size while training models.
discount = 0.98  # Discount factor applied to rewards.
lambda_ = 0.5  # Lambda for GAE.
epsilon = 0.2  # Epsilon for importance sample clipping.
eval_steps = 8  # Number of eval runs to average over.
max_eval_steps = 300  # Max number of steps to take during each eval run.
v_lr = 0.001  # Learning rate of the value net.
p_lr = 0.0001  # Learning rate of the policy net.
device = torch.device("cpu")


class ValueNet(nn.Module):
    def __init__(self, emb_dim: int):
        nn.Module.__init__(self)
        self.attn = nn.MultiheadAttention(emb_dim, 1, batch_first=True)
        self.v_layer1 = nn.Linear(emb_dim, 256)
        self.v_layer2 = nn.Linear(256, 256)
        self.v_layer3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x, _ = self.attn(input, input, input)
        x = torch.mean(x, 1)
        x = self.relu(x)
        x = self.v_layer1(x)
        x = self.relu(x)
        x = self.v_layer2(x)
        x = self.relu(x)
        x = self.v_layer3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, emb_dim: int, action_count: int):
        nn.Module.__init__(self)
        self.attn = nn.MultiheadAttention(emb_dim, 1, batch_first=True)
        self.a_layer1 = nn.Linear(emb_dim, 256)
        self.a_layer2 = nn.Linear(256, 256)
        self.a_layer3 = nn.Linear(256, action_count)
        self.relu = nn.ReLU()
        self.logits = nn.LogSoftmax(1)
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x, _ = self.attn(input, input, input)
        x = torch.mean(x, 1)
        x = self.relu(x)
        x = self.relu(x)
        x = self.a_layer1(x)
        x = self.relu(x)
        x = self.a_layer2(x)
        x = self.relu(x)
        x = self.a_layer3(x)
        x = self.logits(x)
        return x


EMB_DIM = 5
# TODO: Use graph embeddings
width_pos = torch.arange(0, 8).repeat(8, 1).T / 4.0 - 1.0
height_pos = torch.arange(0, 8).repeat(8, 1) / 4.0 - 1.0
pos = torch.stack([width_pos, height_pos])[np.newaxis, ...]
def process_obs(obs: np.ndarray) -> torch.Tensor:
    """
    Converts the base observation into a set of vectors. The original
    observation is an array of size BxCxWxH, where B is batch size, C is
    channels, W is width, and H is height. We concatenate a precalculated
    embedding array of size ExWxH, where E is the position embedding dim, giving
    us an array of Bx(C + E)xWxH. Finally, we flatten the array and swap
    dimensions, giving us Bx(W * H)x(C + E).
    """
    no_batch = len(obs.shape) < 4
    if no_batch:
        obs = obs[np.newaxis, ...]
    batch_size = obs.shape[0]
    positional = np.repeat(pos, batch_size, 0)
    obs = np.concatenate([obs, positional], 1)
    obs = obs.reshape([obs.shape[0], EMB_DIM, obs.shape[2] * obs.shape[3]]).swapaxes(1, 2)
    if no_batch:
        obs = obs.squeeze(0)
    return torch.from_numpy(obs)


env = SyncVectorEnv([lambda: RunAwayEnv() for _ in range(num_envs)])
test_env = RunAwayEnv()

# If evaluating, just run the eval env
if len(sys.argv) >= 2 and sys.argv[1] == "--eval":
    test_env = RunAwayEnv(render_mode="human")
    p_net = torch.load("temp/PNet.pt")
    with torch.no_grad():
        obs = torch.Tensor(test_env.reset()[0])
        for _ in range(max_eval_steps):
            distr = Categorical(logits=p_net(obs.unsqueeze(0)).squeeze())
            action = distr.sample().item()
            obs_, reward, done, _, _ = test_env.step(action)
            obs = torch.Tensor(obs_)
            if done:
                break
    quit()

wandb.init(
    project="pacbot",
    entity="mdrc-pacbot",
    config={
        "experiment": "run away ppo",
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

v_net_artifact = wandb.Artifact("v_net", "model")
p_net_artifact = wandb.Artifact("p_net", "model")

# Init PPO
obs_shape = env.envs[0].observation_space.shape
if not obs_shape:
    raise RuntimeError("Observation space doesn't have shape")
obs_size = torch.Size(obs_shape)
act_space = env.envs[0].action_space
if not isinstance(act_space, Discrete):
    raise RuntimeError("Action space was not discrete")
v_net = ValueNet(EMB_DIM)
p_net = PolicyNet(EMB_DIM, int(act_space.n))
v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)
buffer = RolloutBuffer(
    torch.Size((64, EMB_DIM)),
    torch.Size((1,)),
    torch.Size((5,)),
    torch.int,
    num_envs,
    train_steps,
    device,
)

obs = process_obs(env.reset()[0])
for _ in tqdm(range(iterations), position=0):
    # Collect experience
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
            )
            obs = process_obs(obs_)
        buffer.insert_final_step(obs)

    # Train
    p_net.train()
    v_net.train()

    total_v_loss = 0.0
    total_p_loss = 0.0
    for _ in tqdm(range(train_iters), position=1):
        batches = buffer.samples(train_batch_size, discount, lambda_, v_net)
        for prev_states, actions, action_probs, returns, advantages in batches:
            # Train policy network
            with torch.no_grad():
                old_act_probs = Categorical(logits=action_probs).log_prob(
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
            total_p_loss += p_loss.item()

            # Train value network
            v_opt.zero_grad()
            diff: torch.Tensor = v_net(prev_states) - returns
            v_loss = (diff * diff).mean()
            v_loss.backward()
            v_opt.step()
            total_v_loss += v_loss.item()

    p_net.eval()
    v_net.eval()
    buffer.clear()

    # Evaluate
    eval_done = False
    with torch.no_grad():
        # Visualize
        reward_total = 0
        pred_reward_total = 0
        entropy_total = 0.0
        eval_obs = process_obs(test_env.reset()[0])
        for _ in range(eval_steps):
            avg_entropy = 0.0
            steps_taken = 0
            score = 0
            for _ in range(max_eval_steps):
                distr = Categorical(logits=p_net(eval_obs.unsqueeze(0)).squeeze())
                action = distr.sample().item()
                pred_reward_total += v_net(eval_obs.unsqueeze(0)).squeeze().item()
                obs_, reward, eval_done, eval_trunc, _ = test_env.step(action)
                eval_obs = process_obs(obs_)
                steps_taken += 1
                reward_total += reward
                avg_entropy += distr.entropy()
                if eval_done or eval_trunc:
                    eval_obs = process_obs(test_env.reset()[0])
                    break
            avg_entropy /= steps_taken
            entropy_total += avg_entropy

    wandb.log(
        {
            "avg_eval_episode_reward": reward_total / eval_steps,
            "avg_eval_episode_predicted_reward": pred_reward_total / eval_steps,
            "avg_eval_entropy": entropy_total / eval_steps,
            "avg_v_loss": total_v_loss / train_iters,
            "avg_p_loss": total_p_loss / train_iters,
        }
    )

# Save artifacts
torch.save(v_net, "temp/VNet.pt")
v_net_artifact.add_file("temp/VNet.pt")
wandb.log_artifact(v_net_artifact)

torch.save(p_net, "temp/PNet.pt")
p_net_artifact.add_file("temp/PNet.pt")
wandb.log_artifact(p_net_artifact, "temp/PNet.pt")

wandb.finish()
