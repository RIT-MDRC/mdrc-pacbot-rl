"""
Baseline for Pacman gym, using DQN.

CLI Args:
    --eval: Run the last saved policy in the test environment, with visualization.
    --resume: Resume training from the last saved policy.
"""
import copy
import json
import random
import sys
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from gymnasium.spaces.discrete import Discrete
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.frame_stack import FrameStack
from torch.distributions import Categorical
from tqdm import tqdm

from mdrc_pacbot_rl.algorithms.ppo import train_ppo
from mdrc_pacbot_rl.algorithms.replay_buffer import ReplayBuffer
from mdrc_pacbot_rl.pacman.gym import SemanticChannelPacmanGym as PacmanGym
from mdrc_pacbot_rl.utils import (
    copy_params,
    get_img_size,
    get_img_size_3d,
    init_orthogonal,
)

_: Any
INF = 10**8

# Hyperparameters
num_envs = 32  # Number of environments to step through at once during sampling.
train_steps = 4  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs.
iterations = 100000  # Number of sample/train iterations.
train_iters = 1  # Number of passes over the samples collected.
train_batch_size = 512  # Minibatch size while training models.
discount = 0.999  # Discount factor applied to rewards.
q_epsilon = 0.1  # Epsilon for epsilon greedy strategy. This gets annealed over time.
eval_steps = 1  # Number of eval runs to average over.
max_eval_steps = 300  # Max number of steps to take during each eval run.
q_lr = 0.0001  # Learning rate of the q net.
warmup_steps = 500  # For the first n number of steps, we will only sample randomly.
device = torch.device("cuda")


class BaseNet(nn.Module):
    def __init__(self, obs_shape: torch.Size):
        nn.Module.__init__(self)
        d, c, w, h = obs_shape
        self.cnn1 = nn.Conv2d(c * d, 16, 3, padding=1)
        h, w = get_img_size(h, w, self.cnn1)
        self.cnn2 = nn.Conv2d(16, 32, 3, padding=1)
        h, w = get_img_size(h, w, self.cnn2)
        self.cnn3 = nn.Conv2d(32, 64, 3, padding=1)
        h, w = get_img_size(h, w, self.cnn3)
        self.flat_dim = 64
        self.relu = nn.ReLU()
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        input = input.flatten(1, 2)
        x = self.cnn1(input)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.cnn3(x)
        x = x.max(3).values.max(2).values
        return x


class QNet(nn.Module):
    def __init__(
        self,
        obs_shape: torch.Size,
        action_count: int,
    ):
        nn.Module.__init__(self)
        self.net = BaseNet(obs_shape)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.net.flat_dim, 128)
        self.advantage = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_count)
        )
        self.value = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.action_count = action_count
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x = self.net(input)
        x = self.relu(x)
        x = self.linear(x)
        x = self.relu(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)


stacked_frames = 1
env = SyncVectorEnv(
    [
        lambda: FrameStack(TimeLimit(PacmanGym(random_start=True), 1000), stacked_frames)
        for _ in range(num_envs)
    ]
)
test_env = FrameStack(PacmanGym(), stacked_frames)

# If evaluating, just run the eval env
if len(sys.argv) >= 2 and sys.argv[1] == "--eval":
    test_env = FrameStack(
        PacmanGym(render_mode="human", random_start=False), stacked_frames
    )
    q_net = torch.load("temp/QNet.pt")
    obs_shape = test_env.observation_space.shape
    if not obs_shape:
        raise RuntimeError("Observation space doesn't have shape")
    obs_size = torch.Size(obs_shape)
    with torch.no_grad():
        obs_, info = test_env.reset()
        action_mask = np.array(list(info["action_mask"]))
        obs = torch.from_numpy(np.array(obs_)).float()
        while True:
            q_vals = q_net(obs.unsqueeze(0)).squeeze()
            action = (
                torch.where(torch.from_numpy(action_mask) == 1, -INF, q_vals)
                .argmax(0)
                .item()
            )
            obs_, reward, done, _, info = test_env.step(action)
            print(q_vals, reward)
            obs = torch.from_numpy(np.array(obs_)).float()
            action_mask = np.array(list(info["action_mask"]))
            if done:
                obs_, info = test_env.reset()
                action_mask = np.array(list(info["action_mask"]))
                obs = torch.from_numpy(np.array(obs_)).float()
                q_net = torch.load("temp/QNet.pt")
    quit()

wandb.init(
    project="pacbot",
    entity="mdrc-pacbot",
    config={
        "experiment": "baseline dqn",
        "num_envs": num_envs,
        "train_steps": train_steps,
        "train_iters": train_iters,
        "train_batch_size": train_batch_size,
        "discount": discount,
        "q_epsilon": q_epsilon,
        "max_eval_steps": max_eval_steps,
        "q_lr": q_lr,
    },
)

q_net_artifact = wandb.Artifact("q_net", "model")

# Init DQN
obs_shape = env.envs[0].observation_space.shape
if not obs_shape:
    raise RuntimeError("Observation space doesn't have shape")
obs_size = torch.Size(obs_shape)
act_space = env.envs[0].action_space
if not isinstance(act_space, Discrete):
    raise RuntimeError("Action space was not discrete")
q_net = torch.load("temp/QNet.pt")#QNet(obs_size, int(act_space.n))
q_net_target = copy.deepcopy(q_net)
q_net_target.to(device)
q_opt = torch.optim.Adam(q_net.parameters(), lr=q_lr)
buffer = ReplayBuffer(
    torch.Size(obs_size),
    torch.Size((int(act_space.n),)),
    10000,
)

obs_, info = env.reset()
obs = torch.from_numpy(obs_)
action_mask = np.array(list(info["action_mask"]))
best_eval = 4000
score_total = 0
for step in tqdm(range(iterations), position=0):
    percent_done = step / iterations

    # Collect experience
    with torch.no_grad():
        for _ in range(train_steps):
            if (
                random.random() < q_epsilon * max(1.0 - percent_done, 0.05)
                or step < warmup_steps
            ):
                actions_list = []
                for mask in action_mask:
                    actions_list.append(
                        np.random.choice(act_space.n, p=(1 - mask) / (1 - mask).sum())
                    )
                actions_ = np.array(actions_list)
            else:
                q_vals = q_net(obs)
                actions_ = (
                    torch.where(torch.from_numpy(action_mask) == 1, -INF, q_vals)
                    .argmax(1)
                    .numpy()
                )
            obs_, rewards, dones, truncs, info = env.step(actions_)
            next_obs = torch.from_numpy(obs_)
            next_action_mask = np.array(list(info["action_mask"]))
            buffer.insert_step(
                obs,
                next_obs,
                torch.from_numpy(actions_).squeeze(0),
                rewards,
                dones,
                torch.Tensor(action_mask),
                torch.Tensor(next_action_mask),
            )
            obs = next_obs
            action_mask = next_action_mask

    # Train
    total_q_loss = 0.0
    if buffer.filled:
        q_net.train()
        if device.type != "cpu":
            q_net.to(device)

        total_q_loss = 0.0
        for _ in range(train_iters):
            prev_states, states, actions, rewards, dones, _, next_masks = buffer.sample(
                train_batch_size
            )

            # Move batch to device if applicable
            prev_states = prev_states.to(device=device)
            states = states.to(device=device)
            actions = actions.to(device=device)
            rewards = rewards.to(device=device)
            dones = dones.to(device=device)
            next_masks = next_masks.to(device=device)

            # Train q network
            q_opt.zero_grad()
            with torch.no_grad():
                next_actions = (
                    torch.where(next_masks == 1, -INF, q_net(states))
                    .argmax(1)
                    .squeeze(0)
                )
                q_target = rewards.unsqueeze(1) + discount * q_net_target(
                    states
                ).detach().gather(1, next_actions.unsqueeze(1)) * (
                    1.0 - dones.unsqueeze(1)
                )
            diff = q_net(prev_states).gather(1, actions.unsqueeze(1)) - q_target
            q_loss = (diff * diff).mean()
            q_loss.backward()
            q_opt.step()
            total_q_loss += q_loss.item()

        if device.type != "cpu":
            q_net.cpu()
        q_net.eval()

        # Evaluate
        eval_done = False
        with torch.no_grad():
            # Visualize
            reward_total = 0
            score_total = 0
            pred_reward_total = 0
            obs_, info = test_env.reset()
            eval_obs = torch.from_numpy(np.array(obs_)).float()
            eval_action_mask = np.array(list(info["action_mask"]))
            for _ in range(eval_steps):
                avg_entropy = 0.0
                steps_taken = 0
                score = 0
                for _ in range(max_eval_steps):
                    q_vals = q_net(eval_obs.unsqueeze(0)).squeeze()
                    action = (
                        torch.where(
                            torch.from_numpy(eval_action_mask) == 1, -INF, q_vals
                        )
                        .argmax(0)
                        .item()
                    )
                    score = test_env.score()
                    pred_reward_total += (
                        q_net(eval_obs.unsqueeze(0)).squeeze().max(0).values.item()
                    )
                    obs_, reward, eval_done, eval_trunc, info = test_env.step(action)
                    eval_obs = torch.from_numpy(np.array(obs_)).float()
                    eval_action_mask = np.array(list(info["action_mask"]))
                    steps_taken += 1
                    reward_total += reward
                    if eval_done or eval_trunc:
                        obs_, info = test_env.reset()
                        eval_obs = torch.from_numpy(np.array(obs_)).float()
                        eval_action_mask = np.array(list(info["action_mask"]))
                        break
                avg_entropy /= steps_taken
                score_total += score

        wandb.log(
            {
                "avg_eval_episode_reward": reward_total / eval_steps,
                "avg_eval_episode_predicted_reward": pred_reward_total / eval_steps,
                "avg_eval_episode_score": score_total / eval_steps,
                "avg_q_loss": total_q_loss / train_iters,
                "q_lr": q_opt.param_groups[-1]["lr"],
            }
        )

    # Update Q target
    if (step + 1) % 500 == 0:
        q_net_target.load_state_dict(q_net.state_dict())

    # Perform backups
    if (step + 1) % 100 == 0:
        torch.save(q_net, "temp/QNet.pt")

    eval_score = int(score_total / eval_steps)
    if best_eval < eval_score:
        print(f"New best score: {eval_score}")
        torch.save(q_net, "temp/QNetBest.pt")
        best_eval = eval_score

# Save artifacts
torch.save(q_net, "temp/QNet.pt")
q_net_artifact.add_file("temp/QNet.pt")
wandb.log_artifact(q_net_artifact)

wandb.finish()
