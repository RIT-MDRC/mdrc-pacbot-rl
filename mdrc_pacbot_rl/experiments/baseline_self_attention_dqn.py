"""
Baseline for PPO on Pacman gym, using DQN.

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
from torch.distributions import Categorical
from tqdm import tqdm

from mdrc_pacbot_rl.algorithms.ppo import train_ppo
from mdrc_pacbot_rl.algorithms.replay_buffer import ReplayBuffer
from mdrc_pacbot_rl.algorithms.rollout_buffer import RolloutBuffer
from mdrc_pacbot_rl.algorithms.self_attention import AttnBlock, gen_pos_encoding
from mdrc_pacbot_rl.pacman.gym import NaivePacmanGym as PacmanGym
from mdrc_pacbot_rl.utils import copy_params, get_img_size, init_orthogonal

_: Any

# Hyperparameters
num_envs = 32  # Number of environments to step through at once during sampling.
train_steps = 2  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs/
iterations = 10000  # Number of sample/train iterations.
train_iters = 4  # Number of passes over the samples collected.
train_batch_size = 128  # Minibatch size while training models.
discount = 0.0  # Discount factor applied to rewards.
q_epsilon = 0.5  # Epsilon for epsilon greedy strategy. This gets annealed over time.
eval_steps = 1  # Number of eval runs to average over.
max_eval_steps = 300  # Max number of steps to take during each eval run.
q_lr = 0.0001  # Learning rate of the q net.
warmup_steps = 500 # For the first n number of steps, we will only sample randomly.
emb_dim = (
    16  # Size of the input embeddings to learn, not including positional component.
)
device = torch.device("cuda")


def process_obs(input: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """
    Converts a BxCxWxH array, where B is batch size, C is channels, W is width,
    and H is height, into a BxLxC array, where L is W * H - # of masked
    elements. This significantly reduces the amount of data fed into the
    network.

    Arguments:
        input: An array of size BxCxWxH representing a raw observation.
        mask: A binary int array of size WxH, where a 1 means throw away that cell.
    """
    batch_size, channels, w, h = input.shape
    masked_count = mask.sum()
    l = w * h - masked_count
    batched_mask = np.tile(
        mask[np.newaxis, np.newaxis, ...], [batch_size, channels, 1, 1]
    )
    masked_input: np.ma.masked_array = np.ma.masked_array(input, mask=batched_mask)
    flattened = np.reshape(
        masked_input.compressed(), [batch_size, channels, l]
    ).swapaxes(1, 2)
    return torch.from_numpy(flattened).float()


def load_pos_and_mask(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the positional encoding and mask from the computed data folder.
    """
    node_coords = json.load(open("computed_data/node_coords.json", "r"))
    node_embeddings = np.load("computed_data/node_embeddings.npy")
    emb_dim = node_embeddings.shape[1]
    encodings = np.zeros([width, height, emb_dim])
    mask = np.ones([width, height], dtype=np.int0)
    for i, coord in enumerate(node_coords):
        x, y = coord
        mask[x][y] = 0
        encodings[x][y] = node_embeddings[i]
    encodings = (
        process_obs(np.moveaxis(encodings, -1, 0)[np.newaxis, ...], mask=mask)
        .squeeze(0)
        .numpy()
    )
    return encodings, mask


class BaseNet(nn.Module):
    """
    Base network architecture used by both value and policy networks.
    """

    def __init__(
        self,
        input_shape: torch.Size,
        out_features: int,
        pos_encoding: torch.Tensor,
        emb_dim: int,
    ):
        nn.Module.__init__(self)
        self.input_size = input_shape[0]
        self.emb_dim = emb_dim
        pos_dim = pos_encoding.shape[1]
        self.expand = nn.Sequential(
            nn.Linear(input_shape[1], self.emb_dim * 4),
            nn.ReLU(),
            nn.Linear(self.emb_dim * 4, self.emb_dim),
        )
        self.pos = nn.Parameter(pos_encoding, False)
        self.attn = AttnBlock(self.emb_dim + pos_dim, 4)
        self.linear = nn.Linear(self.emb_dim + pos_dim, out_features)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]

        # We learn a representation for each cell based on its contents.
        x = self.expand(input.flatten(0, 1)).reshape(
            batch_size, self.input_size, self.emb_dim
        )

        # A graph based positional encoding gets appended to our representation
        # embeddings, and self attention is performed. Afterwards, we maxpool
        # the results into a single vector.
        positional = self.pos.repeat([batch_size, 1, 1])
        x = torch.concatenate([x, positional], 2)
        x = self.attn(x)
        x = torch.max(x, 1).values

        # Finally, we run our value through a linear layer.
        x = self.relu(x)
        x = self.linear(x)
        return x


class QNet(nn.Module):
    def __init__(
        self,
        obs_shape: torch.Size,
        action_count: int,
        pos_encoding: torch.Tensor,
        emb_dim: int,
    ):
        nn.Module.__init__(self)
        self.net = BaseNet(obs_shape, 256, pos_encoding, emb_dim)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(256, 256)
        self.advantage = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_count)
        )
        self.value = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
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


env = SyncVectorEnv([lambda: PacmanGym() for _ in range(num_envs)])
test_env = PacmanGym()

# If evaluating, just run the eval env
if len(sys.argv) >= 2 and sys.argv[1] == "--eval":
    test_env = PacmanGym(render_mode="human", random_start=True)
    q_net = torch.load("temp/QNet.pt")
    obs_shape = test_env.observation_space.shape
    if not obs_shape:
        raise RuntimeError("Observation space doesn't have shape")
    obs_size = torch.Size(obs_shape)
    _, obs_mask = load_pos_and_mask(obs_size[1], obs_size[2])
    with torch.no_grad():
        obs = process_obs(test_env.reset()[0][np.newaxis, ...], obs_mask)
        for _ in range(max_eval_steps):
            q_vals = q_net(obs.unsqueeze(0)).squeeze()
            action = q_vals.argmax(0).item()
            obs_, reward, done, _, _ = test_env.step(action)
            print(q_vals, reward)
            obs = process_obs(obs_[np.newaxis, ...], obs_mask)
            if done:
                break
    quit()

wandb.init(
    project="pacbot",
    entity="mdrc-pacbot",
    config={
        "experiment": "baseline dqn with self attention",
        "num_envs": num_envs,
        "train_steps": train_steps,
        "train_iters": train_iters,
        "train_batch_size": train_batch_size,
        "discount": discount,
        "q_epsilon": q_epsilon,
        "max_eval_steps": max_eval_steps,
        "q_lr": q_lr,
        "emb_dim": emb_dim,
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
_pos_encoding, obs_mask = load_pos_and_mask(obs_size[1], obs_size[2])
pos_encoding = torch.from_numpy(_pos_encoding).squeeze(0)
obs_size = torch.Size([pos_encoding.shape[0], obs_size[0]])
q_net = QNet(obs_size, int(act_space.n), pos_encoding, emb_dim)
q_net_target = copy.deepcopy(q_net)
q_net_target.to(device)
q_opt = torch.optim.Adam(q_net.parameters(), lr=q_lr)
buffer = ReplayBuffer(
    torch.Size(obs_size),
    10000,
)

obs = process_obs(env.reset()[0], obs_mask)
for step in tqdm(range(iterations), position=0):
    percent_done = step / iterations

    # Collect experience
    with torch.no_grad():
        for _ in tqdm(range(train_steps), position=1):
            q_vals = q_net(obs)
            if random.random() < q_epsilon * (1.0 - percent_done) or step < warmup_steps:
                actions_ = np.random.randint(0, act_space.n, [num_envs])
            else:
                actions_ = q_vals.argmax(1).numpy()
            obs_, rewards, dones, truncs, _ = env.step(actions_)
            next_obs = process_obs(obs_, obs_mask)
            buffer.insert_step(
                obs,
                next_obs,
                torch.from_numpy(actions_).squeeze(0),
                rewards,
                dones,
            )
            obs = next_obs

    # Train
    total_q_loss = 0.0
    if buffer.filled:
        q_net.train()
        if device.type != "cpu":
            q_net.to(device)

        total_q_loss = 0.0
        for _ in tqdm(range(train_iters), position=1):
            prev_states, states, actions, rewards, dones = buffer.sample(train_batch_size)

            # Move batch to device if applicable
            prev_states = prev_states.to(device=device)
            states = prev_states.to(device=device)
            actions = actions.to(device=device)
            rewards = rewards.to(device=device)
            dones = dones.to(device=device)

            # Train q network
            q_opt.zero_grad()
            next_actions = q_net(states).argmax(1).squeeze(0)
            q_target = rewards + discount * q_net_target(states).detach().index_select(1, next_actions) * (1.0 - dones)
            diff = q_net(prev_states).index_select(1, actions) - q_target
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
            pred_reward_total = 0
            score_total = 0
            eval_obs = process_obs(test_env.reset()[0][np.newaxis, ...], obs_mask).squeeze(
                0
            )
            for _ in range(eval_steps):
                avg_entropy = 0.0
                steps_taken = 0
                score = 0
                for _ in range(max_eval_steps):
                    q_vals = q_net(eval_obs.unsqueeze(0)).squeeze()
                    action = q_vals.argmax(0).item()
                    score = test_env.score()
                    pred_reward_total += q_net(eval_obs.unsqueeze(0)).squeeze().max(0).values.item()
                    obs_, reward, eval_done, eval_trunc, _ = test_env.step(action)
                    eval_obs = process_obs(obs_[np.newaxis, ...], obs_mask).squeeze(0)
                    steps_taken += 1
                    reward_total += reward
                    if eval_done or eval_trunc:
                        eval_obs = process_obs(
                            test_env.reset()[0][np.newaxis, ...], obs_mask
                        ).squeeze(0)
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
    if (step + 1) % 100 == 0:
        q_net_target.load_state_dict(q_net.state_dict())

    # Perform backups
    if (step + 1) % 100 == 0:
        torch.save(q_net, "temp/QNet.pt")

# Save artifacts
torch.save(q_net, "temp/QNet.pt")
q_net_artifact.add_file("temp/QNet.pt")
wandb.log_artifact(q_net_artifact)

wandb.finish()
