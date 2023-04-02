"""
Baseline for PPO on Pacman gym, using self attention.

CLI Args:
    --eval: Run the last saved policy in the test environment, with visualization.
    --resume: Resume training from the last saved policy.
"""
import json
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
from mdrc_pacbot_rl.algorithms.rollout_buffer import RolloutBuffer
from mdrc_pacbot_rl.algorithms.self_attention import AttnBlock, gen_pos_encoding
from mdrc_pacbot_rl.pacman.gym import NaivePacmanGym as PacmanGym
from mdrc_pacbot_rl.utils import copy_params, get_img_size, init_orthogonal

_: Any

# Hyperparameters
num_envs = 32  # Number of environments to step through at once during sampling.
train_steps = 32  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs/
iterations = 4000  # Number of sample/train iterations.
train_iters = 2  # Number of passes over the samples collected.
train_batch_size = 512  # Minibatch size while training models.
discount = 0.9  # Discount factor applied to rewards.
lambda_ = 0.95  # Lambda for GAE.
epsilon = 0.2  # Epsilon for importance sample clipping.
eval_steps = 8  # Number of eval runs to average over.
max_eval_steps = 300  # Max number of steps to take during each eval run.
v_lr = 0.01  # Learning rate of the value net.
p_lr = 0.001  # Learning rate of the policy net.
emb_dim = (
    8  # Size of the input embeddings to learn, not including positional component.
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


class ValueNet(nn.Module):
    def __init__(self, obs_shape: torch.Size, pos_encoding: torch.Tensor, emb_dim: int):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            BaseNet(obs_shape, 256, pos_encoding, emb_dim), nn.ReLU(), nn.Linear(256, 1)
        )
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        return self.net(input)


class PolicyNet(nn.Module):
    def __init__(
        self,
        obs_shape: torch.Size,
        action_count: int,
        pos_encoding: torch.Tensor,
        emb_dim: int,
    ):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            BaseNet(obs_shape, 256, pos_encoding, emb_dim),
            nn.ReLU(),
            nn.Linear(256, action_count),
            nn.LogSoftmax(1),
        )
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        return self.net(input)


env = SyncVectorEnv([lambda: PacmanGym() for _ in range(num_envs)])
test_env = PacmanGym()

# If evaluating, just run the eval env
if len(sys.argv) >= 2 and sys.argv[1] == "--eval":
    test_env = PacmanGym(render_mode="human", random_start=True)
    p_net = torch.load("temp/PNet.pt")
    obs_shape = test_env.observation_space.shape
    if not obs_shape:
        raise RuntimeError("Observation space doesn't have shape")
    obs_size = torch.Size(obs_shape)
    _, obs_mask = load_pos_and_mask(obs_size[1], obs_size[2])
    with torch.no_grad():
        obs = process_obs(test_env.reset()[0][np.newaxis, ...], obs_mask)
        for _ in range(max_eval_steps):
            distr = Categorical(logits=p_net(obs.unsqueeze(0)).squeeze())
            action = distr.sample().item()
            obs_, reward, done, _, _ = test_env.step(action)
            obs = process_obs(obs_[np.newaxis, ...], obs_mask)
            if done:
                break
    quit()

wandb.init(
    project="pacbot",
    entity="mdrc-pacbot",
    config={
        "experiment": "baseline ppo with self attention",
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
        "emb_dim": emb_dim,
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
_pos_encoding, obs_mask = load_pos_and_mask(obs_size[1], obs_size[2])
pos_encoding = torch.from_numpy(_pos_encoding).squeeze(0)
obs_size = torch.Size([pos_encoding.shape[0], obs_size[0]])
v_net = ValueNet(obs_size, pos_encoding, emb_dim)
p_net = PolicyNet(obs_size, int(act_space.n), pos_encoding, emb_dim)
v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)
v_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    v_opt, "max", patience=20, factor=0.5, min_lr=0.0005
)
p_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    p_opt, "max", patience=20, factor=0.5, min_lr=0.00005
)
buffer = RolloutBuffer(
    torch.Size(obs_size),
    torch.Size((1,)),
    torch.Size((5,)),
    torch.int,
    num_envs,
    train_steps,
    device,
)

obs = process_obs(env.reset()[0], obs_mask)
smooth_eval_score = 90.0
last_eval_score = 0.0
last_v_schedule = v_scheduler.state_dict()
last_p_schedule = p_scheduler.state_dict()
for step in tqdm(range(iterations), position=0):
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
            obs = process_obs(obs_, obs_mask)
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
    )
    buffer.clear()

    # Evaluate
    eval_done = False
    with torch.no_grad():
        # Visualize
        reward_total = 0
        pred_reward_total = 0
        score_total = 0
        entropy_total = 0.0
        eval_obs = process_obs(test_env.reset()[0][np.newaxis, ...], obs_mask).squeeze(
            0
        )
        for _ in range(eval_steps):
            avg_entropy = 0.0
            steps_taken = 0
            score = 0
            for _ in range(max_eval_steps):
                distr = Categorical(logits=p_net(eval_obs.unsqueeze(0)).squeeze())
                action = distr.sample().item()
                score = test_env.score()
                pred_reward_total += v_net(eval_obs.unsqueeze(0)).squeeze().item()
                obs_, reward, eval_done, eval_trunc, _ = test_env.step(action)
                eval_obs = process_obs(obs_[np.newaxis, ...], obs_mask).squeeze(0)
                steps_taken += 1
                reward_total += reward
                avg_entropy += distr.entropy()
                if eval_done or eval_trunc:
                    eval_obs = process_obs(
                        test_env.reset()[0][np.newaxis, ...], obs_mask
                    ).squeeze(0)
                    break
            avg_entropy /= steps_taken
            entropy_total += avg_entropy
            score_total += score

    # Update learning rate
    smooth_update = 0.05
    smooth_eval_score += smooth_update * (score_total / eval_steps - smooth_eval_score)
    v_scheduler.step(smooth_eval_score)
    p_scheduler.step(smooth_eval_score)

    wandb.log(
        {
            "avg_eval_episode_reward": reward_total / eval_steps,
            "avg_eval_episode_predicted_reward": pred_reward_total / eval_steps,
            "avg_eval_episode_score": score_total / eval_steps,
            "avg_eval_entropy": entropy_total / eval_steps,
            "avg_v_loss": total_v_loss / train_iters,
            "avg_p_loss": total_p_loss / train_iters,
            "v_lr": v_opt.param_groups[-1]["lr"],
            "p_lr": p_opt.param_groups[-1]["lr"],
            "smooth_eval_score": smooth_eval_score,
        }
    )

    # Perform backups
    if (step + 1) % 100 == 0:
        if smooth_eval_score < last_eval_score:
            v_net.load_state_dict(torch.load("temp/VNet.pt").state_dict())
            p_net.load_state_dict(torch.load("temp/PNet.pt").state_dict())
            v_scheduler.load_state_dict(last_v_schedule)
            p_scheduler.load_state_dict(last_p_schedule)
        else:
            torch.save(v_net, "temp/VNet.pt")
            torch.save(p_net, "temp/PNet.pt")

            last_eval_score = smooth_eval_score
            last_v_schedule = v_scheduler.state_dict()
            last_p_schedule = p_scheduler.state_dict()

# Save artifacts
torch.save(v_net, "temp/VNet.pt")
v_net_artifact.add_file("temp/VNet.pt")
wandb.log_artifact(v_net_artifact)

torch.save(p_net, "temp/PNet.pt")
p_net_artifact.add_file("temp/PNet.pt")
wandb.log_artifact(p_net_artifact, "temp/PNet.pt")

wandb.finish()
