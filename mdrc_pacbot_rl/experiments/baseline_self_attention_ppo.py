"""
Baseline for PPO on Pacman gym, using self attention.

CLI Args:
    --eval: Run the last saved policy in the test environment, with visualization.
    --resume: Resume training from the last saved policy.
"""
import json
import sys
from typing import Any, Tuple, Union

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
from mdrc_pacbot_rl.algorithms.self_attention import AttnBlock
from mdrc_pacbot_rl.pacman.gym import NaivePacmanGym as PacmanGym
from mdrc_pacbot_rl.utils import init_xavier

_: Any

# Hyperparameters
num_envs = 32  # Number of environments to step through at once during sampling.
train_steps = 32  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs/
iterations = 20000  # Number of sample/train iterations.
train_iters = 1  # Number of passes over the samples collected.
train_batch_size = 512  # Minibatch size while training models.
discount = 0.0  # Discount factor applied to rewards.
lambda_ = 0.0  # Lambda for GAE.
epsilon = 0.2  # Epsilon for importance sample clipping.
eval_steps = 4  # Number of eval runs to average over.
max_eval_steps = 300  # Max number of steps to take during each eval run.
warmup_steps = 4000  # Number of steps before dropping the learning rates.
emb_dim = (
    128  # Size of the input embeddings to learn, not including positional component.
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


def load_pos_and_mask(
    width: int, height: int, emb_dim: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the positional encoding and mask from the computed data folder.
    """
    node_coords = json.load(open("computed_data/node_coords.json", "r"))
    node_embeddings = np.load("computed_data/graph_lap_eigvecs.npy")[:emb_dim].swapaxes(
        0, 1
    )  # node_embeddings.npy")
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


def get_lr(step: int, emb_dim: int, warmup_steps: int):
    return emb_dim ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def set_opt_lr(opt: torch.optim.Optimizer, lr):
    for group in opt.param_groups:
        group["lr"] = lr


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
        self.expand = nn.Sequential(
            nn.Linear(input_shape[1], self.emb_dim * 4),
            nn.ReLU(),
            nn.Linear(self.emb_dim * 4, self.emb_dim),
        )
        self.pos = nn.Parameter(pos_encoding, False)
        self.attn = AttnBlock(self.emb_dim, input_shape[0], 4)
        self.linear = nn.Linear(self.emb_dim, out_features)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]

        # We learn a representation for each cell based on its contents.
        x = self.expand(input.flatten(0, 1)).reshape(
            batch_size, self.input_size, self.emb_dim
        )

        # A graph based positional encoding gets added to our representation
        # embeddings, and self attention is performed. Afterwards, we maxpool
        # the results into a single vector.
        positional = self.pos.repeat([batch_size, 1, 1])
        x = x + positional
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
        init_xavier(self)

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
        )
        self.softmax = nn.LogSoftmax(1)
        init_xavier(self)

    def forward(self, input: torch.Tensor, mask: Union[torch.Tensor, np.ndarray]):
        x = self.net(input) * (1 - mask) + mask * -1 * 10**8
        return self.softmax(x)


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
    _, obs_mask = load_pos_and_mask(obs_size[1], obs_size[2], emb_dim)
    with torch.no_grad():
        obs_, info = test_env.reset()
        action_mask = np.array(list(info["action_mask"]))
        obs = process_obs(obs_[np.newaxis, ...], obs_mask)
        for _ in range(max_eval_steps):
            distr = Categorical(logits=p_net(obs.unsqueeze(0), action_mask).squeeze())
            action = distr.sample().item()
            print(distr.probs)
            obs_, reward, done, _, info = test_env.step(action)
            obs = process_obs(obs_[np.newaxis, ...], obs_mask)
            action_mask = np.array(list(info["action_mask"]))
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
_pos_encoding, obs_mask = load_pos_and_mask(obs_size[1], obs_size[2], emb_dim)
pos_encoding = torch.from_numpy(_pos_encoding).squeeze(0)
obs_size = torch.Size([pos_encoding.shape[0], obs_size[0]])
v_net = ValueNet(obs_size, pos_encoding, emb_dim)
p_net = PolicyNet(obs_size, int(act_space.n), pos_encoding, emb_dim)
v_opt = torch.optim.Adam(v_net.parameters(), lr=0.0)
p_opt = torch.optim.Adam(p_net.parameters(), lr=0.0)
buffer = RolloutBuffer(
    torch.Size(obs_size),
    torch.Size((1,)),
    torch.Size((5,)),
    torch.int,
    num_envs,
    train_steps,
    device,
)

obs_, info = env.reset()
obs = process_obs(obs_, obs_mask)
action_mask = np.array(list(info["action_mask"]))
smooth_eval_score = 0.0
last_eval_score = 0.0
anneal_discount_after = 1000
last_discount = discount
discount_change = 0.00005
for step in tqdm(range(iterations), position=0):
    # Collect experience
    with torch.no_grad():
        for _ in tqdm(range(train_steps), position=1):
            action_probs = p_net(obs, action_mask)
            actions = Categorical(logits=action_probs).sample().numpy()
            obs_, rewards, dones, truncs, info = env.step(actions)
            buffer.insert_step(
                obs,
                torch.from_numpy(actions).unsqueeze(-1),
                action_probs,
                rewards,
                dones,
                truncs,
                torch.Tensor(action_mask),
            )
            obs = process_obs(obs_, obs_mask)
            action_mask = np.array(list(info["action_mask"]))
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
        obs_, info = test_env.reset()
        eval_obs = process_obs(obs_[np.newaxis, ...], obs_mask).squeeze(0)
        eval_action_mask = np.array(list(info["action_mask"]))
        for _ in range(eval_steps):
            avg_entropy = 0.0
            steps_taken = 0
            score = 0
            for _ in range(max_eval_steps):
                distr = Categorical(
                    logits=p_net(eval_obs.unsqueeze(0), eval_action_mask).squeeze()
                )
                action = distr.sample().item()
                score = test_env.score()
                pred_reward_total += v_net(eval_obs.unsqueeze(0)).squeeze().item()
                obs_, reward, eval_done, eval_trunc, info = test_env.step(action)
                eval_obs = process_obs(obs_[np.newaxis, ...], obs_mask).squeeze(0)
                eval_action_mask = np.array(list(info["action_mask"]))
                steps_taken += 1
                reward_total += reward
                avg_entropy += distr.entropy()
                if eval_done or eval_trunc:
                    obs_, info = test_env.reset()
                    eval_obs = process_obs(obs_[np.newaxis, ...], obs_mask).squeeze(0)
                    eval_action_mask = np.array(list(info["action_mask"]))
                    break
            avg_entropy /= steps_taken
            entropy_total += avg_entropy
            score_total += score

    # Update learning rate
    set_opt_lr(v_opt, get_lr(step + 1, emb_dim, warmup_steps) * 1.0)
    set_opt_lr(p_opt, get_lr(step + 1, emb_dim, warmup_steps) * 0.1)

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
    smooth_update = 0.04
    smooth_eval_score += smooth_update * (score_total / eval_steps - smooth_eval_score)

    if (step + 1) % 10 == 0:
        if smooth_eval_score >= last_eval_score:
            torch.save(v_net, "temp/VNetBest.pt")
            torch.save(p_net, "temp/PNetBest.pt")
        torch.save(v_net, "temp/VNet.pt")
        torch.save(p_net, "temp/PNet.pt")

        last_eval_score = smooth_eval_score
        last_discount = discount

# Save artifacts
torch.save(v_net, "temp/VNet.pt")
v_net_artifact.add_file("temp/VNet.pt")
wandb.log_artifact(v_net_artifact)

torch.save(p_net, "temp/PNet.pt")
p_net_artifact.add_file("temp/PNet.pt")
wandb.log_artifact(p_net_artifact, "temp/PNet.pt")

wandb.finish()
