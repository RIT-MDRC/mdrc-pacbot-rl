"""
Experiment for testing if the agent can learn to run from a ghost using self
attention. The positional encoding here is just x and y, the full environment
will use the calculated embeddings.

CLI Args:
    --eval: Run the last saved policy in the test environment, with
    visualization.
"""
import copy
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

from mdrc_pacbot_rl.algorithms.ppo import train_ppo
from mdrc_pacbot_rl.algorithms.rollout_buffer import RolloutBuffer
from mdrc_pacbot_rl.algorithms.self_attention import AttnBlock, gen_pos_encoding
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
lambda_ = 0.7  # Lambda for GAE.
epsilon = 0.2  # Epsilon for importance sample clipping.
eval_steps = 8  # Number of eval runs to average over.
max_eval_steps = 300  # Max number of steps to take during each eval run.
v_lr = 0.001  # Learning rate of the value net.
p_lr = 0.0001  # Learning rate of the policy net.
device = torch.device("cuda")


class ValueNet(nn.Module):
    def __init__(self, obs_shape: torch.Size):
        nn.Module.__init__(self)
        self.emb_dim = 16
        w, h = obs_shape[1:]
        self.cnn1 = nn.Conv2d(obs_shape[0], 8, 2)
        h, w = get_img_size(h, w, self.cnn1)
        self.cnn2 = nn.Conv2d(8, self.emb_dim - 2, 2)
        h, w = get_img_size(h, w, self.cnn2)
        self.pos = nn.Parameter(gen_pos_encoding(w, h), False)
        self.input_size = w * h
        self.attn = AttnBlock(self.emb_dim, 4)
        self.v_layer1 = nn.Linear(self.emb_dim, 256)
        self.v_layer2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x = self.cnn1(input)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.relu(x)

        batch_size = input.shape[0]
        positional = self.pos.repeat([batch_size, 1, 1, 1])
        x = (
            torch.concatenate([x, positional], 1)
            .reshape([batch_size, self.emb_dim, self.input_size])
            .swapaxes(1, 2)
        )

        x = self.attn(x)
        x = torch.max(x, 1).values

        x = self.relu(x)
        x = self.v_layer1(x)
        x = self.relu(x)
        x = self.v_layer2(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, obs_shape: torch.Size, action_count: int):
        nn.Module.__init__(self)
        self.emb_dim = 16
        w, h = obs_shape[1:]
        self.cnn1 = nn.Conv2d(obs_shape[0], 8, 2)
        h, w = get_img_size(h, w, self.cnn1)
        self.cnn2 = nn.Conv2d(8, self.emb_dim - 2, 2)
        h, w = get_img_size(h, w, self.cnn2)
        self.pos = nn.Parameter(gen_pos_encoding(w, h), False)
        self.input_size = w * h
        self.attn = AttnBlock(self.emb_dim, 4)
        self.a_layer1 = nn.Linear(self.emb_dim, 256)
        self.a_layer2 = nn.Linear(256, action_count)
        self.relu = nn.ReLU()
        self.logits = nn.LogSoftmax(1)

        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x = self.cnn1(input)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.relu(x)

        batch_size = input.shape[0]
        positional = self.pos.repeat([batch_size, 1, 1, 1])
        x = (
            torch.concatenate([x, positional], 1)
            .reshape([batch_size, self.emb_dim, self.input_size])
            .swapaxes(1, 2)
        )

        x = self.attn(x)
        x = torch.max(x, 1).values

        x = self.relu(x)
        x = self.a_layer1(x)
        x = self.relu(x)
        x = self.a_layer2(x)
        x = self.logits(x)
        return x


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
        "experiment": "run away ppo with self attention",
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
v_net = ValueNet(obs_size)
p_net = PolicyNet(obs_size, int(act_space.n))
v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)
buffer = RolloutBuffer(
    torch.Size(obs_size),
    torch.Size((1,)),
    torch.Size((5,)),
    torch.int,
    num_envs,
    train_steps,
    device,
)

obs = torch.Tensor(env.reset()[0])
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
            obs = torch.tensor(obs_)
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
        reward_total = 0
        pred_reward_total = 0
        entropy_total = 0.0
        eval_obs = torch.tensor(test_env.reset()[0]).float()
        for _ in range(eval_steps):
            avg_entropy = 0.0
            steps_taken = 0
            score = 0
            for _ in range(max_eval_steps):
                distr = Categorical(logits=p_net(eval_obs.unsqueeze(0)).squeeze())
                action = distr.sample().item()
                pred_reward_total += v_net(eval_obs.unsqueeze(0)).squeeze().item()
                obs_, reward, eval_done, eval_trunc, _ = test_env.step(action)
                eval_obs = torch.tensor(obs_).float()
                steps_taken += 1
                reward_total += reward
                avg_entropy += distr.entropy()
                if eval_done or eval_trunc:
                    eval_obs = torch.tensor(test_env.reset()[0]).float()
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
