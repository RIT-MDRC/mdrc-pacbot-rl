"""
Uses a pretrained model with PPO on Pacman gym.

CLI Args:
    --eval: Run the last saved policy in the test environment, with visualization.
    --resume: Resume training from the last saved policy.
"""
import copy
import sys
from typing import Any, Union
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
from mdrc_pacbot_rl.pacman.gym import SelfAttentionPacmanGym as PacmanGym
from mdrc_pacbot_rl.utils import get_img_size, init_orthogonal

_: Any

# Hyperparameters
num_envs = 128  # Number of environments to step through at once during sampling.
train_steps = 32  # Number of steps to step through during sampling. Total # of samples is train_steps * num_envs/
iterations = 20000  # Number of sample/train iterations.
train_iters = 2  # Number of passes over the samples collected.
train_batch_size = 2048  # Minibatch size while training models.
discount = 0.5  # Discount factor applied to rewards.
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
        w, h = obs_shape[1:]
        self.cnn1 = nn.Conv2d(obs_shape[0], 4, 3, 2)
        h, w = get_img_size(h, w, self.cnn1)
        self.cnn2 = nn.Conv2d(4, 8, 3, 2)
        h, w = get_img_size(h, w, self.cnn2)
        flat_dim = h * w * self.cnn2.out_channels
        self.v_layer1 = nn.Linear(flat_dim, 256)
        self.v_layer2 = nn.Linear(256, 256)
        self.v_layer3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x = self.cnn1(input)
        x = self.relu(x)
        x = self.cnn2(x)
        x = x.flatten(1)
        x = self.relu(x)
        x = self.v_layer1(x)
        x = self.relu(x)
        x = self.v_layer2(x)
        x = self.relu(x)
        x = self.v_layer3(x)
        return x

class NewValueNet(nn.Module):
    def __init__(self, base_net: nn.Module, base_out: int, obs_shape: torch.Size):
        nn.Module.__init__(self)
        self.base_net = base_net
        for module in self.base_net.modules():
            module.requires_grad_(False)
        self.v_layer1 = nn.Linear(base_out, 1024)
        self.v_layer2 = nn.Linear(1024, 1024)
        self.v_layer3 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x = self.base_net(input)
        x = self.relu(x)
        x = self.v_layer1(x)
        x = self.relu(x)
        x = self.v_layer2(x)
        x = self.relu(x)
        x = self.v_layer3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, base_net: nn.Module, base_out: int, obs_shape: torch.Size, action_count: int):
        nn.Module.__init__(self)
        self.base_net = base_net
        for module in self.base_net.modules():
            module.requires_grad_(False)
        self.a_layer1 = nn.Linear(base_out, 1024)
        self.a_layer2 = nn.Linear(1024, 1024)
        self.a_layer3 = nn.Linear(1024, action_count)
        self.relu = nn.ReLU()
        self.logits = nn.LogSoftmax(1)
        init_orthogonal(self)

    def forward(self, input: torch.Tensor, mask: Union[torch.Tensor, np.ndarray]):
        x = self.base_net(input)
        x = self.relu(x)
        x = self.a_layer1(x)
        x = self.relu(x)
        x = self.a_layer2(x)
        x = self.relu(x)
        x = self.a_layer3(x)
        x = torch.where(~torch.tensor(mask).bool(), x, -1 * 10**8)
        return self.logits(x)

# Just to make it similar to the pretrained model
env = SyncVectorEnv([lambda: PacmanGym(random_start=True)] * num_envs)
test_env = PacmanGym(random_start=True)

# If evaluating, just run the eval env
if len(sys.argv) >= 2 and sys.argv[1] == "--eval":
    test_env = PacmanGym(random_start=True, render_mode="human")
    p_net = torch.load("temp/PNet.pt")
    obs_shape = test_env.observation_space.shape
    if not obs_shape:
        raise RuntimeError("Observation space doesn't have shape")
    obs_size = torch.Size(obs_shape)
    with torch.no_grad():
        obs_, info = test_env.reset()
        action_mask = np.array(list(info["action_mask"]))
        obs = torch.from_numpy(obs_).float()
        for _ in range(max_eval_steps):
            distr = Categorical(logits=p_net(obs.unsqueeze(0), action_mask).squeeze())
            action = distr.sample().item()
            obs_, reward, done, _, info = test_env.step(action)
            print(distr.probs, reward)
            obs = torch.from_numpy(obs_).float()
            action_mask = np.array(list(info["action_mask"]))
            if done:
                break
    quit()

wandb.init(
    project="pacbot",
    entity="mdrc-pacbot",
    config={
        "experiment": "pretrained ppo",
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
base_net = torch.load("temp/BaseNet.pt")
base_net.v_layer3 = nn.Identity()
v_net = NewValueNet(copy.deepcopy(base_net), 256, torch.Size(obs_shape))
p_net = PolicyNet(base_net, 256, obs_size, int(act_space.n))
v_opt = torch.optim.Adam(v_net.parameters(), lr=v_lr)
p_opt = torch.optim.Adam(p_net.parameters(), lr=p_lr)
buffer = RolloutBuffer(
    obs_size,
    torch.Size((1,)),
    torch.Size((5,)),
    torch.int,
    num_envs,
    train_steps,
    device,
)

obs_, info = env.reset()
obs = torch.from_numpy(obs_)
action_mask = np.array(list(info["action_mask"]))
smooth_eval_score = 0.0
last_eval_score = 0.0
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
                torch.from_numpy(action_mask),
            )
            action_mask = np.array(list(info["action_mask"]))
            obs = torch.from_numpy(obs_)
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
        eval_obs = torch.from_numpy(obs_).float()
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
                obs_, reward, eval_done, eval_trunc, _ = test_env.step(action)
                eval_obs = torch.from_numpy(obs_).float()
                steps_taken += 1
                reward_total += reward
                avg_entropy += distr.entropy()
                if eval_done or eval_trunc:
                    obs_, info = test_env.reset()
                    eval_obs = torch.from_numpy(obs_).float()
                    eval_action_mask = np.array(list(info["action_mask"]))
                    break
            avg_entropy /= steps_taken
            entropy_total += avg_entropy
            score_total += score

    wandb.log(
        {
            "avg_eval_episode_reward": reward_total / eval_steps,
            "avg_eval_episode_predicted_reward": pred_reward_total / eval_steps,
            "avg_eval_episode_score": score_total / eval_steps,
            "avg_eval_entropy": entropy_total / eval_steps,
            "avg_v_loss": total_v_loss / train_iters,
            "avg_p_loss": total_p_loss / train_iters,
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

            last_eval_score = smooth_eval_score
            last_discount = discount
        
        torch.save(v_net, "temp/VNet.pt")
        torch.save(p_net, "temp/PNet.pt")

# Save artifacts
torch.save(v_net, "temp/VNet.pt")
v_net_artifact.add_file("temp/VNet.pt")
wandb.log_artifact(v_net_artifact)

torch.save(p_net, "temp/PNet.pt")
p_net_artifact.add_file("temp/PNet.pt")
wandb.log_artifact(p_net_artifact, "temp/PNet.pt")

wandb.finish()
