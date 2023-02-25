"""
Experiment for checking that PPO is working.
"""
from functools import reduce
from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical
from matplotlib import pyplot as plt
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from tqdm import tqdm

from mdrc_pacbot_rl.algorithms.rollout_buffer import RolloutBuffer

# Constants (for now)
train_steps = 128
iterations = 50000 // train_steps
train_iters = 2
train_batch_size = 64
discount = 1.0
lambda_ = 0.95
epsilon = 0.2
device = torch.device("cpu")


def copy_params(src: nn.Module, dest: nn.Module):
    """
    Copies params from one model to another.
    """
    with torch.no_grad():
        for dest_, src_ in zip(dest.parameters(), src.parameters()):
            dest_.data.copy_(src_.data)


class ValueNet(nn.Module):
    def __init__(self, obs_shape: torch.Size):
        nn.Module.__init__(self)
        flat_obs_dim = reduce(lambda e1, e2: e1 * e2, obs_shape, 1)
        self.v_layer1 = nn.Linear(flat_obs_dim, 32)
        self.v_layer2 = nn.Linear(32, 32)
        self.v_layer3 = nn.Linear(32, 1)
        self.l_relu = nn.LeakyReLU(0.01)

    def forward(self, input: torch.Tensor):
        x = self.v_layer1(input.flatten(1))
        x = self.l_relu(x)
        x = self.v_layer2(x)
        x = self.l_relu(x)
        x = self.v_layer3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, obs_shape: torch.Size, action_count: int):
        nn.Module.__init__(self)
        flat_obs_dim = reduce(lambda e1, e2: e1 * e2, obs_shape, 1)
        self.a_layer1 = nn.Linear(flat_obs_dim, 32)
        self.a_layer2 = nn.Linear(32, 32)
        self.a_layer3 = nn.Linear(32, action_count)
        self.l_relu = nn.LeakyReLU(0.01)
        self.logits = nn.LogSoftmax(1)

    def forward(self, input: torch.Tensor):
        x = self.a_layer1(input.flatten(1))
        x = self.l_relu(x)
        x = self.a_layer2(x)
        x = self.l_relu(x)
        x = self.a_layer3(x)
        x = self.logits(x)
        return x


env = CartPoleEnv()
reward_totals = []

# Init PPO
obs_space = env.observation_space
act_space = env.action_space
v_net = ValueNet(obs_space.shape)
p_net = PolicyNet(obs_space.shape, act_space.n)
p_net_old = PolicyNet(obs_space.shape, act_space.n)
p_net_old.eval()
v_opt = torch.optim.Adam(v_net.parameters(), lr=0.001)
p_opt = torch.optim.Adam(p_net.parameters(), lr=0.003)
buffer = RolloutBuffer(
    obs_space.shape, torch.Size((act_space.n,)), torch.int, 1, train_steps, device
)

obs = torch.Tensor(env.reset()[0])
done = False
for _ in tqdm(range(iterations), position=0):
    # Collect experience
    with torch.no_grad():
        for _ in tqdm(range(train_steps), position=1):
            action = (
                Categorical(logits=p_net(obs.unsqueeze(0)).squeeze()).sample().item()
            )
            obs_, reward, done, _, _ = env.step(action)
            buffer.insert_step(
                obs.unsqueeze(0), torch.Tensor([action]), [reward], [done]
            )
            obs = torch.Tensor(obs_)
            if done:
                obs = torch.Tensor(env.reset()[0])
                done = False
        buffer.insert_final_step(obs.unsqueeze(0))

    # Train
    p_net.train()
    v_net.train()
    copy_params(p_net, p_net_old)

    for _ in range(train_iters):
        batches = buffer.samples(train_batch_size, discount, lambda_, v_net)
        for prev_states, _, actions, _, rewards_to_go, advantages, _ in batches:
            # Train policy network
            old_log_probs = p_net_old(prev_states)
            p_opt.zero_grad()
            new_log_probs = p_net(prev_states)
            term1: torch.Tensor = (new_log_probs - old_log_probs).exp() * advantages
            term2: torch.Tensor = (1.0 + epsilon * advantages.sign()) * advantages
            p_loss = -(term1.minimum(term2).mean())
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
    obs = torch.Tensor(env.reset()[0])
    done = False
    with torch.no_grad():
        reward_total = 0
        for _ in range(8):
            for _ in range(16):
                action = (
                    Categorical(logits=p_net(obs.unsqueeze(0)).squeeze())
                    .sample()
                    .item()
                )
                obs_, reward, done, _, _ = env.step(action)
                obs = torch.Tensor(obs_)
                reward_total += reward
                if done:
                    obs = torch.Tensor(env.reset()[0])
                    done = False
                    break
        reward_totals.append(reward_total)

    obs = torch.Tensor(env.reset()[0])
    done = False

plt.plot(reward_totals)
plt.show()
