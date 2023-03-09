"""
Baseline for PPO on Pacman gym.

CLI Args:
    --eval: Run the last saved policy in the test environment, with visualization.
    --resume: Resume training from the last saved policy.
"""
from typing import Any
import sys

import torch
import torch.nn as nn
from gymnasium.spaces.discrete import Discrete
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from matplotlib import pyplot as plt  # type: ignore
from torch.distributions import Categorical
from tqdm import tqdm

from mdrc_pacbot_rl.algorithms.rollout_buffer import RolloutBuffer
from mdrc_pacbot_rl.pacman.gym import PacmanGym
from mdrc_pacbot_rl.utils import copy_params, get_img_size, init_orthogonal

_: Any

# Hyperparameters
num_envs = 128
train_steps = 500
iterations = 50
train_iters = 2
train_batch_size = 256
discount = 0.98
lambda_ = 0.95
epsilon = 0.2
max_eval_steps = 100
device = torch.device("cpu")


class ValueNet(nn.Module):
    def __init__(self, obs_shape: torch.Size):
        nn.Module.__init__(self)
        w, h = obs_shape[1:]
        self.cnn1 = nn.Conv2d(obs_shape[0], 8, 3, 2)
        h, w = get_img_size(h, w, self.cnn1)
        self.cnn2 = nn.Conv2d(8, 12, 3, 2)
        h, w = get_img_size(h, w, self.cnn2)
        flat_dim = h * w * self.cnn2.out_channels
        self.v_layer1 = nn.Linear(flat_dim, 128)
        self.v_layer2 = nn.Linear(128, 1)
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
        return x


class PolicyNet(nn.Module):
    def __init__(self, obs_shape: torch.Size, action_count: int):
        nn.Module.__init__(self)
        w, h = obs_shape[1:]
        self.cnn1 = nn.Conv2d(obs_shape[0], 8, 3, 2)
        h, w = get_img_size(h, w, self.cnn1)
        self.cnn2 = nn.Conv2d(8, 12, 3, 2)
        h, w = get_img_size(h, w, self.cnn2)
        flat_dim = h * w * self.cnn2.out_channels
        self.a_layer1 = nn.Linear(flat_dim, 128)
        self.a_layer2 = nn.Linear(128, action_count)
        self.relu = nn.ReLU()
        self.logits = nn.LogSoftmax(1)
        init_orthogonal(self)

    def forward(self, input: torch.Tensor):
        x = self.cnn1(input)
        x = self.relu(x)
        x = self.cnn2(x)
        x = x.flatten(1)
        x = self.relu(x)
        x = self.a_layer1(x)
        x = self.relu(x)
        x = self.a_layer2(x)
        x = self.logits(x)
        return x


env = SyncVectorEnv([lambda: PacmanGym()] * num_envs)
test_env = PacmanGym()
reward_totals = []
entropies = []

# If evaluating, just run the eval env
if len(sys.argv) >= 2 and sys.argv[1] == "--eval":
    test_env = PacmanGym(render_mode="human")
    p_net = torch.load("temp/PNet.pt")
    with torch.no_grad():
        obs = torch.Tensor(test_env.reset()[0])
        for _ in range(max_eval_steps):
            distr = Categorical(logits=p_net(obs.unsqueeze(0)).squeeze())
            action = distr.sample().item()
            obs_, reward, done, _, _ = test_env.step(action)
            obs = torch.Tensor(obs_)
            if done:
                obs = torch.Tensor(test_env.reset()[0])
                break
    quit()

# Init PPO
obs_shape = env.envs[0].observation_space.shape
if not obs_shape:
    raise RuntimeError("Observation space doesn't have shape")
obs_size = torch.Size(obs_shape)
act_space = env.envs[0].action_space
if not isinstance(act_space, Discrete):
    raise RuntimeError("Action space was not discrete")
if len(sys.argv) >= 2 and sys.argv[1] == "--resume":
    v_net = torch.load("temp/VNet.pt")
    p_net = torch.load("temp/PNet.pt")
    p_net_old = torch.load("temp/PNet.pt")
else:
    v_net = ValueNet(torch.Size(obs_shape))
    p_net = PolicyNet(obs_size, int(act_space.n))
    p_net_old = PolicyNet(obs_size, int(act_space.n))
p_net_old.eval()
v_opt = torch.optim.Adam(v_net.parameters(), lr=0.01)
p_opt = torch.optim.Adam(p_net.parameters(), lr=0.0001)
buffer = RolloutBuffer(
    obs_size,
    torch.Size((1,)),
    torch.int,
    num_envs,
    train_steps,
    device,
)

obs = torch.Tensor(env.reset()[0])
done = False
for _ in tqdm(range(iterations), position=0):
    # Collect experience
    with torch.no_grad():
        for _ in tqdm(range(train_steps), position=1):
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
    obs = torch.Tensor(test_env.reset()[0])
    done = False
    with torch.no_grad():
        # Visualize
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
                avg_entropy += distr.entropy()
            avg_entropy /= steps_taken
            entropy_total += avg_entropy
        reward_totals.append(reward_total / eval_steps)
        entropies.append(entropy_total / eval_steps)
    obs = torch.Tensor(env.reset()[0])
    done = False

torch.save(v_net, "temp/VNet.pt")
torch.save(p_net, "temp/PNet.pt")

figure, axis = plt.subplots(2, 1)
axis[0].plot(reward_totals)
axis[0].set_title("eval reward")
axis[1].plot(entropies)
axis[1].set_title("entropy")
plt.show()
