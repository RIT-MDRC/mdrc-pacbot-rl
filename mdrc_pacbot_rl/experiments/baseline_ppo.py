"""
Baseline for PPO on Pacman gym.
"""
from functools import reduce
from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical
from matplotlib import pyplot as plt
from tqdm import tqdm
from gymnasium.vector.sync_vector_env import SyncVectorEnv

from mdrc_pacbot_rl.algorithms.rollout_buffer import RolloutBuffer
from mdrc_pacbot_rl.pacman.gym import PacmanGym

# Constants (for now)
num_envs = 128
train_steps = 500
iterations = 1000
train_iters = 2
train_batch_size = 512
discount = 0.98
lambda_ = 0.95
epsilon = 0.2
max_eval_steps = 100
device = torch.device("cpu")

def get_img_size(old_w: int, old_h: int, conv: torch.nn.Conv2d) -> Tuple[int, int]:
    """
    Returns the size of the image after the convolution is run on it.
    """
    w = (old_w + 2 * int(conv.padding[1]) - conv.dilation[1] * (conv.kernel_size[1] - 1) - 1) // conv.stride[1] + 1
    h = (old_h + 2 * int(conv.padding[0]) - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
    return w, h

def copy_params(src: nn.Module, dest: nn.Module):
    """
    Copies params from one model to another.
    """
    with torch.no_grad():
        for dest_, src_ in zip(dest.parameters(), src.parameters()):
            dest_.data.copy_(src_.data)


def init_orthogonal(src: nn.Module):
    with torch.no_grad():
        for param in src.parameters():
            if len(param.size()) >= 2:
                param.copy_(torch.nn.init.orthogonal_(param.data))


class ValueNet(nn.Module):
    def __init__(self, obs_shape: torch.Size):
        nn.Module.__init__(self)
        h, w = obs_shape[1:]
        self.cnn1 = nn.Conv2d(obs_shape[0], 8, 3)
        w, h = get_img_size(w, h, self.cnn1)
        self.cnn2 = nn.Conv2d(8, 16, 3)
        w, h = get_img_size(w, h, self.cnn2)
        flat_dim = w * h * 16
        self.v_layer1 = nn.Linear(1736, 256)
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
        h, w = obs_shape[1:]
        self.cnn1 = nn.Conv2d(obs_shape[0], 8, 3)
        w, h = get_img_size(w, h, self.cnn1)
        self.cnn2 = nn.Conv2d(8, 16, 3)
        w, h = get_img_size(w, h, self.cnn2)
        flat_dim = w * h * 16
        self.a_layer1 = nn.Linear(1736, 256)
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


env = SyncVectorEnv([lambda: PacmanGym()] * num_envs)
test_env = PacmanGym()
reward_totals = []
entropies = []

# Init PPO
obs_space = env.envs[0].observation_space
act_space = env.envs[0].action_space
v_net = ValueNet(obs_space.shape)
p_net = PolicyNet(obs_space.shape, act_space.n)
p_net_old = PolicyNet(obs_space.shape, act_space.n)
p_net_old.eval()
v_opt = torch.optim.Adam(v_net.parameters(), lr=0.01)
p_opt = torch.optim.Adam(p_net.parameters(), lr=0.0001)
buffer = RolloutBuffer(
    obs_space.shape,
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
                old_act_probs = Categorical(logits=old_log_probs).log_prob(actions.squeeze())
            p_opt.zero_grad()
            new_log_probs = p_net(prev_states)
            new_act_probs = Categorical(logits=new_log_probs).log_prob(actions.squeeze())
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

torch.save(v_net, "VNet.pt")
torch.save(p_net, "PNet.pt")

figure, axis = plt.subplots(2, 1)
axis[0].plot(reward_totals)
axis[0].set_title("eval reward")
axis[1].plot(entropies)
axis[1].set_title("entropy")
plt.show()