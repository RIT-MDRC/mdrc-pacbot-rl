"""
Experiment where the agent performs random actions.
"""
import torch.nn as nn
from matplotlib import pyplot as plt
from mdrc_pacbot_rl.pacman.gym import PacmanGym
from tqdm import tqdm

# Constants (for now)
train_steps = 100

env = PacmanGym(render_mode="human")
action_space = env.action_space
obs, _ = env.reset()
done = False
reward_means = []
for _ in tqdm(range(100), position=0):
    reward_mean = []
    for _ in tqdm(range(train_steps), position=1):
        action = action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        if done:
            obs = env.reset()
            done = False
        reward_mean.append(reward)
    obs = env.reset()
    done = False
    reward_means.append(sum(reward_mean) / train_steps)

plt.plot(reward_means)
plt.show()
