"""
Experiment where the agent performs random actions.
"""
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt  # type: ignore
from tqdm import tqdm

from mdrc_pacbot_rl.pacman.gym import PacmanGym

max_eval_steps = 300
eval_steps = 8
iterations = 300

wandb.init(
    project="pacbot",
    entity="mdrc-pacbot",
    config={"experiment": "random actions baseline"},
)

env = PacmanGym()
action_space = env.action_space
obs, _ = env.reset()
done = False
for _ in tqdm(range(iterations)):
    total_reward = 0.0
    for _ in range(eval_steps):
        for _ in range(max_eval_steps):
            action = action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        obs = env.reset()
        done = False
    avg_eval_episode_reward = total_reward / eval_steps
    wandb.log({"avg_eval_episode_reward": avg_eval_episode_reward})

wandb.finish()
