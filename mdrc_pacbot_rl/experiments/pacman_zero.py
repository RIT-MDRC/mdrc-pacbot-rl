"""
Experiment for combining Monte Carlo Tree Search with a value network.
This is eval only.
"""
import copy
import json
import random
import sys
from typing import Any, Tuple

import numpy as np
from pacbot_rs import MCTSContext, PacmanGym
import torch
import torch.nn as nn
import wandb
from gymnasium.spaces.discrete import Discrete
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.time_limit import TimeLimit
from torch.distributions import Categorical
from tqdm import tqdm

from mdrc_pacbot_rl.algorithms.ppo import train_ppo
from mdrc_pacbot_rl.algorithms.replay_buffer import ReplayBuffer
from mdrc_pacbot_rl.pacman.gym import NaivePacmanGym as PyPacmanGym
from mdrc_pacbot_rl.utils import copy_params, get_img_size, init_orthogonal

mcts = MCTSContext()
test_env = PacmanGym(False)
test_env_viz = PyPacmanGym(render_mode="human")
with torch.no_grad():
    test_env.reset()
    while True:
        mcts = MCTSContext()
        action = mcts.ponder_and_choose(test_env, 10)
        act_dist = mcts.action_distribution()
        reward, done = test_env.step(action)
        test_env_viz.game_state = test_env.game_state
        test_env_viz.step(0)
        print(act_dist, reward)
        if done:
            test_env.reset()