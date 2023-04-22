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
import torch

from mdrc_pacbot_rl.pacman.gym import BasePacmanGym as PyPacmanGym
from mdrc_pacbot_rl.utils import copy_params, get_img_size, init_orthogonal
from pacbot_rs import MCTSContext, PacmanGym

use_net = len(sys.argv) >= 2 and sys.argv[1] == "--use-net"
print("Use net:", use_net)

mcts = MCTSContext()
test_env = PacmanGym(False)
mcts = MCTSContext()
with torch.no_grad():
    test_env.reset()
    done = False
    while not done:
        action = mcts.ponder_and_choose(test_env, 50, use_net)
        act_dist = mcts.action_distribution()
        reward, done = test_env.step(action)
        mcts.clear()
        print("Score:", test_env.score())
        if done:
            print("Pellets remaining:", test_env.game_state.pellets)
