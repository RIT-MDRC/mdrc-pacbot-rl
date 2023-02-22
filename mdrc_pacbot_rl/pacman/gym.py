"""
Gym environment wrapper for the Pacman game.
Observation: Box space of 28x31x2. The first two dimensinos are the width and height,
while the third dimension is a stack of grid data and entity (Pacman, ghosts) data.
Action: Discrete space of nothing, up, down, left, right.
"""

import gym
import numpy as np
from gym.spaces import Discrete, Box
from .gameState import GameState

GRID_WIDTH = 28
GRID_HEIGHT = 31


class PacmanGym(gym.Env):
    def __init__(self):
        self.observation_space = Box(0.0, 5.0, (GRID_HEIGHT, GRID_WIDTH, 1))
        self.action_space = Discrete(5)
        self.game_state = GameState()
        self.last_score = 0

    def reset(self):
        self.last_score = 0
        self.game_state.restart()
        self.game_state.unpause()
        return self.create_obs()

    def step(self, action):
        old_pos = self.game_state.pacbot.pos
        if action == 0:
            new_pos = (old_pos[0], old_pos[1])
        if action == 1:
            new_pos = (old_pos[0], old_pos[1] + 1)
        if action == 2:
            new_pos = (old_pos[0], old_pos[1] - 1)
        if action == 3:
            new_pos = (old_pos[0] - 1, old_pos[1])
        if action == 4:
            new_pos = (old_pos[0] + 1, old_pos[1])
        self.game_state.pacbot.update(new_pos)
        self.game_state.next_step()
        reward = self.game_state.score - self.last_score
        self.last_score = self.game_state.score
        return self.create_obs(), reward, not self.game_state.play

    def create_obs(self):
        grid = np.array(self.game_state.grid)
        entities = np.zeros(grid.shape)
        # Add entities
        entity_positions = [
            self.game_state.pacbot.pos,
            self.game_state.red.pos["current"],
            self.game_state.blue.pos["current"],
            self.game_state.pink.pos["current"],
            self.game_state.orange.pos["current"],
        ]
        for (i, pos) in enumerate(entity_positions):
            entities[pos[0]][pos[1]] = i + 1
        obs = np.moveaxis(np.stack([grid, entities]), 0, -1)
        return obs
