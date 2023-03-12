"""
Gym environment wrapper for the Pacman game.
This environment's observation space is split across channels with more semantic meaning.

Observation: Box space of 5x28x31. Dims 2 and 3 are the width and height.
For the first dimension, the channels are:
1. Wall channel: Binary channel indicating 1 if wall, 0 if empty.
2. Reward channel: Reward for each item (pellet, super pellet, cherry, frightened ghost) divided by 200.
3. Self channel: Binary channel of 1 if pacman, 0 if not.
4. Ghost channel: 0.25, 0.5, 0.75, 1 for different ghost colors. 0 otherwise.
5. Ghost channel prev pos: 0.25, 0.5, 0.75, 1 for different ghosts' previous cells. 0 otherwise.
Action: Discrete space of nothing, up, down, left, right.
"""

import math
import random

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete

from mdrc_pacbot_rl.pacman import variables

from .gameState import GameState

GRID_WIDTH = 28
GRID_HEIGHT = 31
RENDER_PIXEL_SCALE = 10


class SemanticChannelPacmanGym(gym.Env):
    def __init__(
        self,
        random_start: bool = False,
        ticks_per_step: int = 12,
        render_mode: str = "",
    ):
        """
        Args:
            random_start: If Pacman should start on a random cell.
            ticks_per_step: How many ticks the game should move every step. Ghosts move every 12 ticks.
        """
        self.observation_space = Box(-1.0, 1.0, (5, GRID_WIDTH, GRID_HEIGHT))
        self.action_space = Discrete(5)
        self.render_mode = render_mode
        self.game_state = GameState()
        self.last_score = 0
        self.ticks_per_step = ticks_per_step
        self.random_start = random_start

        self.valid_cells = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                # Skip unreachable black areas
                if (y > 9 and y < GRID_HEIGHT - 5) or y == 27:
                    continue
                if self.game_state.grid[x][y] != 1:
                    self.valid_cells.append((x, y))

        if render_mode == "human":
            pygame.init()
            self.window_surface = pygame.display.set_mode(
                (GRID_WIDTH * RENDER_PIXEL_SCALE, GRID_HEIGHT * RENDER_PIXEL_SCALE)
            )
            self.surface = pygame.Surface((GRID_WIDTH, GRID_HEIGHT))
            self.clock = pygame.time.Clock()
            self.update_surface()

    def reset(self):
        self.last_score = 0
        self.game_state.restart()
        self.game_state.unpause()
        if self.random_start:
            self.game_state.pacbot.update(random.choice(self.valid_cells))
        entity_positions = [
            self.game_state.red.pos["current"],
            self.game_state.pink.pos["current"],
            self.game_state.orange.pos["current"],
            self.game_state.blue.pos["current"],
        ]
        self.last_ghost_pos = entity_positions
        return self.create_obs(), {}

    def step(self, action):
        old_pos = self.game_state.pacbot.pos
        if action == 0:
            new_pos = (old_pos[0], old_pos[1])
            reward = -1.0
        if action == 1:
            new_pos = (old_pos[0], min(old_pos[1] + 1, GRID_HEIGHT - 1))
        if action == 2:
            new_pos = (old_pos[0], max(old_pos[1] - 1, 0))
        if action == 3:
            new_pos = (max(old_pos[0] - 1, 0), old_pos[1])
        if action == 4:
            new_pos = (min(old_pos[0] + 1, GRID_WIDTH - 1), old_pos[1])
        if self.game_state.grid[new_pos[0]][new_pos[1]] != 1:
            self.game_state.pacbot.update(new_pos)

        entity_positions = [
            self.game_state.red.pos["current"],
            self.game_state.pink.pos["current"],
            self.game_state.orange.pos["current"],
            self.game_state.blue.pos["current"],
        ]

        for _ in range(self.ticks_per_step):
            self.game_state.next_step()

        new_entity_positions = [
            self.game_state.red.pos["current"],
            self.game_state.pink.pos["current"],
            self.game_state.orange.pos["current"],
            self.game_state.blue.pos["current"],
        ]

        pos_changed = any(old != new for old, new in zip(entity_positions, new_entity_positions))
        if pos_changed:
            self.last_ghost_pos = entity_positions

        done = not self.game_state.play

        reward = math.log(1 + self.game_state.score - self.last_score) / math.log(200)
        if done:
            reward = -1.0
        if reward == float("Nan"):
            reward = 0
        self.last_score = self.game_state.score

        if self.render_mode == "human":
            self.update_surface()
            self.clock.tick(5)
            pygame.transform.scale(
                self.surface,
                (GRID_WIDTH * RENDER_PIXEL_SCALE, GRID_HEIGHT * RENDER_PIXEL_SCALE),
                self.window_surface,
            )
            pygame.display.update()

        return self.create_obs(), reward, done, {}, {}

    def update_surface(self):
        fright = self.game_state.state == variables.frightened
        fright_color = (10, 10, 10)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                grid_colors = [
                    (0, 0, 0),
                    (0, 0, 255),
                    (128, 128, 128),
                    (0, 0, 0),
                    (255, 255, 255),
                    (20, 20, 20),
                    (255, 0, 0),
                ]
                color = grid_colors[self.game_state.grid[x][y]]
                self.surface.set_at((x, y), color)
                entity_colors = [
                    (255, 255, 0),
                    fright_color if fright else (255, 0, 0),
                    fright_color if fright else (0, 0, 255),
                    fright_color if fright else (255, 128, 128),
                    fright_color if fright else (255, 128, 0),
                ]
                entity_positions = [
                    self.game_state.pacbot.pos,
                    self.game_state.red.pos["current"],
                    self.game_state.blue.pos["current"],
                    self.game_state.pink.pos["current"],
                    self.game_state.orange.pos["current"],
                ]
                for i, pos in enumerate(entity_positions):
                    self.surface.set_at((pos[0], pos[1]), entity_colors[i])

    def create_obs(self):
        grid = np.array(self.game_state.grid)

        fright = self.game_state.state == variables.frightened
        entity_positions = [
            self.game_state.red.pos["current"],
            self.game_state.pink.pos["current"],
            self.game_state.orange.pos["current"],
            self.game_state.blue.pos["current"],
        ]
        ghost = np.zeros(grid.shape)
        for i, pos in enumerate(entity_positions):
            ghost[pos[0]][pos[1]] = (i + 1) / 4
        last_ghost = np.zeros(grid.shape)
        for i, pos in enumerate(self.last_ghost_pos):
            last_ghost[pos[0]][pos[1]] = (i + 1) / 4
        fright_ghost = np.where(ghost > 0, 1, 0) * int(fright)
        wall = np.where(grid == 1, 1, 0)
        reward = (
            np.where(grid == 2, 1, 0) * variables.pellet_score
            + np.where(grid == 6, 1, 0) * variables.cherry_score
            + np.where(grid == 4, 1, 0) * variables.power_pellet_score
            + fright_ghost * variables.ghost_score
        ) / variables.ghost_score

        pac_pos = self.game_state.pacbot.pos
        pacman = np.zeros(grid.shape)
        pacman[pac_pos[0]][pac_pos[1]] = 1
        obs = np.stack([wall, reward, pacman, ghost, last_ghost])
        return obs
