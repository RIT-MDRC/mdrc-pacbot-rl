"""
Gym environment wrapper for the Pacman game.
Observation: Box space of 2x28x31. Dims 2 and 3 are the width and height,
while the first is a stack of grid data and entity (Pacman, ghosts) data.
Action: Discrete space of nothing, up, down, left, right.
"""

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete

from .gameState import GameState

GRID_WIDTH = 28
GRID_HEIGHT = 31
RENDER_PIXEL_SCALE = 10


class PacmanGym(gym.Env):
    def __init__(self, render_mode: str = ""):
        self.observation_space = Box(0.0, 5.0, (2, GRID_WIDTH, GRID_HEIGHT))
        self.action_space = Discrete(5)
        self.render_mode = render_mode
        self.game_state = GameState()
        self.last_score = 0

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
        return self.create_obs(), {}

    def step(self, action):
        old_pos = self.game_state.pacbot.pos
        if action == 0:
            new_pos = (old_pos[0], old_pos[1])
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
        self.game_state.next_step()

        reward = self.game_state.score - self.last_score
        if reward == float("Nan"):
            reward = 0
        self.last_score = self.game_state.score

        if self.render_mode == "human":
            self.update_surface()
            self.clock.tick(1)
            pygame.transform.scale(
                self.surface,
                (GRID_WIDTH * RENDER_PIXEL_SCALE, GRID_HEIGHT * RENDER_PIXEL_SCALE),
                self.window_surface,
            )
            pygame.display.update()

        return self.create_obs(), reward, not self.game_state.play, {}, {}

    def update_surface(self):
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
                    (255, 0, 0),
                    (0, 0, 255),
                    (255, 128, 128),
                    (255, 128, 0),
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
        entities = np.zeros(grid.shape)
        # Add entities
        entity_positions = [
            self.game_state.pacbot.pos,
            self.game_state.red.pos["current"],
            self.game_state.blue.pos["current"],
            self.game_state.pink.pos["current"],
            self.game_state.orange.pos["current"],
        ]
        for i, pos in enumerate(entity_positions):
            entities[pos[0]][pos[1]] = i + 1
        obs = np.stack([grid, entities])
        return obs
