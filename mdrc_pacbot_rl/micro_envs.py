"""
Micro environments for testing agent sub-behaviors.
"""

import math
import random
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import pygame

GRID_SIZE = 8
GRID = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]
RENDER_PIXEL_SCALE = 10


def move_one_cell(
    pos: Tuple[int, int], grid: List[List[int]], action: int
) -> Tuple[int, int]:
    if action == 0:
        new_pos = (pos[0], pos[1])
    if action == 1:
        new_pos = (pos[0], min(pos[1] + 1, GRID_SIZE - 1))
    if action == 2:
        new_pos = (pos[0], max(pos[1] - 1, 0))
    if action == 3:
        new_pos = (max(pos[0] - 1, 0), pos[1])
    if action == 4:
        new_pos = (min(pos[0] + 1, GRID_SIZE - 1), pos[1])
    if grid[new_pos[0]][new_pos[1]] != 1:
        return new_pos
    return pos


class BaseMicroGym(gym.Env):
    """
    Base class for microgyms.
    All microgyms use the same grid and timer.
    The action space is also the same.
    """

    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = gym.spaces.Discrete(5)
        self.grid = np.array(GRID)
        self.timer = 100
        self.render_mode = render_mode

        # Valid cells are any cells that aren't a wall
        self.valid_cells = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if GRID[x][y] == 0:
                    self.valid_cells.append((x, y))
        self.pos = random.choice(self.valid_cells)

        if render_mode == "human":
            pygame.init()
            self.window_surface = pygame.display.set_mode(
                (GRID_SIZE * RENDER_PIXEL_SCALE, GRID_SIZE * RENDER_PIXEL_SCALE)
            )
            self.surface = pygame.Surface((GRID_SIZE, GRID_SIZE))
            self.clock = pygame.time.Clock()

    def update_timer(self) -> bool:
        """
        Decrements timer and returns True if done.
        """
        self.timer -= 1
        return self.timer == 0

    def reset(self):
        """
        Resets timer and Pacman position.
        """
        self.pos = random.choice(self.valid_cells)
        self.timer = 100

    def handle_rendering(self):
        """
        Call this to render in a step.
        """
        if self.render_mode == "human":
            self.update_surface()
            self.clock.tick(5)
            pygame.transform.scale(
                self.surface,
                (GRID_SIZE * RENDER_PIXEL_SCALE, GRID_SIZE * RENDER_PIXEL_SCALE),
                self.window_surface,
            )
            pygame.display.update()

    def update_surface(self):
        """
        Override this to modify the rendering code.
        """
        self.draw_grid_and_pacman()

    def draw_grid_and_pacman(self):
        """
        Updates the surface with Pacman's position and the grid.
        """
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                color = (0, 0, 0)
                if self.grid[x][y] == 1:
                    color = (0, 0, 255)
                self.surface.set_at((x, y), color)
        self.surface.set_at((self.pos[0], self.pos[1]), (255, 255, 0))


class GetAllPelletsEnv(BaseMicroGym):
    """
    A maze filled with pellets. The agent must learn to collect all pellets
    within a timeframe.
    Obs: 3x8x8
        Layer 1: Grid
        Layer 2: Pellets
        Layer 3: Pacman
    Reward: 1 for each pellet retrieved.
    """

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.observation_space = gym.spaces.Box(0.0, 1.0, [3, 8, 8])
        self.pellets = 1.0 - self.grid

    def step(self, action):
        self.pos = move_one_cell(self.pos, GRID, action)
        pacman = np.zeros([GRID_SIZE, GRID_SIZE])
        pacman[self.pos[0]][self.pos[1]] = 1

        reward = 0
        if self.pellets[self.pos[0]][self.pos[1]] == 1:
            reward = 1
            self.pellets[self.pos[0]][self.pos[1]] = 0

        obs = np.stack([self.grid, self.pellets, pacman])

        done = self.pellets.sum() == 0
        if self.update_timer():
            done = True

        self.handle_rendering()

        return obs, reward, done, {}, {}

    def reset(self):
        super().reset()
        self.pellets = 1.0 - self.grid
        pacman = np.zeros([GRID_SIZE, GRID_SIZE])
        pacman[self.pos[0]][self.pos[1]] = 1
        return np.stack([self.grid, self.pellets, pacman]), {}

    def update_surface(self):
        self.draw_grid_and_pacman()
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.pellets[x][y] == 1:
                    color = (128, 128, 128)
                    self.surface.set_at((x, y), color)


class RunAwayEnv(BaseMicroGym):
    """
    A maze with a ghost chasing the agent. The agent must learn to avoid the
    ghost.
    Obs: 3x8x8
        Layer 1: Grid
        Layer 2: Ghost
        Layer 3: Pacman
    Reward: 1 for every step not dead. -1 on death.
    """

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.observation_space = gym.spaces.Box(0.0, 1.0, [3, 8, 8])
        self.ghost_pos = random.choice(self.valid_cells)

    def step(self, action):
        done = False
        reward = 1

        self.pos = move_one_cell(self.pos, GRID, action)
        pacman = np.zeros([GRID_SIZE, GRID_SIZE])
        pacman[self.pos[0]][self.pos[1]] = 1

        # We have to do this check twice or else Pacman can "go inside" a ghost
        if self.ghost_pos[0] == self.pos[0] and self.ghost_pos[1] == self.pos[1]:
            reward = -1
            done = True

        # Red ghost moves using euclidian distance to Pacman
        next_moves = [
            (self.ghost_pos[0] + 1, self.ghost_pos[1]),
            (self.ghost_pos[0] - 1, self.ghost_pos[1]),
            (self.ghost_pos[0], self.ghost_pos[1] + 1),
            (self.ghost_pos[0], self.ghost_pos[1] - 1),
        ]
        next_moves = [
            (move, math.hypot(move[0] - self.pos[0], move[1] - self.pos[1]))
            for move in next_moves
            if self.grid[move[0]][move[1]] == 0
        ]
        # To make sure red doesn't get stuck, we randomly sample ties
        next_moves = sorted(next_moves, key=lambda x: x[1])
        smallest_dist = next_moves[0][1]
        next_moves = [x for x in next_moves if x[1] == smallest_dist]
        self.ghost_pos = random.choice(next_moves)[0]
        if self.ghost_pos[0] == self.pos[0] and self.ghost_pos[1] == self.pos[1]:
            reward = -1
            done = True

        ghost = np.zeros([GRID_SIZE, GRID_SIZE])
        ghost[self.ghost_pos[0]][self.ghost_pos[1]] = 1
        obs = np.stack([self.grid, ghost, pacman])

        if self.update_timer():
            done = True

        self.handle_rendering()

        return obs, reward, done, {}, {}

    def reset(self):
        super().reset()
        self.ghost_pos = random.choice(self.valid_cells)
        pacman = np.zeros([GRID_SIZE, GRID_SIZE])
        pacman[self.pos[0]][self.pos[1]] = 1
        ghost = np.zeros([GRID_SIZE, GRID_SIZE])
        ghost[self.ghost_pos[0]][self.ghost_pos[1]] = 1
        return np.stack([self.grid, ghost, pacman]), {}

    def update_surface(self):
        self.draw_grid_and_pacman()
        self.surface.set_at(self.ghost_pos, (255, 0, 0))


class CollectAndRunEnv(BaseMicroGym):
    """
    Combines the collection and avoiding environments. The agent must learn to avoid the
    ghost while collecting pellets.
    Obs: 3x8x8
        Layer 1: Grid
        Layer 2: Ghost
        Layer 3: Pacman
        Layer 4: Pellets
    Reward: 1 for each pellet. -1 on death.
    """

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.observation_space = gym.spaces.Box(0.0, 1.0, [4, 8, 8])
        self.ghost_pos = random.choice(self.valid_cells)
        self.pellets = 1.0 - self.grid

    def step(self, action):
        done = False
        reward = 0

        self.pos = move_one_cell(self.pos, GRID, action)
        pacman = np.zeros([GRID_SIZE, GRID_SIZE])
        pacman[self.pos[0]][self.pos[1]] = 1

        # We have to do this check twice or else Pacman can "go inside" a ghost
        if self.ghost_pos[0] == self.pos[0] and self.ghost_pos[1] == self.pos[1]:
            reward = -1
            done = True

        # Red ghost moves using euclidian distance to Pacman
        next_moves = [
            (self.ghost_pos[0] + 1, self.ghost_pos[1]),
            (self.ghost_pos[0] - 1, self.ghost_pos[1]),
            (self.ghost_pos[0], self.ghost_pos[1] + 1),
            (self.ghost_pos[0], self.ghost_pos[1] - 1),
        ]
        next_moves = [
            (move, math.hypot(move[0] - self.pos[0], move[1] - self.pos[1]))
            for move in next_moves
            if self.grid[move[0]][move[1]] == 0
        ]
        # To make sure red doesn't get stuck, we randomly sample ties
        next_moves = sorted(next_moves, key=lambda x: x[1])
        smallest_dist = next_moves[0][1]
        next_moves = [x for x in next_moves if x[1] == smallest_dist]
        self.ghost_pos = random.choice(next_moves)[0]
        if self.ghost_pos[0] == self.pos[0] and self.ghost_pos[1] == self.pos[1]:
            reward = -1
            done = True

        if self.pellets[self.pos[0]][self.pos[1]] == 1:
            reward = 1
            self.pellets[self.pos[0]][self.pos[1]] = 0

        ghost = np.zeros([GRID_SIZE, GRID_SIZE])
        ghost[self.ghost_pos[0]][self.ghost_pos[1]] = 1
        obs = np.stack([self.grid, ghost, pacman, self.pellets])

        if self.update_timer():
            done = True

        self.handle_rendering()

        return obs, reward, done, {}, {}

    def reset(self):
        super().reset()
        self.ghost_pos = random.choice(self.valid_cells)
        pacman = np.zeros([GRID_SIZE, GRID_SIZE])
        pacman[self.pos[0]][self.pos[1]] = 1
        ghost = np.zeros([GRID_SIZE, GRID_SIZE])
        ghost[self.ghost_pos[0]][self.ghost_pos[1]] = 1
        self.pellets = 1.0 - self.grid
        return np.stack([self.grid, ghost, pacman, self.pellets]), {}

    def update_surface(self):
        self.draw_grid_and_pacman()
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.pellets[x][y] == 1:
                    color = (128, 128, 128)
                    self.surface.set_at((x, y), color)
        self.surface.set_at(self.ghost_pos, (255, 0, 0))
