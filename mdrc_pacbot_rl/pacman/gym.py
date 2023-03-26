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


class BasePacmanGym(gym.Env):
    """
    Base for Pacman environments.
    Handles rendering, but not much else.
    """

    def __init__(
        self,
        random_start: bool = False,
        render_mode: str = "",
    ):
        """
        Args:
            random_start: If Pacman and the ghosts should start on random cells.
        """
        self.render_mode = render_mode
        self.game_state = GameState()
        self.last_score = 0
        self.random_start = random_start

        if random_start:
            self.game_state.red.start_path = []
            self.game_state.pink.start_path = []
            self.game_state.orange.start_path = []
            self.game_state.blue.start_path = []

        self.valid_cells = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                # Skip middle of board
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
        if self.random_start:
            self.game_state.pacbot.update(random.choice(self.valid_cells))
            # self.game_state.red.pos["current"] = random.choice(self.valid_cells)
            # self.game_state.red.pos["next"] = self.game_state.red.pos["current"]
            # self.game_state.pink.pos["current"] = random.choice(self.valid_cells)
            # self.game_state.pink.pos["next"] = self.game_state.pink.pos["current"]
            # self.game_state.orange.pos["current"] = random.choice(self.valid_cells)
            # self.game_state.orange.pos["next"] = self.game_state.orange.pos["current"]
            # self.game_state.blue.pos["current"] = random.choice(self.valid_cells)
            # self.game_state.blue.pos["next"] = self.game_state.blue.pos["current"]
        self.game_state.unpause()
        return self.create_obs(), {}

    def step(self, action):
        """
        Override this to return the obs, reward, done flag, and {}, {}
        """
        raise NotImplementedError()

    def move_one_cell(self, action):
        """
        Moves Pacman by one cell.
        If a wall is hit, Pacman doesn't move.

        Actions:
            0: Stay in place
            1: Down
            2: Up
            3: Left
            4: Right
        """
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

    def handle_rendering(self):
        """
        If render mode is set to human, renders the frame.
        """
        if self.render_mode == "human":
            self.update_surface()
            self.clock.tick(5)
            pygame.transform.scale(
                self.surface,
                (GRID_WIDTH * RENDER_PIXEL_SCALE, GRID_HEIGHT * RENDER_PIXEL_SCALE),
                self.window_surface,
            )
            pygame.display.update()

    def update_surface(self):
        """
        Renders the current game state.
        """
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

    def score(self):
        """
        Returns the current score of the underlying game.
        This is useful for directly comparing the performance of two gyms with
        different reward scales/distributions.
        """
        return self.game_state.score

    def create_obs(self):
        """
        Override this to return the current observation.
        """
        raise NotImplementedError()


class NaivePacmanGym(BasePacmanGym):
    """
    Naive Pacman environment with little preprocessing.

    Observation: Box space of 2x28x31. Dims 2 and 3 are the width and height,
    while the first is a stack of grid data and entity (Pacman, ghosts) data.
    Action: Discrete space of nothing, up, down, left, right.
    Rewards: Difference between score after action and before.
    """

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
        self.observation_space = Box(-1.0, 5.0, (2, GRID_WIDTH, GRID_HEIGHT))
        self.action_space = Discrete(5)
        self.ticks_per_step = ticks_per_step
        BasePacmanGym.__init__(self, random_start, render_mode)

    def step(self, action):
        # Update Pacman pos
        self.move_one_cell(action)

        # Step through environment multiple times
        for _ in range(self.ticks_per_step):
            self.game_state.next_step()

        done = not self.game_state.play

        # Reward is raw difference in game score, or -100 if eaten
        reward = self.game_state.score - self.last_score
        if done:
            reward = -100
        if reward == float("Nan"):
            reward = 0
        self.last_score = self.game_state.score

        self.handle_rendering()

        return self.create_obs(), reward, done, {}, {}

    def create_obs(self):
        fright = self.game_state.state == variables.frightened
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
            entities[pos[0]][pos[1]] = -1 if fright and i > 0 else i + 1
        obs = np.stack([grid, entities])
        return obs


class SemanticChannelPacmanGym(BasePacmanGym):
    """
    This environment's observation space is split across channels with more semantic meaning.

    Observation: Box space of 5x28x31. Dims 2 and 3 are the width and height.
    For the first dimension, the channels are:
        1. Wall channel: Binary channel indicating 1 if wall, 0 if empty.
        2. Reward channel: Reward for each item (pellet, super pellet, cherry, frightened ghost) log normalized.
        3. Self channel: Binary channel of 1 if pacman, 0 if not.
        4. Ghost channel: 0.25, 0.5, 0.75, 1 for different ghost colors. 0 otherwise.
        5. Ghost channel prev pos: 0.25, 0.5, 0.75, 1 for different ghosts' previous cells. 0 otherwise.
    Action: Discrete space of nothing, up, down, left, right.
    Rewards: Log normalized difference between score after and before action.
    """

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
        self.ticks_per_step = ticks_per_step
        BasePacmanGym.__init__(self, random_start, render_mode)
        self.last_ghost_pos = [
            self.game_state.red.pos["current"],
            self.game_state.pink.pos["current"],
            self.game_state.orange.pos["current"],
            self.game_state.blue.pos["current"],
        ]

    def reset(self):
        results = super().reset()
        entity_positions = [
            self.game_state.red.pos["current"],
            self.game_state.pink.pos["current"],
            self.game_state.orange.pos["current"],
            self.game_state.blue.pos["current"],
        ]
        self.last_ghost_pos = entity_positions
        return results

    def step(self, action):
        self.move_one_cell(action)

        entity_positions = [
            self.game_state.red.pos["current"],
            self.game_state.pink.pos["current"],
            self.game_state.orange.pos["current"],
            self.game_state.blue.pos["current"],
        ]

        for _ in range(self.ticks_per_step):
            self.game_state.next_step()

        # If the ghost positions change, update the last ghost positions
        new_entity_positions = [
            self.game_state.red.pos["current"],
            self.game_state.pink.pos["current"],
            self.game_state.orange.pos["current"],
            self.game_state.blue.pos["current"],
        ]
        pos_changed = any(
            old != new for old, new in zip(entity_positions, new_entity_positions)
        )
        if pos_changed:
            self.last_ghost_pos = entity_positions

        done = not self.game_state.play

        # Use log normalized rewards
        reward = math.log(1 + self.game_state.score - self.last_score) / math.log(200)
        if done:
            reward = -1.0
        if reward == float("Nan"):
            reward = 0
        self.last_score = self.game_state.score

        self.handle_rendering()

        return self.create_obs(), reward, done, {}, {}

    def create_obs(self):
        grid = np.array(self.game_state.grid)
        wall = np.where(grid == 1, 1, 0)

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
        reward = np.log(
            1
            + np.where(grid == 2, 1, 0) * variables.pellet_score
            + np.where(grid == 6, 1, 0) * variables.cherry_score
            + np.where(grid == 4, 1, 0) * variables.power_pellet_score
            + fright_ghost * variables.ghost_score
        ) / math.log(variables.ghost_score)

        pac_pos = self.game_state.pacbot.pos
        pacman = np.zeros(grid.shape)
        pacman[pac_pos[0]][pac_pos[1]] = 1

        obs = np.stack([wall, reward, pacman, ghost, last_ghost])
        return obs


class SingleGhostPacmanGym(BasePacmanGym):
    """
    This environment is almost exactly the same as SemanticChannelPacmanGym. The
    only difference is that there's only one ghost type, the red ghost. There
    are still 4 ghosts, though. Also, all ghosts show up as 1 for the ghost channels.
    """

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
        self.observation_space = Box(-1.0, 1.0, (6, GRID_WIDTH, GRID_HEIGHT))
        self.action_space = Discrete(5)
        self.ticks_per_step = ticks_per_step
        BasePacmanGym.__init__(self, random_start, render_mode)
        self.game_state.pink.color = variables.red
        self.game_state.orange.color = variables.red
        self.game_state.blue.color = variables.red
        self.last_ghost_pos = [
            self.game_state.red.pos["current"],
            self.game_state.pink.pos["current"],
            self.game_state.orange.pos["current"],
            self.game_state.blue.pos["current"],
        ]
        self.last_pacman_pos = self.game_state.pacbot.pos

    def reset(self):
        results = super().reset()
        self.last_pacman_pos = self.game_state.pacbot.pos
        entity_positions = [
            self.game_state.red.pos["current"],
            self.game_state.pink.pos["current"],
            self.game_state.orange.pos["current"],
            self.game_state.blue.pos["current"],
        ]
        self.last_ghost_pos = entity_positions
        return results

    def step(self, action):
        self.last_pacman_pos = self.game_state.pacbot.pos
        self.move_one_cell(action)

        entity_positions = [
            self.game_state.red.pos["current"],
            self.game_state.pink.pos["current"],
            self.game_state.orange.pos["current"],
            self.game_state.blue.pos["current"],
        ]

        for _ in range(self.ticks_per_step):
            self.game_state.next_step()

        # If the ghost positions change, update the last ghost positions
        new_entity_positions = [
            self.game_state.red.pos["current"],
            self.game_state.pink.pos["current"],
            self.game_state.orange.pos["current"],
            self.game_state.blue.pos["current"],
        ]
        pos_changed = any(
            old != new for old, new in zip(entity_positions, new_entity_positions)
        )
        if pos_changed:
            self.last_ghost_pos = entity_positions

        done = not self.game_state.play

        # Use log normalized rewards
        reward = math.log(1 + self.game_state.score - self.last_score) / math.log(200)
        if done:
            reward = -1.0
        if reward == float("Nan"):
            reward = 0
        self.last_score = self.game_state.score

        self.handle_rendering()

        return self.create_obs(), reward, done, {}, {}

    def create_obs(self):
        grid = np.array(self.game_state.grid)
        wall = np.where(grid == 1, 1, 0)

        fright = self.game_state.state == variables.frightened
        entity_positions = [
            self.game_state.red.pos["current"],
            self.game_state.pink.pos["current"],
            self.game_state.orange.pos["current"],
            self.game_state.blue.pos["current"],
        ]
        ghost = np.zeros(grid.shape)
        for i, pos in enumerate(entity_positions):
            ghost[pos[0]][pos[1]] = 1

        last_ghost = np.zeros(grid.shape)
        for i, pos in enumerate(self.last_ghost_pos):
            last_ghost[pos[0]][pos[1]] = 1

        fright_ghost = np.where(ghost > 0, 1, 0) * int(fright)
        reward = np.log(
            1
            + np.where(grid == 2, 1, 0) * variables.pellet_score
            + np.where(grid == 6, 1, 0) * variables.cherry_score
            + np.where(grid == 4, 1, 0) * variables.power_pellet_score
            + fright_ghost * variables.ghost_score
        ) / math.log(variables.ghost_score)

        last_pacman = np.zeros(grid.shape)
        last_pacman[self.last_pacman_pos[0]][self.last_pacman_pos[1]] = 1
        pac_pos = self.game_state.pacbot.pos
        pacman = np.zeros(grid.shape)
        pacman[pac_pos[0]][pac_pos[1]] = 1

        obs = np.stack([wall, reward, pacman, last_pacman, ghost, last_ghost])
        return obs
