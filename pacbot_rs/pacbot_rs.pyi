from typing import Literal, Mapping, Optional

import numpy as np

class GhostAgent:
    def clear_start_path(self) -> None: ...
    @property
    def pos(self) -> Mapping[Literal["current", "next"], tuple[int, int]]: ...

class PacBot:
    def update(self, position: tuple[int, int]) -> None: ...
    @property
    def direction(self) -> int: ...
    @property
    def pos(self) -> tuple[int, int]: ...

class GameState:
    def __init__(self) -> None: ...
    def is_frightened(self) -> bool: ...
    def next_step(self) -> None: ...
    def pause(self) -> None: ...
    def unpause(self) -> None: ...
    def restart(self) -> None: ...
    def print_ghost_pos(self) -> None: ...
    def state(self) -> int: ...
    @property
    def pacbot(self) -> PacBot: ...
    @property
    def red(self) -> GhostAgent: ...
    @property
    def pink(self) -> GhostAgent: ...
    @property
    def orange(self) -> GhostAgent: ...
    @property
    def blue(self) -> GhostAgent: ...
    @property
    def pellets(self) -> int: ...
    @property
    def power_pellets(self) -> int: ...
    @property
    def cherry(self) -> bool: ...
    @property
    def score(self) -> int: ...
    @property
    def play(self) -> bool: ...
    @property
    def lives(self) -> int: ...

class PacmanGym:
    def __init__(self, random_start: bool) -> None: ...
    def reset(self) -> None: ...
    def step(self, action: int) -> tuple[int, bool]: ...
    def score(self) -> int: ...
    def is_done(self) -> bool: ...
    def action_mask(self) -> list[bool]: ...
    @property
    def game_state(self) -> GameState: ...
    @property
    def random_start(self) -> bool: ...

class MCTSContext:
    def __init__(self) -> None: ...
    def update_root(self, action: int) -> None: ...
    def best_action(self) -> int: ...
    def ponder_and_choose(self, env: PacmanGym, num_iterations: int) -> int: ...
    def max_depth(self) -> int: ...
    def node_count(self) -> int: ...

class ParticleFilter:
    def __init__(self, pacbot_x: int, pacbot_y: int, pacbot_angle: f64): ...
    def get_points(self) -> list[tuple[tuple[float, float], float]]: ...
    def get_empty_grid_cells(self) -> list[tuple[int, int]]: ...
    def get_map_segments_list(self) -> list[tuple[int, int, int, int]]: ...
    def update(self, magnitude: float, direction: float, sensors: list[float]) -> tuple[tuple[float, float], float]: ...

def create_obs_semantic(game_state: GameState) -> np.ndarray: ...
def get_heuristic_value(
    game_state: GameState, pos: tuple[int, int]
) -> Optional[float]: ...
def get_action_heuristic_values(
    game_state: GameState,
) -> tuple[list[Optional[float]], int]: ...
