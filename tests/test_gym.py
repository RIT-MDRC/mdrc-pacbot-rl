from typing import Tuple
import numpy as np
import pytest

from mdrc_pacbot_rl.pacman.gym import PacmanGym


@pytest.fixture
def gym_env():
    return PacmanGym()


def get_entity_pos(entity_id: int, obs: np.ndarray) -> Tuple[int, int]:
    pos = np.where(np.moveaxis(obs, -1, 0)[1] == entity_id)
    return (pos[0].item(), pos[1].item())


def test_reset_ok(gym_env: PacmanGym):
    gym_env.reset()


def test_random_action(gym_env: PacmanGym):
    action_space = gym_env.action_space
    gym_env.reset()
    gym_env.step(action_space.sample())


@pytest.mark.parametrize(
    "action,dx,dy",
    [
        (0, 0, 0),
        (1, 0, 1),
        (2, 0, -1),
        (3, -1, 0),
        (4, 1, 0),
    ],
)
def test_move_pacman(gym_env: PacmanGym, action: int, dx: int, dy: int):
    obs = gym_env.reset()
    pac_pos_old = get_entity_pos(1, obs)
    obs, _, _ = gym_env.step(action)
    pac_pos = get_entity_pos(1, obs)
    assert pac_pos[0] - pac_pos_old[0] == dx
    assert pac_pos[1] - pac_pos_old[1] == dy


def test_game_ends(gym_env: PacmanGym):
    gym_env.reset()
    done = False
    ran_once = False
    while not done:
        _, _, done = gym_env.step(0)
        ran_once = True
    assert ran_once


def test_reward_on_pellet(gym_env: PacmanGym):
    gym_env.reset()
    _, reward, _ = gym_env.step(4)
    assert reward == pytest.approx(10)


def test_no_reward_on_empty(gym_env: PacmanGym):
    gym_env.reset()
    _, reward, _ = gym_env.step(0)
    assert reward == pytest.approx(0)
