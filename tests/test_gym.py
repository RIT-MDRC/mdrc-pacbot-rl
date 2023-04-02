from typing import Tuple

import numpy as np
import pytest

from mdrc_pacbot_rl.pacman.gym import (
    BasePacmanGym,
    NaivePacmanGym,
    SemanticChannelPacmanGym,
    SemanticPacmanGym,
)


class TestNaivePacmanGym:
    def get_entity_pos(self, entity_id: int, obs: np.ndarray) -> Tuple[int, int]:
        pos = np.where(obs[1] == entity_id)
        return (pos[0].item(), pos[1].item())

    @pytest.mark.parametrize(
        "action,dx,dy",
        [
            (0, 0, 0),
            (1, 0, 0),  # Can't move due to being blocked
            (2, 0, 0),  # Can't move due to being blocked
            (3, -1, 0),
            (4, 1, 0),
        ],
    )
    def test_move_pacman(self, action: int, dx: int, dy: int):
        gym_env = NaivePacmanGym()
        obs, _ = gym_env.reset()
        pac_pos_old = self.get_entity_pos(1, obs)
        obs, _, _, _, _ = gym_env.step(action)
        pac_pos = self.get_entity_pos(1, obs)
        assert pac_pos[0] - pac_pos_old[0] == dx
        assert pac_pos[1] - pac_pos_old[1] == dy


class TestSemanticChannelPacmanGym:
    def get_entity_pos(self, entity_id: int, obs: np.ndarray) -> Tuple[int, int]:
        pos = np.where(obs[2] == entity_id)
        return (pos[0].item(), pos[1].item())

    @pytest.mark.parametrize(
        "action,dx,dy",
        [
            (0, 0, 0),
            (1, 0, 0),  # Can't move due to being blocked
            (2, 0, 0),  # Can't move due to being blocked
            (3, -1, 0),
            (4, 1, 0),
        ],
    )
    def test_move_pacman(self, action: int, dx: int, dy: int):
        gym_env = SemanticChannelPacmanGym()
        obs, _ = gym_env.reset()
        pac_pos_old = self.get_entity_pos(1, obs)
        obs, _, _, _, _ = gym_env.step(action)
        pac_pos = self.get_entity_pos(1, obs)
        assert pac_pos[0] - pac_pos_old[0] == dx
        assert pac_pos[1] - pac_pos_old[1] == dy


@pytest.mark.parametrize(
    "gym_env", [NaivePacmanGym(), SemanticChannelPacmanGym(), SemanticPacmanGym()]
)
class TestGym:
    def test_reset_ok(self, gym_env: BasePacmanGym):
        gym_env.reset()

    def test_random_action(self, gym_env: BasePacmanGym):
        action_space = gym_env.action_space
        gym_env.reset()
        gym_env.step(action_space.sample())

    def test_game_ends(self, gym_env: BasePacmanGym):
        gym_env.reset()
        done = False
        ran_once = False
        while not done:
            _, _, done, _, _ = gym_env.step(0)
            ran_once = True
        assert ran_once

    def test_score_on_pellet(self, gym_env: BasePacmanGym):
        gym_env.reset()
        gym_env.step(4)
        assert gym_env.score() == pytest.approx(10)

    def test_no_reward_on_empty(self, gym_env: BasePacmanGym):
        gym_env.reset()
        _, reward, _, _, _ = gym_env.step(0)
        assert reward == pytest.approx(0)

    def test_random_start_reachable_by_ghosts(self, gym_env: BasePacmanGym):
        gym_env.random_start = True
        for _ in range(100):
            gym_env.reset()
            for step in range(500):
                _, _, done, _, _ = gym_env.step(0)
                if done:
                    break
            if step == 499:
                assert False
