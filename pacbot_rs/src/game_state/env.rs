use pyo3::prelude::*;
use rand::seq::SliceRandom;

use crate::grid::{self, NODE_COORDS};

use super::GameState;

/// How many ticks the game should move every step. Ghosts move every 12 ticks.
const TICKS_PER_STEP: u32 = 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[pyclass]
pub enum Action {
    Stay = 0,
    Down = 1,
    Up = 2,
    Left = 3,
    Right = 4,
}

#[pyclass]
pub struct PacmanGym {
    pub game_state: GameState,
    #[pyo3(get, set)]
    pub random_start: bool,
    last_score: u32,
}

#[pymethods]
impl PacmanGym {
    #[new]
    pub fn new(random_start: bool) -> Self {
        let mut env = Self {
            game_state: GameState::new(),
            random_start,
            last_score: 0,
        };
        if random_start {
            for mut ghost in env.game_state.ghosts_mut() {
                ghost.start_path = &[];
            }
        }
        env
    }

    pub fn reset(&mut self) {
        self.last_score = 0;
        self.game_state.restart();
        if self.random_start {
            let rng = &mut rand::thread_rng();
            let pac_pos = *NODE_COORDS.choose(rng).unwrap();
            self.game_state.pacbot.update(pac_pos);
        }
        self.game_state.unpause();
    }

    /// Performs an action and steps the environment.
    /// Returns (reward, done).
    pub fn step(&mut self, action: Action) -> (i32, bool) {
        // update Pacman pos
        self.move_one_cell(action);

        // step through environment multiple times
        for _ in 0..TICKS_PER_STEP {
            self.game_state.next_step();
        }

        let done = !self.game_state.play;

        // reward is raw difference in game score, or -100 if eaten
        let reward = if done {
            -100
        } else {
            self.game_state.score as i32 - self.last_score as i32
        };
        self.last_score = self.game_state.score;

        (reward, done)
    }

    pub fn score(&self) -> u32 {
        self.game_state.score
    }
}

impl PacmanGym {
    fn move_one_cell(&mut self, action: Action) {
        use std::cmp::{max, min};
        let old_pos = self.game_state.pacbot.pos;
        let new_pos = match action {
            Action::Stay => (old_pos.0, old_pos.1),
            Action::Down => (old_pos.0, min(old_pos.1 + 1, grid::GRID[0].len() - 1)),
            Action::Up => (old_pos.0, max(old_pos.1 - 1, 0)),
            Action::Left => (max(old_pos.0 - 1, 0), old_pos.1),
            Action::Right => (min(old_pos.0 + 1, grid::GRID.len() - 1), old_pos.1),
        };
        if grid::is_walkable(new_pos) {
            self.game_state.pacbot.update(new_pos);
        }
    }
}
