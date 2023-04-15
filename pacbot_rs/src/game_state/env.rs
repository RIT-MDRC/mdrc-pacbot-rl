use num_enum::{IntoPrimitive, TryFromPrimitive};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::seq::SliceRandom;

use crate::grid::{self, coords_to_node, NODE_COORDS, VALID_ACTIONS};

use super::GameState;

/// How many ticks the game should move every step. Ghosts move every 12 ticks.
const TICKS_PER_STEP: u32 = 12;

/// Whether to randomize the ghosts' positions when `random_start = true`.
const RANDOMIZE_GHOSTS: bool = true;

#[derive(Clone, Copy, Debug, Eq, PartialEq, TryFromPrimitive, IntoPrimitive)]
#[repr(u8)]
pub enum Action {
    Stay = 0,
    Down = 1,
    Up = 2,
    Left = 3,
    Right = 4,
}

impl<'source> FromPyObject<'source> for Action {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let index: u8 = ob.extract()?;
        Action::try_from_primitive(index).map_err(|_| PyValueError::new_err("Invalid action"))
    }
}

impl IntoPy<PyObject> for Action {
    fn into_py(self, py: Python<'_>) -> PyObject {
        u8::from(self).into_py(py)
    }
}

#[derive(Clone)]
#[pyclass]
pub struct PacmanGym {
    #[pyo3(get)]
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
        if random_start && RANDOMIZE_GHOSTS {
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
            let mut random_pos = || *NODE_COORDS.choose(rng).unwrap();

            self.game_state.pacbot.update(random_pos());

            if RANDOMIZE_GHOSTS {
                for mut ghost in self.game_state.ghosts_mut() {
                    ghost.current_pos = random_pos();
                    ghost.next_pos = ghost.current_pos;
                }
            }
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

        let done = self.is_done();

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

    pub fn is_done(&self) -> bool {
        !self.game_state.play
    }

    /// Returns the action mask that is `True` for currently-valid actions and
    /// `False` for currently-invalid actions.
    pub fn action_mask(&self) -> [bool; 5] {
        let pacbot_pos = self.game_state.pacbot.pos;
        let pacbot_node = coords_to_node(pacbot_pos).expect("PacBot is in an invalid location");
        VALID_ACTIONS[pacbot_node]
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
