use std::cell::RefCell;

use pyo3::{
    exceptions::{PyIndexError, PyKeyError},
    prelude::*,
};

use super::GameState;
use crate::ghost_agent::GhostAgent;

/// Wraps a reference to one of a GameState's ghosts.
#[pyclass]
struct GhostAgentWrapper {
    game_state: Py<GameState>,
    get_ghost: fn(&GameState) -> &RefCell<GhostAgent>,
}

#[pymethods]
impl GhostAgentWrapper {
    #[getter]
    fn pos(&self) -> GhostPosWrapper {
        GhostPosWrapper {
            game_state: self.game_state.clone(),
            get_ghost: self.get_ghost,
        }
    }
}

/// Wraps a reference to one of a ghost's positions.
#[pyclass]
struct GhostPosWrapper {
    game_state: Py<GameState>,
    get_ghost: fn(&GameState) -> &RefCell<GhostAgent>,
}

#[pymethods]
impl GhostPosWrapper {
    fn __getitem__(&self, item: &str) -> PyResult<(usize, usize)> {
        Python::with_gil(|py| {
            let game_state = self.game_state.borrow(py);
            let ghost = (self.get_ghost)(&game_state).borrow();
            match item {
                "current" => Ok(ghost.current_pos),
                "next" => Ok(ghost.next_pos),
                _ => Err(PyKeyError::new_err(item.to_owned())),
            }
        })
    }
}

pub(super) fn wrap_ghost_agent(
    game_state: Py<GameState>,
    get_ghost: fn(&GameState) -> &RefCell<GhostAgent>,
) -> impl IntoPy<Py<PyAny>> {
    GhostAgentWrapper {
        game_state,
        get_ghost,
    }
}

/// Wraps a reference to a GameState's grid.
#[pyclass]
struct GridWrapper {
    game_state: Py<GameState>,
}

#[pymethods]
impl GridWrapper {
    fn __getitem__(&self, index: usize) -> PyResult<GridRowWrapper> {
        Python::with_gil(|py| {
            let game_state = self.game_state.borrow(py);
            if index < game_state.grid.len() {
                Ok(GridRowWrapper {
                    game_state: self.game_state.clone(),
                    row: index,
                })
            } else {
                Err(PyIndexError::new_err("grid row index out of range"))
            }
        })
    }
}

/// Wraps a reference to a row of a GameState's grid.
#[pyclass]
struct GridRowWrapper {
    game_state: Py<GameState>,
    row: usize,
}

#[pymethods]
impl GridRowWrapper {
    fn __getitem__(&self, index: usize) -> PyResult<u8> {
        Python::with_gil(|py| {
            let game_state = self.game_state.borrow(py);
            if index < game_state.grid[self.row].len() {
                Ok(game_state.grid[self.row][index].into())
            } else {
                Err(PyIndexError::new_err("grid column index out of range"))
            }
        })
    }
}

pub(super) fn wrap_grid(game_state: Py<GameState>) -> impl IntoPy<Py<PyAny>> {
    GridWrapper { game_state }
}

/// Wraps a reference to a GameState's grid.
#[pyclass]
struct PacBotWrapper {
    game_state: Py<GameState>,
}

#[pymethods]
impl PacBotWrapper {
    #[getter]
    fn pos(&self) -> (usize, usize) {
        Python::with_gil(|py| {
            let game_state = self.game_state.borrow(py);
            game_state.pacbot.pos
        })
    }

    #[getter]
    fn direction(&self) -> u8 {
        Python::with_gil(|py| {
            let game_state = self.game_state.borrow(py);
            game_state.pacbot.direction.into()
        })
    }

    fn update(&self, position: (usize, usize)) {
        Python::with_gil(|py| {
            let mut game_state = self.game_state.borrow_mut(py);
            game_state.pacbot.update(position);
        })
    }
}

pub(super) fn wrap_pacbot(game_state: Py<GameState>) -> impl IntoPy<Py<PyAny>> {
    PacBotWrapper { game_state }
}
