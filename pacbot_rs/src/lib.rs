pub mod game_state;
pub mod ghost_agent;
pub mod ghost_paths;
pub mod grid;
pub mod observations;
pub mod pacbot;
pub mod variables;

use pyo3::prelude::*;

use game_state::GameState;

/// A Python module containing Rust implementations of the PacBot environment.
#[pymodule]
fn pacbot_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GameState>()?;
    m.add_function(wrap_pyfunction!(observations::create_obs_semantic, m)?)?;
    Ok(())
}
