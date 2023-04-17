pub mod game_state;
pub mod ghost_agent;
pub mod ghost_paths;
pub mod grid;
pub mod heuristic_values;
pub mod mcts;
pub mod observations;
pub mod pacbot;
pub mod particle_filter;
pub mod variables;

use pyo3::prelude::*;

use mcts::MCTSContext;

use game_state::{env::PacmanGym, GameState};
use particle_filter::ParticleFilter;

/// Generates `FromPyObject` and `IntoPy` implementations for the given enum so
/// that it can be seamlessly converted to/from Python `int` values.
#[macro_export]
macro_rules! impl_enum_pyint_conversion {
    ($Enum:ident) => {
        impl<'source> FromPyObject<'source> for $Enum {
            fn extract(ob: &'source PyAny) -> PyResult<Self> {
                let index: u8 = ob.extract()?;
                $Enum::try_from_primitive(index).map_err(|_| {
                    ::pyo3::exceptions::PyValueError::new_err(format!(
                        concat!("Invalid ", stringify!($Enum), ": {}"),
                        index
                    ))
                })
            }
        }

        impl IntoPy<PyObject> for $Enum {
            fn into_py(self, py: Python<'_>) -> PyObject {
                u8::from(self).into_py(py)
            }
        }
    };
}

/// A Python module containing Rust implementations of the PacBot environment.
#[pymodule]
fn pacbot_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GameState>()?;
    m.add_class::<ParticleFilter>()?;
    m.add_class::<PacmanGym>()?;
    m.add_class::<MCTSContext>()?;
    m.add_function(wrap_pyfunction!(observations::create_obs_semantic, m)?)?;
    m.add_function(wrap_pyfunction!(heuristic_values::get_heuristic_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        heuristic_values::get_action_heuristic_values,
        m
    )?)?;
    Ok(())
}
