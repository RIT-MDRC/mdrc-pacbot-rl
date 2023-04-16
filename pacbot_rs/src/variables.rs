use num_enum::{IntoPrimitive, TryFromPrimitive};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum Direction {
    Right = 0,
    Left = 1,
    Up = 2,
    Down = 3,
}

impl<'source> FromPyObject<'source> for Direction {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let index: u8 = ob.extract()?;
        Direction::try_from_primitive(index).map_err(|_| PyValueError::new_err("Invalid direction"))
    }
}

impl IntoPy<PyObject> for Direction {
    fn into_py(self, py: Python<'_>) -> PyObject {
        u8::from(self).into_py(py)
    }
}

/// Enum for grid cell values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum GridValue {
    /// Wall
    I = 1,
    /// Normal pellet
    o = 2,
    /// Empty space
    e = 3,
    /// Power pellet
    O = 4,
    /// Ghost chambers
    n = 5,
    /// Cherry position
    c = 6,
}

/*
# input signal enums
sig_normal = 0
sig_quit = 1
sig_restart = 2
*/

// game params
pub const STARTING_LIVES: u8 = 3;
pub const FRIGHTENED_LENGTH: u32 = 40;
pub const PELLET_SCORE: u32 = 10;
pub const POWER_PELLET_SCORE: u32 = 50;
pub const CHERRY_SCORE: u32 = 100;
pub const GHOST_SCORE: u32 = 200;
pub const STATE_SWAP_TIMES: [u32; 7] = [35, 135, 170, 270, 295, 395, 420];
pub const PACBOT_STARTING_POS: (usize, usize) = (14, 7);
pub const PACBOT_STARTING_DIR: Direction = Direction::Left;
pub const CHERRY_POS: (usize, usize) = (13, 13);
pub const GAME_FREQUENCY: f32 = 2.0;
pub const TICKS_PER_UPDATE: u32 = 12;

pub const INNER_CELL_WIDTH: f64 = 1.5;
pub const ROBOT_WIDTH: f64 = 0.75;
