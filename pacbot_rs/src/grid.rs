use crate::variables::GridValue::{self, *};

pub const GRID: [[GridValue; 31]; 28] = include!("grid_data.txt");

// data computed by the build script (build.rs):
pub const GRID_PELLET_COUNT: u32 = include!(concat!(env!("OUT_DIR"), "/GRID_PELLET_COUNT.txt"));
pub const GRID_POWER_PELLET_COUNT: u32 =
    include!(concat!(env!("OUT_DIR"), "/GRID_POWER_PELLET_COUNT.txt"));
