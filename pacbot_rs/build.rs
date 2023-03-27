//! Build script that precomputes various things from the game grid.

use std::env;
use std::fs;
use std::io;
use std::path::Path;

#[derive(PartialEq, Eq)]
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
use GridValue::*;

pub const GRID: [[GridValue; 31]; 28] = include!("src/grid_data.txt");

fn output_count<P: AsRef<Path>>(cell_type: GridValue, out_path: P) -> io::Result<()> {
    let count = GRID.iter().flatten().filter(|v| **v == cell_type).count();
    fs::write(out_path, count.to_string())
}

fn main() -> io::Result<()> {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);

    output_count(o, out_dir.join("GRID_PELLET_COUNT.txt"))?;
    output_count(O, out_dir.join("GRID_POWER_PELLET_COUNT.txt"))?;

    println!("cargo:rerun-if-changed=src/grid_data.txt");
    Ok(())
}
