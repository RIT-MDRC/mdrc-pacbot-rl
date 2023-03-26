pub mod grid;
pub mod pacbot;
pub mod variables;

use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> String {
    (a + b).to_string()
}

/// A Python module implemented in Rust.
#[pymodule]
fn pacbot_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
