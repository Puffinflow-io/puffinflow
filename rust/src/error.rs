/// Error types and PyErr mappings.

use pyo3::exceptions::PyValueError;
use pyo3::PyErr;

/// Convert a state-not-found condition into a Python ValueError.
#[allow(dead_code)]
pub fn state_not_found(name: &str) -> PyErr {
    PyValueError::new_err(format!("State '{}' not found", name))
}
