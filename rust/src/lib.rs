//! PuffinFlow Rust extension — `puffinflow._rust_core`
//!
//! Provides `StateMachineCore`, a high-performance replacement for
//! Python-level bookkeeping in the Agent execution loop.

use pyo3::prelude::*;

mod agent_core;
mod bitset;
mod core;
mod error;
mod heap;
mod metadata;

/// Python module `_rust_core`.
#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<core::StateMachineCore>()?;
    m.add_class::<agent_core::AgentCore>()?;
    Ok(())
}
