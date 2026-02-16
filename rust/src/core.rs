/// StateMachineCore — #[pyclass] with the same API as the Python fallback.
///
/// Owns all tracking state: queue, dependencies, metadata, bitsets.
/// All bookkeeping happens in Rust; only async state function calls stay in Python.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::bitset::StateBitSet;
use crate::heap::IndexedHeap;
use crate::metadata::{StateMetadataEntry, STATUS_COMPLETED, STATUS_FAILED, STATUS_RUNNING};

#[pyclass]
pub struct StateMachineCore {
    state_names: Vec<String>,
    name_to_index: HashMap<String, usize>,
    num_states: usize,

    /// Forward dependencies: state_idx -> Vec<dep_idx>
    dependencies: Vec<Vec<usize>>,
    /// Reverse dependencies: state_idx -> Vec<dependent_idx>
    dependents: Vec<Vec<usize>>,

    running: StateBitSet,
    completed: StateBitSet,
    completed_once: StateBitSet,

    heap: IndexedHeap,
    metadata: Vec<StateMetadataEntry>,
}

#[pymethods]
impl StateMachineCore {
    /// Create a new StateMachineCore.
    ///
    /// Args:
    ///     state_configs: List of (name, priority, max_retries, dep_names) tuples.
    #[new]
    fn new(state_configs: Vec<(String, i32, u32, Vec<String>)>) -> Self {
        let n = state_configs.len();

        let mut state_names = Vec::with_capacity(n);
        let mut name_to_index = HashMap::with_capacity(n);
        let mut metadata = Vec::with_capacity(n);
        let mut dependencies = vec![Vec::new(); n];
        let mut dependents = vec![Vec::new(); n];

        // First pass: register names and metadata
        for (i, (name, priority, max_retries, _)) in state_configs.iter().enumerate() {
            state_names.push(name.clone());
            name_to_index.insert(name.clone(), i);
            metadata.push(StateMetadataEntry::new(*priority, *max_retries));
        }

        // Second pass: resolve dependencies
        for (i, (_, _, _, dep_names)) in state_configs.iter().enumerate() {
            for dn in dep_names {
                if let Some(&dep_idx) = name_to_index.get(dn) {
                    dependencies[i].push(dep_idx);
                    dependents[dep_idx].push(i);
                }
            }
        }

        Self {
            state_names,
            name_to_index,
            num_states: n,
            dependencies,
            dependents,
            running: StateBitSet::new(n),
            completed: StateBitSet::new(n),
            completed_once: StateBitSet::new(n),
            heap: IndexedHeap::new(n),
            metadata,
        }
    }

    /// Add state to priority queue with O(1) membership guard.
    fn add_to_queue(&mut self, state_name: &str, priority_boost: i32) {
        if let Some(&idx) = self.name_to_index.get(state_name) {
            let priority = self.metadata[idx].priority;
            self.heap.push(idx, priority, priority_boost);
        }
    }

    /// Pop ready states from heap. Returns list of state names.
    fn get_ready_states(&mut self) -> Vec<String> {
        let mut ready = Vec::new();
        let mut reinsert = Vec::new();

        while let Some(entry) = self.heap.pop() {
            let idx = entry.state_index;
            if self.can_run(idx) {
                ready.push(self.state_names[idx].clone());
            } else if !self.completed_once.contains(idx) {
                reinsert.push(entry);
            }
            // else: already completed, just discard
        }

        for entry in reinsert {
            self.heap.reinsert(entry);
        }

        ready
    }

    /// Mark state as running.
    fn mark_running(&mut self, state_name: &str) {
        if let Some(&idx) = self.name_to_index.get(state_name) {
            self.running.set(idx);
            self.metadata[idx].status = STATUS_RUNNING;
        }
    }

    /// Mark state completed, check dependents, queue newly-ready states.
    /// Returns list of newly queued state names.
    fn mark_completed(&mut self, state_name: &str) -> Vec<String> {
        let idx = match self.name_to_index.get(state_name) {
            Some(&i) => i,
            None => return Vec::new(),
        };

        self.running.clear(idx);
        self.completed.set(idx);
        self.completed_once.set(idx);
        self.metadata[idx].status = STATUS_COMPLETED;
        self.metadata[idx].attempts += 1;

        // Check dependents
        let mut newly_queued = Vec::new();
        // Clone dependents to avoid borrow issues
        let deps = self.dependents[idx].clone();
        for dep_idx in deps {
            if self.completed_once.contains(dep_idx) {
                continue;
            }
            if self.running.contains(dep_idx) {
                continue;
            }
            if self.heap.in_queue.contains(dep_idx) {
                continue;
            }
            if self.can_run(dep_idx) {
                let priority = self.metadata[dep_idx].priority;
                self.heap.push(dep_idx, priority, 0);
                newly_queued.push(self.state_names[dep_idx].clone());
            }
        }

        newly_queued
    }

    /// Parse None/str/list[str] result and queue next states.
    fn handle_result(&mut self, _state_name: &str, result: &Bound<'_, PyAny>) -> PyResult<()> {
        if result.is_none() {
            return Ok(());
        }

        if let Ok(next_name) = result.extract::<String>() {
            if let Some(&idx) = self.name_to_index.get(&next_name) {
                if !self.completed_once.contains(idx) {
                    let priority = self.metadata[idx].priority;
                    self.heap.push(idx, priority, 0);
                }
            }
        } else if let Ok(names) = result.downcast::<PyList>() {
            for item in names.iter() {
                if let Ok(ns) = item.extract::<String>() {
                    if let Some(&idx) = self.name_to_index.get(&ns) {
                        if !self.completed_once.contains(idx) {
                            let priority = self.metadata[idx].priority;
                            self.heap.push(idx, priority, 0);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Mark state as failed.
    fn mark_failed(&mut self, state_name: &str) {
        if let Some(&idx) = self.name_to_index.get(state_name) {
            self.running.clear(idx);
            self.heap.remove(idx);
            self.metadata[idx].status = STATUS_FAILED;
        }
    }

    /// Check if all work is complete.
    fn is_done(&self) -> bool {
        self.heap.is_empty() && self.running.is_empty()
    }

    /// Check if there are states in the queue.
    fn has_queued(&self) -> bool {
        !self.heap.is_empty()
    }

    /// Get list of currently completed state names.
    fn get_completed_states(&self) -> Vec<String> {
        self.completed
            .iter_set()
            .map(|i| self.state_names[i].clone())
            .collect()
    }

    /// Get list of states completed at least once.
    fn get_completed_once(&self) -> Vec<String> {
        self.completed_once
            .iter_set()
            .map(|i| self.state_names[i].clone())
            .collect()
    }

    /// Get list of currently running state names.
    fn get_running_states(&self) -> Vec<String> {
        self.running
            .iter_set()
            .map(|i| self.state_names[i].clone())
            .collect()
    }

    /// Get status string for a state.
    fn get_state_status(&self, state_name: &str) -> String {
        match self.name_to_index.get(state_name) {
            Some(&idx) => self.metadata[idx].status_name().to_string(),
            None => "pending".to_string(),
        }
    }

    /// Number of states in queue.
    fn queue_len(&self) -> usize {
        self.heap.len()
    }

    /// Total number of states.
    fn num_states(&self) -> usize {
        self.num_states
    }

    /// Clear all tracking state.
    fn reset(&mut self) {
        for m in &mut self.metadata {
            m.reset();
        }
        self.heap.clear();
        self.running.clear_all();
        self.completed.clear_all();
        self.completed_once.clear_all();
    }
}

impl StateMachineCore {
    /// Check if state at index can run (all deps completed, not running/done).
    fn can_run(&self, idx: usize) -> bool {
        if self.running.contains(idx) {
            return false;
        }
        if self.completed_once.contains(idx) {
            return false;
        }
        self.dependencies[idx]
            .iter()
            .all(|&dep| self.completed.contains(dep))
    }
}
