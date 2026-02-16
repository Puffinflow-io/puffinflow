/// AgentCore — #[pyclass] that supersedes StateMachineCore.
///
/// Owns all agent bookkeeping: state registry, dependency graph,
/// metadata, queue, bitsets, validation cache, and retry config.
/// States are added incrementally via `add_state()`.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::bitset::StateBitSet;
use crate::heap::IndexedHeap;
use crate::metadata::{
    StateMetadataEntry, STATUS_COMPLETED, STATUS_FAILED, STATUS_RUNNING,
};

#[pyclass]
pub struct AgentCore {
    #[allow(dead_code)]
    agent_name: String,

    // State registry (incremental via add_state)
    state_names: Vec<String>,
    name_to_index: HashMap<String, usize>,

    // Dependency graph
    dependencies: Vec<Vec<usize>>,
    dependents: Vec<Vec<usize>>,

    // Tracking
    metadata: Vec<StateMetadataEntry>,
    heap: IndexedHeap,
    running: StateBitSet,
    completed: StateBitSet,
    completed_once: StateBitSet,

    // Validation cache
    validated: bool,
    cached_entry_seq: Vec<usize>,
    cached_entry_par: Vec<usize>,

    // Per-state retry config (parallel arrays)
    retry_max: Vec<u32>,
    retry_initial_delay: Vec<f64>,
    retry_exp_base: Vec<f64>,
    retry_jitter: Vec<bool>,
}

#[pymethods]
impl AgentCore {
    /// Create a new empty AgentCore. States are added via `add_state()`.
    #[new]
    fn new(agent_name: String) -> Self {
        Self {
            agent_name,
            state_names: Vec::new(),
            name_to_index: HashMap::new(),
            dependencies: Vec::new(),
            dependents: Vec::new(),
            metadata: Vec::new(),
            heap: IndexedHeap::new(0),
            running: StateBitSet::new(0),
            completed: StateBitSet::new(0),
            completed_once: StateBitSet::new(0),
            validated: false,
            cached_entry_seq: Vec::new(),
            cached_entry_par: Vec::new(),
            retry_max: Vec::new(),
            retry_initial_delay: Vec::new(),
            retry_exp_base: Vec::new(),
            retry_jitter: Vec::new(),
        }
    }

    /// Register a state, build dependency edges, grow parallel arrays.
    /// Clears the validated flag.
    #[pyo3(signature = (name, priority, max_retries, dep_names, retry_delay=1.0, retry_base=2.0, retry_jitter=true))]
    fn add_state(
        &mut self,
        name: String,
        priority: i32,
        max_retries: u32,
        dep_names: Vec<String>,
        retry_delay: f64,
        retry_base: f64,
        retry_jitter: bool,
    ) -> PyResult<()> {
        if self.name_to_index.contains_key(&name) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "State '{}' already exists",
                name
            )));
        }

        let idx = self.state_names.len();
        self.state_names.push(name.clone());
        self.name_to_index.insert(name, idx);
        self.metadata.push(StateMetadataEntry::new(priority, max_retries));

        // Resolve dependencies
        let mut deps = Vec::new();
        for dn in &dep_names {
            if let Some(&dep_idx) = self.name_to_index.get(dn) {
                deps.push(dep_idx);
                // Grow dependents if needed
                while self.dependents.len() <= dep_idx {
                    self.dependents.push(Vec::new());
                }
                self.dependents[dep_idx].push(idx);
            }
        }
        self.dependencies.push(deps);
        // Ensure dependents vec covers this index too
        while self.dependents.len() <= idx {
            self.dependents.push(Vec::new());
        }

        // Retry config
        self.retry_max.push(max_retries);
        self.retry_initial_delay.push(retry_delay);
        self.retry_exp_base.push(retry_base);
        self.retry_jitter.push(retry_jitter);

        // Invalidate
        self.validated = false;

        Ok(())
    }

    /// Iterative DFS cycle detection + cache entry states.
    fn validate(&mut self, _mode: &str) -> PyResult<()> {
        let n = self.state_names.len();
        if n == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No states defined. Agent must have at least one state to run.",
            ));
        }

        // Iterative topological sort (Kahn's algorithm) for cycle detection
        let mut in_degree = vec![0usize; n];
        for i in 0..n {
            in_degree[i] = self.dependencies[i].len();
        }
        let mut queue: Vec<usize> = Vec::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push(i);
            }
        }
        let mut visited = 0usize;
        let mut head = 0;
        while head < queue.len() {
            let node = queue[head];
            head += 1;
            visited += 1;
            if node < self.dependents.len() {
                for &dep_idx in &self.dependents[node] {
                    in_degree[dep_idx] -= 1;
                    if in_degree[dep_idx] == 0 {
                        queue.push(dep_idx);
                    }
                }
            }
        }
        if visited != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Circular dependency detected in workflow. \
                 State dependencies form a cycle, which would prevent execution.",
            ));
        }

        // Cache entry states (states with no dependencies)
        let mut entry_all: Vec<usize> = Vec::new();
        for i in 0..n {
            if self.dependencies[i].is_empty() {
                entry_all.push(i);
            }
        }

        if entry_all.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Sequential execution mode requires at least one state without dependencies \
                 to serve as an entry point. All states have dependencies, creating a deadlock.",
            ));
        }

        // Sequential: first entry state only
        self.cached_entry_seq = if entry_all.is_empty() {
            Vec::new()
        } else {
            vec![entry_all[0]]
        };

        // Parallel: all entry states
        self.cached_entry_par = entry_all;

        self.validated = true;
        Ok(())
    }

    /// Validate if needed, reset bitsets/heap/metadata, return entry state names.
    fn prepare_run(&mut self, mode: &str) -> PyResult<Vec<String>> {
        if !self.validated {
            self.validate(mode)?;
        }

        let n = self.state_names.len();

        // Reset metadata
        for m in &mut self.metadata {
            m.reset();
        }

        // Rebuild heap and bitsets for potentially changed state count
        self.heap = IndexedHeap::new(n);
        self.running = StateBitSet::new(n);
        self.completed = StateBitSet::new(n);
        self.completed_once = StateBitSet::new(n);

        // Return entry states based on mode
        let entries = if mode == "parallel" {
            &self.cached_entry_par
        } else {
            &self.cached_entry_seq
        };

        Ok(entries.iter().map(|&i| self.state_names[i].clone()).collect())
    }

    // ---- All StateMachineCore methods ----

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

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        self.metadata[idx].last_execution = now;
        self.metadata[idx].last_success = now;

        // Check dependents
        let mut newly_queued = Vec::new();
        if idx < self.dependents.len() {
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

    /// Combined execute_step: mark_running + mark_completed + handle_result + return is_done.
    /// Reduces 4 PyO3 boundary crossings to 1 per state step.
    fn execute_step(&mut self, state_name: &str, result: &Bound<'_, PyAny>) -> PyResult<bool> {
        let idx = match self.name_to_index.get(state_name) {
            Some(&i) => i,
            None => return Ok(self.heap.is_empty() && self.running.is_empty()),
        };

        // mark_completed (running was already set by get_next_ready_state or mark_running)
        self.running.clear(idx);
        self.completed.set(idx);
        self.completed_once.set(idx);
        self.metadata[idx].status = STATUS_COMPLETED;
        self.metadata[idx].attempts += 1;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        self.metadata[idx].last_execution = now;
        self.metadata[idx].last_success = now;

        // Check dependents and queue newly-ready states
        if idx < self.dependents.len() {
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
                }
            }
        }

        // handle_result
        if !result.is_none() {
            if let Ok(next_name) = result.extract::<String>() {
                if let Some(&nidx) = self.name_to_index.get(&next_name) {
                    if !self.completed_once.contains(nidx) {
                        let priority = self.metadata[nidx].priority;
                        self.heap.push(nidx, priority, 0);
                    }
                }
            } else if let Ok(names) = result.downcast::<PyList>() {
                for item in names.iter() {
                    if let Ok(ns) = item.extract::<String>() {
                        if let Some(&nidx) = self.name_to_index.get(&ns) {
                            if !self.completed_once.contains(nidx) {
                                let priority = self.metadata[nidx].priority;
                                self.heap.push(nidx, priority, 0);
                            }
                        }
                    }
                }
            }
        }

        // return is_done
        Ok(self.heap.is_empty() && self.running.is_empty())
    }

    /// Get a single next ready state (avoids Vec allocation for the common case).
    /// Returns None if no state is ready, or the state name + marks it running.
    fn get_next_ready_state(&mut self) -> Option<String> {
        let mut reinsert = Vec::new();
        let mut found = None;

        while let Some(entry) = self.heap.pop() {
            let idx = entry.state_index;
            if self.can_run(idx) {
                // Mark as running immediately
                self.running.set(idx);
                self.metadata[idx].status = STATUS_RUNNING;
                found = Some(self.state_names[idx].clone());
                break;
            } else if !self.completed_once.contains(idx) {
                reinsert.push(entry);
            }
        }

        for entry in reinsert {
            self.heap.reinsert(entry);
        }

        found
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
        self.state_names.len()
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

    // ---- Metadata accessors for proxy ----

    fn get_state_attempts(&self, state_name: &str) -> u32 {
        match self.name_to_index.get(state_name) {
            Some(&idx) => self.metadata[idx].attempts,
            None => 0,
        }
    }

    fn get_state_max_retries(&self, state_name: &str) -> u32 {
        match self.name_to_index.get(state_name) {
            Some(&idx) => self.metadata[idx].max_retries,
            None => 0,
        }
    }

    fn get_state_priority(&self, state_name: &str) -> i32 {
        match self.name_to_index.get(state_name) {
            Some(&idx) => self.metadata[idx].priority,
            None => 0,
        }
    }

    fn get_state_last_execution(&self, state_name: &str) -> Option<f64> {
        match self.name_to_index.get(state_name) {
            Some(&idx) => {
                let v = self.metadata[idx].last_execution;
                if v == 0.0 {
                    None
                } else {
                    Some(v)
                }
            }
            None => None,
        }
    }

    fn get_state_last_success(&self, state_name: &str) -> Option<f64> {
        match self.name_to_index.get(state_name) {
            Some(&idx) => {
                let v = self.metadata[idx].last_success;
                if v == 0.0 {
                    None
                } else {
                    Some(v)
                }
            }
            None => None,
        }
    }

    /// Get all registered state names.
    fn get_all_state_names(&self) -> Vec<String> {
        self.state_names.clone()
    }

    /// Get dependency names for a state.
    fn get_dependencies(&self, state_name: &str) -> Vec<String> {
        match self.name_to_index.get(state_name) {
            Some(&idx) => self.dependencies[idx]
                .iter()
                .map(|&d| self.state_names[d].clone())
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get dependent names for a state.
    fn get_dependents(&self, state_name: &str) -> Vec<String> {
        match self.name_to_index.get(state_name) {
            Some(&idx) => {
                if idx < self.dependents.len() {
                    self.dependents[idx]
                        .iter()
                        .map(|&d| self.state_names[d].clone())
                        .collect()
                } else {
                    Vec::new()
                }
            }
            None => Vec::new(),
        }
    }

    /// Check if state should be retried: attempts < max_retries.
    fn should_retry(&self, state_name: &str) -> bool {
        match self.name_to_index.get(state_name) {
            Some(&idx) => self.metadata[idx].attempts < self.retry_max[idx],
            None => false,
        }
    }

    /// Compute exponential backoff + optional jitter for a retry attempt.
    fn get_retry_delay(&self, state_name: &str, attempt: u32) -> f64 {
        match self.name_to_index.get(state_name) {
            Some(&idx) => {
                let base_delay = self.retry_initial_delay[idx]
                    * self.retry_exp_base[idx].powi(attempt as i32);
                let delay = base_delay.min(60.0);
                if self.retry_jitter[idx] {
                    // Deterministic jitter: scale by 0.75 (midpoint of 0.5..1.0)
                    // Real jitter happens Python-side if needed
                    delay * 0.75
                } else {
                    delay
                }
            }
            None => 0.0,
        }
    }
}

impl AgentCore {
    /// Check if state at index can run.
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
