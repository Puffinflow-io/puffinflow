/// Priority queue with O(1) membership checking via bitset.
///
/// Uses a BinaryHeap with lazy deletion: entries whose state index has been
/// cleared from the `in_queue` bitset are skipped on pop.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::bitset::StateBitSet;

/// Entry in the priority heap.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeapEntry {
    /// Negated priority (higher priority = more negative = pops first in max-heap)
    pub neg_priority: i32,
    /// Monotonic sequence number for FIFO within same priority
    pub sequence: u64,
    /// Index of the state
    pub state_index: usize,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap. We want lowest neg_priority first (highest
        // actual priority), then lowest sequence first (FIFO).
        other
            .neg_priority
            .cmp(&self.neg_priority)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Indexed priority queue with O(1) membership via bitset.
pub struct IndexedHeap {
    heap: BinaryHeap<HeapEntry>,
    pub in_queue: StateBitSet,
    sequence_counter: u64,
}

impl IndexedHeap {
    pub fn new(num_states: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(num_states),
            in_queue: StateBitSet::new(num_states),
            sequence_counter: 0,
        }
    }

    /// Push a state onto the heap if not already present.
    /// Returns true if it was added.
    pub fn push(&mut self, state_index: usize, priority: i32, priority_boost: i32) -> bool {
        if self.in_queue.contains(state_index) {
            return false;
        }
        self.in_queue.set(state_index);
        self.sequence_counter += 1;
        let entry = HeapEntry {
            neg_priority: -(priority + priority_boost),
            sequence: self.sequence_counter,
            state_index,
        };
        self.heap.push(entry);
        true
    }

    /// Pop the highest-priority entry that is still in the queue.
    /// Uses lazy deletion: skips entries no longer in `in_queue`.
    pub fn pop(&mut self) -> Option<HeapEntry> {
        while let Some(entry) = self.heap.pop() {
            if self.in_queue.contains(entry.state_index) {
                self.in_queue.clear(entry.state_index);
                return Some(entry);
            }
            // Stale entry — skip
        }
        None
    }

    /// Re-insert an entry (put it back without changing sequence).
    pub fn reinsert(&mut self, entry: HeapEntry) {
        self.in_queue.set(entry.state_index);
        self.heap.push(entry);
    }

    /// Remove a state from the queue (lazy — just clears the bitset).
    pub fn remove(&mut self, state_index: usize) {
        self.in_queue.clear(state_index);
    }

    /// Check if queue is logically empty.
    pub fn is_empty(&self) -> bool {
        self.in_queue.is_empty()
    }

    /// Number of states logically in queue.
    pub fn len(&self) -> usize {
        self.in_queue.count()
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.heap.clear();
        self.in_queue.clear_all();
        self.sequence_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop() {
        let mut h = IndexedHeap::new(5);
        h.push(0, 1, 0); // priority 1
        h.push(1, 3, 0); // priority 3 (highest)
        h.push(2, 2, 0); // priority 2

        let e = h.pop().unwrap();
        assert_eq!(e.state_index, 1); // highest priority first

        let e = h.pop().unwrap();
        assert_eq!(e.state_index, 2);

        let e = h.pop().unwrap();
        assert_eq!(e.state_index, 0);

        assert!(h.pop().is_none());
    }

    #[test]
    fn test_duplicate_push() {
        let mut h = IndexedHeap::new(3);
        assert!(h.push(0, 1, 0));
        assert!(!h.push(0, 1, 0)); // duplicate
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn test_lazy_deletion() {
        let mut h = IndexedHeap::new(3);
        h.push(0, 1, 0);
        h.push(1, 2, 0);
        h.remove(1); // lazy remove

        let e = h.pop().unwrap();
        assert_eq!(e.state_index, 0);
        assert!(h.pop().is_none());
    }

    #[test]
    fn test_priority_boost() {
        let mut h = IndexedHeap::new(3);
        h.push(0, 1, 0); // effective priority 1
        h.push(1, 1, 5); // effective priority 6

        let e = h.pop().unwrap();
        assert_eq!(e.state_index, 1); // boosted state first
    }
}
