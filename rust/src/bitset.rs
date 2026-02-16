/// Compact bitset backed by Vec<u64>.
///
/// For N <= 64 states, uses a single u64 (8 bytes) vs Python set (~776 bytes
/// for 10 elements). All operations are O(1) for single-word bitsets.

#[derive(Clone, Debug)]
pub struct StateBitSet {
    words: Vec<u64>,
    len: usize,
}

impl StateBitSet {
    /// Create a new bitset that can hold `n` bits.
    pub fn new(n: usize) -> Self {
        let num_words = if n == 0 { 1 } else { (n + 63) / 64 };
        Self {
            words: vec![0u64; num_words],
            len: n,
        }
    }

    /// Set bit at position `idx`.
    #[inline]
    pub fn set(&mut self, idx: usize) {
        let word = idx / 64;
        let bit = idx % 64;
        self.words[word] |= 1u64 << bit;
    }

    /// Clear bit at position `idx`.
    #[inline]
    pub fn clear(&mut self, idx: usize) {
        let word = idx / 64;
        let bit = idx % 64;
        self.words[word] &= !(1u64 << bit);
    }

    /// Check if bit at position `idx` is set.
    #[inline]
    pub fn contains(&self, idx: usize) -> bool {
        let word = idx / 64;
        let bit = idx % 64;
        (self.words[word] & (1u64 << bit)) != 0
    }

    /// Count number of set bits.
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Check if no bits are set.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Clear all bits.
    pub fn clear_all(&mut self) {
        for w in &mut self.words {
            *w = 0;
        }
    }

    /// Iterate over set bit indices.
    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(word_idx, &word)| {
            let base = word_idx * 64;
            let max = self.len;
            (0..64).filter_map(move |bit| {
                let idx = base + bit;
                if idx < max && (word & (1u64 << bit)) != 0 {
                    Some(idx)
                } else {
                    None
                }
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        let mut bs = StateBitSet::new(10);
        assert!(bs.is_empty());
        assert_eq!(bs.count(), 0);

        bs.set(0);
        bs.set(5);
        bs.set(9);
        assert!(bs.contains(0));
        assert!(bs.contains(5));
        assert!(bs.contains(9));
        assert!(!bs.contains(3));
        assert_eq!(bs.count(), 3);

        bs.clear(5);
        assert!(!bs.contains(5));
        assert_eq!(bs.count(), 2);
    }

    #[test]
    fn test_large_bitset() {
        let mut bs = StateBitSet::new(200);
        bs.set(0);
        bs.set(64);
        bs.set(128);
        bs.set(199);
        assert_eq!(bs.count(), 4);
        assert!(bs.contains(128));

        let set_bits: Vec<usize> = bs.iter_set().collect();
        assert_eq!(set_bits, vec![0, 64, 128, 199]);
    }

    #[test]
    fn test_clear_all() {
        let mut bs = StateBitSet::new(100);
        bs.set(10);
        bs.set(50);
        bs.set(99);
        bs.clear_all();
        assert!(bs.is_empty());
        assert_eq!(bs.count(), 0);
    }
}
