//! CM62 cached_refusal_preserved: only skip an already tested full signature
//! whose cached syntax costs no more. Eviction repeats work, never drops work.
use std::collections::HashMap;

pub struct Cache {
    tested: HashMap<Vec<u64>, usize>,
    capacity: usize,
    pub hits: u64,
    pub cheaper_retests: u64,
    pub evictions: u64,
    pub peak_entries: usize,
}
impl Cache {
    pub fn new(capacity: usize) -> Self {
        Self {
            tested: HashMap::new(),
            capacity,
            hits: 0,
            cheaper_retests: 0,
            evictions: 0,
            peak_entries: 0,
        }
    }
    pub fn skip(&mut self, bits: &[u64], nodes: usize) -> bool {
        if let Some(&old) = self.tested.get(bits) {
            if old <= nodes {
                self.hits += 1;
                return true;
            }
            self.cheaper_retests += 1;
        }
        false
    }
    /// Call only after the complete indexed query has returned no fitting pair.
    pub fn record_refusal(&mut self, bits: &[u64], nodes: usize) {
        if self.capacity == 0 {
            return;
        }
        if let Some(old) = self.tested.get_mut(bits) {
            *old = (*old).min(nodes);
            return;
        }
        if self.tested.len() == self.capacity {
            self.evictions += self.tested.len() as u64;
            self.tested.clear();
        }
        self.tested.insert(bits.to_vec(), nodes);
        self.peak_entries = self.peak_entries.max(self.tested.len());
    }
}

pub fn buffer_bound(prefixes: usize, capacity: usize) -> u128 {
    // Full keys, allocator/table slack, and the next candidate; no ASTs retained.
    capacity as u128 * (prefixes.div_ceil(64) as u128 * 32 + 1024) + 1024 * 1024
}
