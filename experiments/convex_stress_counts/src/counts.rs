//! CFt1–CFt3, ConvexChannel/Counts.lean: fixed causal counts enlarge the keys.
//! Every original aligned slot keeps its key and value. Counts are raw input
//! frequencies, with no output rule, task identifier or arithmetic operation.
use crate::{features::Adapter, input::Observation};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Aligned {
    pub base: Adapter,
}

/// Separate feature values from output classes. Growing the count alphabet
/// must not add artificial classes to ordinary full-vocabulary cross-entropy.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub vocab: usize,
    context: usize,
}
impl Layout {
    pub fn old_keys(self) -> usize {
        self.context + 1 + 3 * self.vocab
    }
    pub fn keys(self) -> usize {
        self.old_keys() + self.vocab
    }
    pub fn values(self) -> usize {
        self.vocab + 2 + self.context
    }
    pub fn pad(self) -> usize {
        self.vocab
    }
    pub fn count_value(self, count: usize) -> usize {
        assert!(count <= self.context);
        if count == 0 {
            self.pad()
        } else {
            self.vocab + 1 + count
        }
    }
}
impl Aligned {
    pub fn storage(self) -> Layout {
        Layout {
            vocab: self.base.vocab,
            context: self.base.context,
        }
    }
    pub fn encode(self, prefix: &[i64]) -> Vec<(usize, usize)> {
        let mut items = convex_stress_repair::Aligned { base: self.base }.encode(prefix);
        let query = *prefix.last().unwrap() as usize;
        let mut counts = vec![0; self.base.vocab];
        for &token in prefix {
            counts[token as usize] += 1;
        }
        let a = self.storage();
        for (token, count) in counts.into_iter().enumerate() {
            if count > 0 {
                let key = a.old_keys() + (token + a.vocab - query) % a.vocab;
                items.push((key, a.count_value(count)));
            }
        }
        items.sort_unstable();
        items
    }
    pub fn observation(self, prefix: &[i64], label: usize) -> Observation {
        Observation {
            query: *prefix.last().unwrap() as usize,
            items: self.encode(prefix),
            label,
        }
    }
}
