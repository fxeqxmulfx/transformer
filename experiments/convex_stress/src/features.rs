//! Fixed causal categorical inputs for the channel; no task identifier or label.
//! Relative tokens, completed successors, and presence are shared across all tasks.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Adapter {
    pub vocab: usize,
    pub context: usize,
}
impl Adapter {
    pub fn keys(self) -> usize {
        self.context + 3 * self.vocab
    }
    pub fn values(self) -> usize {
        self.vocab + 2
    }
    pub fn pad(self) -> usize {
        self.vocab
    }
    /// All omitted slots have PAD. Entries are sorted by key, with no duplicates.
    pub fn encode(self, prefix: &[i64]) -> (usize, Vec<(usize, usize)>) {
        assert!(!prefix.is_empty() && prefix.len() <= self.context);
        assert!(prefix.iter().all(|&x| x >= 0 && (x as usize) < self.vocab));
        let mut values = vec![self.pad(); self.keys()];
        for (lag, &token) in prefix.iter().rev().enumerate() {
            values[lag] = token as usize;
        }
        for (i, &token) in prefix.iter().enumerate() {
            let token = token as usize;
            // A successor is visible only after the successor token has occurred.
            if i + 1 < prefix.len() {
                values[self.context + token] = prefix[i + 1] as usize;
            }
            if i + 2 < prefix.len() {
                values[self.context + self.vocab + token] = prefix[i + 2] as usize;
            }
            values[self.context + 2 * self.vocab + token] = self.vocab + 1;
        }
        (
            *prefix.last().unwrap() as usize,
            values
                .into_iter()
                .enumerate()
                .filter(|&(_, v)| v != self.pad())
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_successors_are_causal_and_use_latest_completed_occurrence() {
        let adapter = Adapter {
            vocab: 10,
            context: 8,
        };
        let (_, items) = adapter.encode(&[1, 2, 3, 2]);
        let value = |k| {
            items
                .iter()
                .find(|(key, _)| *key == k)
                .map(|x| x.1)
                .unwrap_or(adapter.pad())
        };
        assert_eq!(value(0), 2);
        assert_eq!(value(3), 1);
        assert_eq!(value(4), adapter.pad());
        assert_eq!(value(8 + 2), 3); // The current 2 has no visible successor.
        assert_eq!(value(8 + 10 + 1), 3);
        assert_eq!(value(8 + 20 + 3), 11);
        assert_eq!(value(8 + 20 + 4), adapter.pad());
        assert_ne!(
            adapter.encode(&[1, 2, 3, 2]),
            adapter.encode(&[1, 2, 3, 2, 7])
        );
    }
}
