//! Cache only fixed causal feature extraction; labels are consumed by minibatch updates.
use crate::{data::Example, features::Adapter};

#[derive(Clone, Debug)]
pub struct Observation {
    pub query: usize,
    pub items: Vec<(usize, usize)>,
    pub label: usize,
}
pub type Encoded = Vec<Vec<Observation>>;

pub fn encode(adapter: Adapter, examples: &[Example]) -> Encoded {
    examples
        .iter()
        .map(|e| {
            e.targets
                .iter()
                .enumerate()
                .filter(|&(_, &y)| y >= 0)
                .map(|(position, &label)| {
                    let (query, items) = adapter.encode(&e.tokens[..=position]);
                    assert_eq!(items.first(), Some(&(0, query)));
                    Observation {
                        query,
                        items,
                        label: label as usize,
                    }
                })
                .collect()
        })
        .collect()
}

impl Observation {
    pub fn value(&self, key: usize, pad: usize) -> usize {
        self.items
            .binary_search_by_key(&key, |x| x.0)
            .map(|i| self.items[i].1)
            .unwrap_or(pad)
    }
}
