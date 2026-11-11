//! Sparse-query port of the persistent CF6 learner proved at e9b0260c and
//! implemented at c1b5f814 in convex_channel_minibatch/src/streaming.rs.
//! Allocate records only when a query arrives, and discard eliminated records.
//! Every live candidate survives across minibatches; snapshots never mutate it.
use crate::{
    channel::{Channel, QueryChannel},
    features::Adapter,
    input::Observation,
};
use std::collections::{BTreeMap, BTreeSet};

type Candidates = Vec<Option<Vec<u16>>>;
pub struct Persistent {
    pub adapter: Adapter,
    records: Vec<Option<Candidates>>,
    pub updates: usize,
    pub observations: usize,
    pub candidate_visits: usize,
}
impl Persistent {
    pub fn new(adapter: Adapter) -> Self {
        assert!(adapter.vocab < u16::MAX as usize);
        Self {
            adapter,
            records: vec![None; adapter.vocab],
            updates: 0,
            observations: 0,
            candidate_visits: 0,
        }
    }
    pub fn update_batch(&mut self, samples: &[&Observation]) {
        for sample in samples {
            let candidates = self.records[sample.query].get_or_insert_with(|| {
                (0..self.adapter.keys())
                    .map(|_| Some(vec![u16::MAX; self.adapter.values()]))
                    .collect()
            });
            let mut values = vec![self.adapter.pad(); self.adapter.keys()];
            for &(k, v) in &sample.items {
                values[k] = v;
            }
            for (k, state) in candidates.iter_mut().enumerate() {
                self.candidate_visits += 1;
                let Some(record) = state else {
                    continue;
                };
                let cell = &mut record[values[k]];
                if *cell == u16::MAX {
                    *cell = sample.label as u16;
                } else if *cell as usize != sample.label {
                    *state = None;
                }
            }
        }
        self.updates += 1;
        self.observations += samples.len();
    }
    pub fn snapshot(&self) -> std::result::Result<Channel, Vec<usize>> {
        let mut queries = BTreeMap::new();
        let mut failed = vec![];
        for (q, candidates) in self.records.iter().enumerate() {
            let selected = match candidates {
                None => Some((0, None)),
                Some(states) => states
                    .iter()
                    .enumerate()
                    .find_map(|(k, state)| state.as_ref().map(|r| (k, Some(r)))),
            };
            let Some((k, record)) = selected else {
                failed.push(q);
                continue;
            };
            let classes: Vec<usize> = record
                .into_iter()
                .flat_map(|r| {
                    r.iter()
                        .copied()
                        .filter(|&c| c != u16::MAX)
                        .map(usize::from)
                })
                .chain([0])
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let mut rows = BTreeMap::new();
            if let Some(record) = record {
                for (v, &c) in record.iter().enumerate() {
                    if c == u16::MAX {
                        continue;
                    }
                    let mut weights = vec![0.; classes.len()];
                    weights[classes.binary_search(&(c as usize)).unwrap()] = 1.;
                    rows.insert(k * self.adapter.values() + v, weights);
                }
            }
            queries.insert(
                q,
                QueryChannel {
                    classes,
                    routes: [(k, 1.)].into(),
                    rows,
                },
            );
        }
        if failed.is_empty() {
            Ok(Channel {
                adapter: self.adapter,
                queries,
            })
        } else {
            Err(failed)
        }
    }
}

/// Exact deterministic tensor signature, including unobserved values and queries.
/// Rejects fractional channels; never converts a soft model into a hard one.
pub fn signature(model: &Channel) -> Option<Vec<(usize, Vec<usize>)>> {
    let mut result = vec![];
    for q in 0..model.adapter.vocab {
        let channel = model.queries.get(&q)?;
        if channel.routes.len() != 1 {
            return None;
        }
        let (&k, &mass) = channel.routes.first_key_value()?;
        if mass != 1. {
            return None;
        }
        let mut decoder = vec![0; model.adapter.values()];
        for (v, label) in decoder.iter_mut().enumerate() {
            if let Some(row) = channel.rows.get(&(k * model.adapter.values() + v)) {
                let i = row.iter().position(|&x| x == 1.)?;
                if row.iter().enumerate().any(|(j, &x)| j != i && x != 0.) {
                    return None;
                }
                *label = channel.classes[i];
            }
        }
        result.push((k, decoder));
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{data::Example, forward, input};
    #[test]
    fn all_partial_boolean_data_and_batch_boundaries_match_complete_fitting() {
        let adapter = Adapter {
            vocab: 3,
            context: 3,
        };
        for code in 0..81usize {
            let mut state = code;
            let mut examples = vec![];
            for a in 0..2 {
                for b in 0..2 {
                    let digit = state % 3;
                    state /= 3;
                    if digit != 0 {
                        examples.push(Example::generation(
                            "witness",
                            vec![a, b, 2],
                            vec![(digit - 1) as i64],
                        ));
                    }
                }
            }
            let encoded = input::encode(adapter, &examples);
            for size in [1, 2, 3, 7] {
                let mut learner = Persistent::new(adapter);
                learner.update_batch(&[]);
                for (batch, chunk) in encoded.chunks(size).enumerate() {
                    learner.update_batch(&chunk.iter().flatten().collect::<Vec<_>>());
                    let prefix = &examples[..((batch + 1) * size).min(examples.len())];
                    let (reference, report) = forward::fit(adapter, prefix, 0);
                    match (learner.snapshot(), reference) {
                        (Ok(model), Some(reference)) => {
                            assert_eq!(signature(&model), signature(&reference))
                        }
                        (Err(failed), None) => {
                            assert_eq!(serde_json::json!(failed), report["failed_queries"])
                        }
                        _ => panic!("prefix completion disagreement"),
                    }
                }
                assert_eq!(
                    learner.candidate_visits,
                    learner.observations * adapter.keys()
                );
                let (reference, report) = forward::fit(adapter, &examples, 0);
                match (learner.snapshot(), reference) {
                    (Ok(model), Some(reference)) => {
                        assert_eq!(signature(&model), signature(&reference))
                    }
                    (Err(failed), None) => {
                        assert_eq!(serde_json::json!(failed), report["failed_queries"])
                    }
                    _ => panic!("final completion disagreement"),
                }
            }
        }
    }
}
