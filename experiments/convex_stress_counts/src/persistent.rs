//! CF6 persistent learner, ported from convex_stress_minibatch at 8619a891.
//! Only the independent key/value dimensions and snapshot container change.
use crate::{channel::QueryChannel, counts::Layout, input::Observation};
use std::collections::{BTreeMap, BTreeSet};

type Candidates = Vec<Option<Vec<u16>>>;
pub struct Persistent {
    pub adapter: Layout,
    records: Vec<Option<Candidates>>,
    pub updates: usize,
    pub observations: usize,
    pub candidate_visits: usize,
}
impl Persistent {
    pub fn new(adapter: Layout) -> Self {
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
    pub fn snapshot(&self) -> std::result::Result<BTreeMap<usize, QueryChannel>, Vec<usize>> {
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
            Ok(queries)
        } else {
            Err(failed)
        }
    }
}
