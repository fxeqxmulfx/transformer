//! Port of ConvexChannel.Forward.Minibatch at e9b0260c, CF6.
//!
//! Arrays replace Lean's functional records. An alive flag represents the outer
//! Option; records of dead candidates are ignored. Query observations are applied
//! in arrival order, retaining all candidates. No sample or target rule is stored.
//! Snapshots use the original model representation and never alter training state.

use convex_channel_forward::model::{Model, Shape};
use convex_recall_compare::data::Sample;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Streaming {
    shape: Shape,
    default: usize,
    records: Vec<Option<usize>>,
    alive: Vec<bool>,
    visits: usize,
}

impl Streaming {
    pub fn new(shape: Shape, default: usize) -> Self {
        assert!(
            [shape.queries, shape.keys, shape.values, shape.classes]
                .iter()
                .all(|&n| n > 0)
        );
        assert!(default < shape.classes);
        Self {
            shape,
            default,
            records: vec![None; shape.queries * shape.keys * shape.values],
            alive: vec![true; shape.queries * shape.keys],
            visits: 0,
        }
    }

    /// Advance every candidate; empty batches are allowed. Dimensions and class
    /// IDs must be valid. A contradiction permanently eliminates that candidate.
    pub fn update_batch(&mut self, samples: &[Sample]) {
        let s = self.shape;
        for sample in samples {
            assert_eq!(sample.values.len(), s.keys);
            assert!(sample.values.iter().all(|&v| v < s.values));
            assert_eq!(sample.queries.len(), sample.targets.len());
            for (&q, &label) in sample.queries.iter().zip(&sample.targets) {
                assert!(q < s.queries && label < s.classes);
                for (k, &value) in sample.values.iter().enumerate() {
                    self.visits += 1;
                    let candidate = q * s.keys + k;
                    if self.alive[candidate] {
                        let cell = &mut self.records[candidate * s.values + value];
                        match *cell {
                            None => *cell = Some(label),
                            Some(previous) => self.alive[candidate] = previous == label,
                        }
                    }
                }
            }
        }
    }

    /// Select the first survivor for each query. Returns the first failed query
    /// if every candidate conflicts; a default prediction is never observed data.
    pub fn snapshot(&self) -> Result<Model, usize> {
        let s = self.shape;
        let mut model = Model {
            shape: s,
            routes: Vec::with_capacity(s.queries),
            outputs: Vec::with_capacity(s.queries * s.values),
        };
        for q in 0..s.queries {
            let k = (0..s.keys).find(|&k| self.alive[q * s.keys + k]).ok_or(q)?;
            model.routes.push(k);
            let start = (q * s.keys + k) * s.values;
            model.outputs.extend(
                self.records[start..start + s.values]
                    .iter()
                    .map(|c| c.unwrap_or(self.default)),
            );
        }
        Ok(model)
    }

    /// Counts all candidate observations, including visits after elimination.
    pub fn candidate_visits(&self) -> usize {
        self.visits
    }

    /// Live array payload only, independent of the number of supplied examples.
    /// Excludes container metadata, capacity, snapshots and caller-owned data.
    pub fn state_payload_bytes(&self) -> usize {
        self.records.len() * size_of::<Option<usize>>() + self.alive.len() * size_of::<bool>()
    }
}
