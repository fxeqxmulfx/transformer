//! Port of ConvexChannel.Forward at a7c246fe, CF1-CF5.
//!
//! Array records replace functional updates. Examples are grouped by query,
//! candidates are enumerated in increasing order, and eliminated candidates
//! still consume the remainder of their scan, exactly as in Lean's scan.
//! The target rule is not an input. Failure has no silent fallback model.

use convex_channels::channel::Channel;
use convex_recall_compare::data::Sample;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub queries: usize,
    pub keys: usize,
    pub values: usize,
    pub classes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    pub shape: Shape,
    pub routes: Vec<usize>,
    pub outputs: Vec<usize>,
}

impl Model {
    /// CF1's finite search. The count includes every executed candidate observation.
    /// Allocation, validation and query grouping are part of this training call.
    pub fn fit(shape: Shape, samples: &[Sample], default: usize) -> Result<(Self, usize), usize> {
        assert!(
            [shape.queries, shape.keys, shape.values, shape.classes]
                .iter()
                .all(|&n| n > 0)
        );
        assert!(default < shape.classes);
        let mut rows = vec![Vec::new(); shape.queries];
        for sample in samples {
            assert_eq!(sample.values.len(), shape.keys);
            assert!(sample.values.iter().all(|&v| v < shape.values));
            assert_eq!(sample.queries.len(), sample.targets.len());
            for (&q, &label) in sample.queries.iter().zip(&sample.targets) {
                assert!(q < shape.queries && label < shape.classes);
                rows[q].push((&sample.values, label));
            }
        }
        let mut model = Self {
            shape,
            routes: Vec::with_capacity(shape.queries),
            outputs: Vec::with_capacity(shape.queries * shape.values),
        };
        let mut visits = 0;
        for (q, row) in rows.iter().enumerate() {
            let mut selected = false;
            for k in 0..shape.keys {
                let mut record = vec![None; shape.values];
                let mut alive = true;
                for &(values, label) in row {
                    visits += 1;
                    if alive {
                        let cell = &mut record[values[k]];
                        match *cell {
                            None => *cell = Some(label),
                            Some(previous) => alive = previous == label,
                        }
                    }
                }
                if alive {
                    model.routes.push(k);
                    model
                        .outputs
                        .extend(record.into_iter().map(|c| c.unwrap_or(default)));
                    selected = true;
                    break;
                }
            }
            if !selected {
                return Err(q);
            }
        }
        Ok((model, visits))
    }

    /// Prediction sees only the supplied dictionary and queries, never labels.
    pub fn predict(&self, values: &[usize], queries: &[usize]) -> Vec<usize> {
        assert_eq!(values.len(), self.shape.keys);
        queries
            .iter()
            .map(|&q| {
                let value = values[self.routes[q]];
                assert!(value < self.shape.values);
                self.outputs[q * self.shape.values + value]
            })
            .collect()
    }

    pub fn logits(&self, values: &[usize], queries: &[usize]) -> Vec<f64> {
        let predictions = self.predict(values, queries);
        let mut z = vec![0.0; queries.len() * self.shape.classes];
        for (i, &c) in predictions.iter().enumerate() {
            z[i * self.shape.classes + c] = 1.0;
        }
        z
    }

    /// Materialize precisely the joint tensor of CF3, only for diagnostics.
    pub fn to_channel(&self) -> Channel {
        let s = self.shape;
        let mut channel = Channel::new(s.queries, s.keys, s.values, s.classes);
        channel.weights.fill(0.0);
        for q in 0..s.queries {
            for v in 0..s.values {
                let c = self.outputs[q * s.values + v];
                let at = channel.index(q, self.routes[q], v, c);
                channel.weights[at] = 1.0;
            }
        }
        channel
    }

    /// Exact representation control. No threshold, label or target rule is used.
    /// A fractional tensor or value-dependent route is explicitly rejected.
    pub fn compact(channel: &Channel) -> Option<Self> {
        let shape = Shape {
            queries: channel.queries,
            keys: channel.keys,
            values: channel.values,
            classes: channel.classes,
        };
        let mut result = Self {
            shape,
            routes: Vec::new(),
            outputs: Vec::new(),
        };
        for q in 0..shape.queries {
            let k = (0..shape.keys).find(|&k| {
                (0..shape.classes).any(|c| channel.weights[channel.index(q, k, 0, c)] == 1.0)
            })?;
            result.routes.push(k);
            for v in 0..shape.values {
                result.outputs.push(
                    (0..shape.classes)
                        .find(|&c| channel.weights[channel.index(q, k, v, c)] == 1.0)?,
                );
            }
        }
        (result.to_channel().weights == channel.weights).then_some(result)
    }

    /// Live categorical payload; excludes vector headers, capacity and scratch.
    pub fn payload_bytes(&self) -> usize {
        (self.routes.len() + self.outputs.len()) * size_of::<usize>()
    }
}

pub fn dense_predict(channel: &Channel, sample: &Sample) -> Vec<usize> {
    channel
        .logits(sample)
        .chunks_exact(channel.classes)
        .map(argmax)
        .collect()
}

pub fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for c in 1..values.len() {
        if values[c] > values[best] {
            best = c;
        }
    }
    best
}
