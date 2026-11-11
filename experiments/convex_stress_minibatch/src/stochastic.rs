//! Stochastic block Frank-Wolfe on the same joint tensor, updated every minibatch.
//! Exact weighted vertex counts represent W directly. They are never independent
//! routing and decoder parameters. Zero-gradient values retain the previous oracle
//! class as a valid tie choice; unseen queries receive no update.
use crate::{
    channel::{Channel, QueryChannel},
    features::Adapter,
    input::Observation,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Key {
    choices: Vec<usize>,
    weight: u64,
    nonzero: Vec<BTreeMap<usize, u64>>,
}
impl Key {
    fn new(values: usize) -> Self {
        Self {
            choices: vec![0; values],
            weight: 0,
            nonzero: vec![],
        }
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Block {
    visits: u64,
    total: u64,
    keys: BTreeMap<usize, Key>,
}
impl Block {
    fn logits(&self, adapter: Adapter, items: &[(usize, usize)]) -> Vec<f64> {
        if self.total == 0 {
            return vec![1. / adapter.vocab as f64; adapter.vocab];
        }
        let mut logits = vec![0.; adapter.vocab];
        for (&k, key) in &self.keys {
            if key.weight == 0 {
                continue;
            }
            let value = items
                .binary_search_by_key(&k, |x| x.0)
                .map(|i| items[i].1)
                .unwrap_or(adapter.pad());
            let row = &key.nonzero[value];
            let mut zero = key.weight;
            for (&class, &weight) in row {
                logits[class] += weight as f64 / self.total as f64;
                zero -= weight;
            }
            logits[0] += zero as f64 / self.total as f64;
        }
        logits
    }
}

#[derive(Default)]
struct GradientRow {
    count: usize,
    values: Vec<f64>,
}
struct Oracle {
    key: usize,
    choices: BTreeMap<usize, Vec<(usize, usize)>>,
    loss: f64,
    dot: f64,
    cost: f64,
}

/// Classes absent from minibatch labels have nonnegative gradient on every
/// observed row, so a minimum lies among the supplied labels. The full vocabulary
/// still participates in the softmax denominator. All-PAD candidate keys have the
/// same cost as the constant lag-zero key, whose index is the smallest possible.
fn oracle(adapter: Adapter, block: &Block, samples: &[&Observation]) -> Oracle {
    assert!(!samples.is_empty());
    let classes: Vec<_> = samples
        .iter()
        .map(|s| s.label)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut rows: BTreeMap<usize, BTreeMap<usize, GradientRow>> = BTreeMap::new();
    let mut total = vec![0.; classes.len()];
    let mut loss = 0.;
    let mut dot = 0.;
    for sample in samples {
        assert_eq!(sample.items.first(), Some(&(0, sample.query)));
        assert_eq!(sample.query, samples[0].query);
        let z = block.logits(adapter, &sample.items);
        let denominator = z.iter().map(|x| x.exp()).sum::<f64>();
        loss += denominator.ln() - z[sample.label];
        dot += z.iter().map(|&x| x.exp() * x / denominator).sum::<f64>() - z[sample.label];
        let errors: Vec<_> = classes
            .iter()
            .map(|&c| z[c].exp() / denominator - f64::from(c == sample.label))
            .collect();
        for (sum, &error) in total.iter_mut().zip(&errors) {
            *sum += error;
        }
        for &(k, v) in &sample.items {
            let row = rows
                .entry(k)
                .or_default()
                .entry(v)
                .or_insert_with(|| GradientRow {
                    count: 0,
                    values: vec![0.; classes.len()],
                });
            row.count += 1;
            for (value, &error) in row.values.iter_mut().zip(&errors) {
                *value += error;
            }
        }
    }
    let mut choices = BTreeMap::new();
    let (mut best_key, mut best_cost) = (0, f64::INFINITY);
    for (key, mut values) in rows {
        let present: usize = values.values().map(|row| row.count).sum();
        if present < samples.len() {
            let mut padding = total.clone();
            for row in values.values() {
                for (entry, &value) in padding.iter_mut().zip(&row.values) {
                    *entry -= value;
                }
            }
            values.insert(
                adapter.pad(),
                GradientRow {
                    count: samples.len() - present,
                    values: padding,
                },
            );
        }
        let mut decoded = vec![];
        let mut cost = 0.;
        for (value, row) in values {
            let best = (0..classes.len())
                .min_by(|&a, &b| row.values[a].total_cmp(&row.values[b]).then(a.cmp(&b)))
                .unwrap();
            decoded.push((value, classes[best]));
            cost += row.values[best];
        }
        choices.insert(key, decoded);
        if cost < best_cost {
            best_key = key;
            best_cost = cost;
        }
    }
    Oracle {
        key: best_key,
        choices,
        loss,
        dot,
        cost: best_cost,
    }
}

pub struct Stochastic {
    pub adapter: Adapter,
    blocks: BTreeMap<usize, Block>,
    pub updates: usize,
    pub observations: usize,
}
impl Stochastic {
    pub fn predict(&self, prefix: &[i64]) -> Vec<f64> {
        let (q, items) = self.adapter.encode(prefix);
        match self.blocks.get(&q) {
            Some(block) => block.logits(self.adapter, &items),
            None => vec![1. / self.adapter.vocab as f64; self.adapter.vocab],
        }
    }
    pub fn new(adapter: Adapter) -> Self {
        Self {
            adapter,
            blocks: BTreeMap::new(),
            updates: 0,
            observations: 0,
        }
    }
    pub fn update_batch(&mut self, batch: &[&Observation]) {
        assert!(!batch.is_empty());
        let mut groups: BTreeMap<usize, Vec<&Observation>> = BTreeMap::new();
        for sample in batch {
            groups.entry(sample.query).or_default().push(sample);
        }
        for (q, samples) in groups {
            let block = self.blocks.entry(q).or_default();
            let selected = oracle(self.adapter, block, &samples);
            // Remember all computed oracle choices; unobserved values have zero
            // current gradient, so retaining their earlier choice is still exact.
            for (k, choices) in selected.choices {
                let key = block
                    .keys
                    .entry(k)
                    .or_insert_with(|| Key::new(self.adapter.values()));
                for (v, c) in choices {
                    key.choices[v] = c;
                }
            }
            block.visits += 1;
            let increment = block.visits;
            block.total += increment;
            let key = block.keys.get_mut(&selected.key).unwrap();
            if key.nonzero.is_empty() {
                key.nonzero
                    .resize_with(self.adapter.values(), BTreeMap::new);
            }
            key.weight += increment;
            for (v, &c) in key.choices.iter().enumerate() {
                if c != 0 {
                    *key.nonzero[v].entry(c).or_default() += increment;
                }
            }
        }
        self.updates += 1;
        self.observations += batch.len();
    }
    pub fn snapshot(&self) -> Channel {
        let mut queries = BTreeMap::new();
        for (&q, block) in &self.blocks {
            let classes: Vec<_> = block
                .keys
                .values()
                .flat_map(|k| k.nonzero.iter().flat_map(|r| r.keys().copied()))
                .chain([0])
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let mut routes = BTreeMap::new();
            let mut rows = BTreeMap::new();
            for (&k, key) in &block.keys {
                if key.weight == 0 {
                    continue;
                }
                routes.insert(k, key.weight as f64 / block.total as f64);
                for (v, counts) in key.nonzero.iter().enumerate() {
                    if counts.is_empty() {
                        continue;
                    }
                    let mut weights = vec![0.; classes.len()];
                    let mut zero = key.weight;
                    for (&class, &count) in counts {
                        zero -= count;
                        weights[classes.binary_search(&class).unwrap()] =
                            count as f64 / block.total as f64;
                    }
                    weights[0] = zero as f64 / block.total as f64;
                    rows.insert(k * self.adapter.values() + v, weights);
                }
            }
            queries.insert(
                q,
                QueryChannel {
                    classes,
                    routes,
                    rows,
                },
            );
        }
        Channel {
            adapter: self.adapter,
            queries,
        }
    }
    pub fn objective(&self, samples: &[&Observation]) -> (f64, f64) {
        let mut groups: BTreeMap<usize, Vec<&Observation>> = BTreeMap::new();
        for sample in samples {
            groups.entry(sample.query).or_default().push(sample);
        }
        let (mut loss, mut gap) = (0., 0.);
        for (q, group) in groups {
            let empty = Block::default();
            let block = self.blocks.get(&q).unwrap_or(&empty);
            let evaluation = oracle(self.adapter, block, &group);
            loss += evaluation.loss;
            gap += evaluation.dot - evaluation.cost;
        }
        (loss / samples.len() as f64, gap / samples.len() as f64)
    }
}

impl crate::metrics::Predictor for Stochastic {
    fn vocab(&self) -> usize {
        self.adapter.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.predict(prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::cross_entropy;

    fn observations(adapter: Adapter) -> Vec<Observation> {
        (0..2)
            .flat_map(|a| {
                (0..2).map(move |b| {
                    let (query, items) = adapter.encode(&[a, b, 2]);
                    Observation {
                        query,
                        items,
                        label: (a ^ b) as usize,
                    }
                })
            })
            .collect()
    }
    fn dense(model: &Channel, query: usize) -> Vec<f64> {
        let a = model.adapter;
        let Some(block) = model.queries.get(&query) else {
            return vec![1. / (a.keys() * a.vocab) as f64; a.keys() * a.values() * a.vocab];
        };
        let mut weights = vec![0.; a.keys() * a.values() * a.vocab];
        for (&k, &mass) in &block.routes {
            for v in 0..a.values() {
                if let Some(row) = block.rows.get(&(k * a.values() + v)) {
                    for (&c, &w) in block.classes.iter().zip(row) {
                        weights[(k * a.values() + v) * a.vocab + c] = w;
                    }
                } else {
                    weights[(k * a.values() + v) * a.vocab] = mass;
                }
            }
        }
        weights
    }
    fn dense_logits(a: Adapter, weights: &[f64], s: &Observation) -> Vec<f64> {
        (0..a.vocab)
            .map(|c| {
                (0..a.keys())
                    .map(|k| weights[(k * a.values() + s.value(k, a.pad())) * a.vocab + c])
                    .sum()
            })
            .collect()
    }
    fn dense_loss(a: Adapter, weights: &[f64], samples: &[Observation]) -> f64 {
        samples
            .iter()
            .map(|s| cross_entropy(&dense_logits(a, weights, s), s.label))
            .sum()
    }
    #[test]
    fn sparse_batch_oracle_matches_full_gradient_and_exhaustive_vertices() {
        let a = Adapter {
            vocab: 3,
            context: 3,
        };
        let samples = observations(a);
        let refs: Vec<_> = samples.iter().collect();
        let mut learner = Stochastic::new(a);
        learner.update_batch(&refs[..2]);
        learner.update_batch(&refs[2..]);
        let weights = dense(&learner.snapshot(), 2);
        let mut gradient = vec![0.; weights.len()];
        for s in &samples {
            let logits = dense_logits(a, &weights, s);
            let denom = logits.iter().map(|x| x.exp()).sum::<f64>();
            for k in 0..a.keys() {
                for c in 0..a.vocab {
                    gradient[(k * a.values() + s.value(k, a.pad())) * a.vocab + c] +=
                        logits[c].exp() / denom - f64::from(c == s.label);
                }
            }
        }
        for (i, &expected) in gradient.iter().enumerate() {
            let mut changed = weights.clone();
            changed[i] += 1e-6;
            let plus = dense_loss(a, &changed, &samples);
            changed[i] -= 2e-6;
            let minus = dense_loss(a, &changed, &samples);
            assert!(
                ((plus - minus) / 2e-6 - expected).abs() < 2e-8,
                "coordinate {i}"
            );
        }
        let actual = oracle(a, &learner.blocks[&2], &refs);
        let mut best = f64::INFINITY;
        for k in 0..a.keys() {
            for assignment in 0..a.vocab.pow(a.values() as u32) {
                let cost: f64 = (0..a.values())
                    .map(|v| {
                        let c = assignment / a.vocab.pow(v as u32) % a.vocab;
                        gradient[(k * a.values() + v) * a.vocab + c]
                    })
                    .sum();
                best = best.min(cost);
            }
        }
        assert!((best - actual.cost).abs() < 1e-12);
        let dot: f64 = gradient.iter().zip(&weights).map(|(g, w)| g * w).sum();
        assert!((dot - actual.dot).abs() < 1e-12);
        assert!((actual.loss - dense_loss(a, &weights, &samples)).abs() < 1e-12);
    }
    #[test]
    fn each_minibatch_is_the_exact_joint_tensor_convex_step() {
        let a = Adapter {
            vocab: 3,
            context: 3,
        };
        let samples = observations(a);
        let refs: Vec<_> = samples.iter().collect();
        let mut learner = Stochastic::new(a);
        for batch in [&refs[..1], &refs[1..3], &refs[3..]] {
            let before = dense(&learner.snapshot(), 2);
            let old = learner.blocks.get(&2).cloned().unwrap_or_default();
            let selected = oracle(a, &old, batch);
            let mut decoder = old
                .keys
                .get(&selected.key)
                .map(|k| k.choices.clone())
                .unwrap_or_else(|| vec![0; a.values()]);
            for &(v, c) in &selected.choices[&selected.key] {
                decoder[v] = c;
            }
            let eta = 2. / (old.visits + 2) as f64;
            learner.update_batch(batch);
            let model = learner.snapshot();
            assert!(model.feasibility_error().unwrap() < 1e-14);
            let after = dense(&model, 2);
            for k in 0..a.keys() {
                for (v, &c_star) in decoder.iter().enumerate() {
                    for c in 0..a.vocab {
                        let i = (k * a.values() + v) * a.vocab + c;
                        let vertex = f64::from(k == selected.key && c == c_star);
                        assert!((after[i] - ((1. - eta) * before[i] + eta * vertex)).abs() < 1e-14);
                    }
                }
            }
            for s in &samples {
                let (q, items) =
                    a.encode(&[s.value(2, a.pad()) as i64, s.value(1, a.pad()) as i64, 2]);
                let observed = Observation {
                    query: q,
                    items,
                    label: s.label,
                };
                assert_eq!(
                    learner.blocks[&2].logits(a, &observed.items),
                    model.logits(&[s.value(2, a.pad()) as i64, s.value(1, a.pad()) as i64, 2])
                );
            }
        }
        assert_eq!(learner.updates, 3);
        assert_eq!(learner.observations, 4);
    }
    #[test]
    fn absent_queries_keep_their_state_and_prior_oracle_choices_survive() {
        let a = Adapter {
            vocab: 3,
            context: 3,
        };
        let samples = observations(a);
        let mut learner = Stochastic::new(a);
        learner.update_batch(&[&samples[1]]);
        let retained = learner.blocks[&2].clone();
        let (query, items) = a.encode(&[2, 1]);
        let other = Observation {
            query,
            items,
            label: 0,
        };
        learner.update_batch(&[&other]);
        assert_eq!(learner.blocks[&2], retained);
        // A new observation does not clear every unobserved value's oracle choice.
        let old_choices = learner.blocks[&2].keys[&1].choices.clone();
        learner.update_batch(&[&samples[0]]);
        assert_eq!(learner.blocks[&2].keys[&1].choices[1], old_choices[1]);
    }
}
