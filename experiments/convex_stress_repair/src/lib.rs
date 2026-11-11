//! MF1–MF3: retain sample gradients and share aligned joint coefficients.
//! Source: ConvexChannel.Minibatch and ConvexChannel.Shared in this repository.
//! Stale-gradient convergence is not assumed; measure the actual full-data gap.
pub use convex_stress_minibatch::{Result, channel, data, features, hash, input, metrics, random};
use features::Adapter;
use input::Observation;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Input-only permutation: relative lags stay fixed; token-indexed blocks are
/// aligned to the visible query. A new constant slot preserves the oracle's
/// representation of all-PAD keys without discarding the old lag-zero token.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Aligned {
    pub base: Adapter,
}
impl Aligned {
    pub fn storage(self) -> Adapter {
        Adapter {
            context: self.base.context + 1,
            ..self.base
        }
    }
    pub fn encode(self, prefix: &[i64]) -> Vec<(usize, usize)> {
        let (query, items) = self.base.encode(prefix);
        let mut result = vec![(0, 0)];
        for (key, value) in items {
            let aligned = if key < self.base.context {
                key + 1
            } else {
                let position = key - self.base.context;
                self.base.context
                    + 1
                    + position / self.base.vocab * self.base.vocab
                    + (position % self.base.vocab + self.base.vocab - query) % self.base.vocab
            };
            result.push((aligned, value));
        }
        result.sort_unstable();
        result
    }
    pub fn observation(self, prefix: &[i64], label: usize) -> Observation {
        Observation {
            query: 0,
            items: self.encode(prefix),
            label,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharedModel {
    pub features: Aligned,
    pub scale: f64,
    pub channel: channel::Channel,
}
fn channel_logits(model: &channel::Channel, items: &[(usize, usize)]) -> Vec<f64> {
    let block = &model.queries[&0];
    let mut z = vec![0.; model.adapter.vocab];
    for (&key, &mass) in &block.routes {
        let value = items
            .binary_search_by_key(&key, |x| x.0)
            .map(|i| items[i].1)
            .unwrap_or(model.adapter.pad());
        if let Some(row) = block.rows.get(&(key * model.adapter.values() + value)) {
            for (&class, &weight) in block.classes.iter().zip(row) {
                z[class] += weight;
            }
        } else {
            z[0] += mass;
        }
    }
    z
}
impl metrics::Predictor for SharedModel {
    fn vocab(&self) -> usize {
        self.features.base.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        channel_logits(&self.channel, &self.features.encode(prefix))
            .into_iter()
            .map(|z| self.scale * z)
            .collect()
    }
}

type Residual = BTreeMap<usize, f64>;
#[derive(Default)]
struct GradientKey {
    present: usize,
    totals: Residual,
    rows: BTreeMap<usize, Residual>,
}
#[derive(Default)]
struct Aggregate {
    seen: usize,
    labels: BTreeSet<usize>,
    total: Residual,
    keys: BTreeMap<usize, GradientKey>,
}
fn add_scaled(target: &mut Residual, source: &Residual, factor: f64) {
    for (&c, &value) in source {
        *target.entry(c).or_default() += factor * value;
    }
}
impl Aggregate {
    fn add(&mut self, sample: &Observation, delta: &Residual, fresh: bool) {
        self.seen += usize::from(fresh);
        self.labels.insert(sample.label);
        add_scaled(&mut self.total, delta, 1.);
        for &(key, value) in &sample.items {
            let slot = self.keys.entry(key).or_default();
            slot.present += usize::from(fresh);
            add_scaled(&mut slot.totals, delta, 1.);
            add_scaled(slot.rows.entry(value).or_default(), delta, 1.);
        }
    }
    fn oracle(&self, pad: usize) -> Oracle {
        assert!(self.seen > 0);
        let mut choices = BTreeMap::new();
        let (mut selected, mut best_cost) = (0, f64::INFINITY);
        let choose = |row: &Residual| {
            self.labels
                .iter()
                .copied()
                .map(|c| (c, row.get(&c).copied().unwrap_or(0.)))
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
                .unwrap()
        };
        for (&key, slot) in &self.keys {
            let mut decoded = vec![];
            let mut cost = 0.;
            for (&value, row) in &slot.rows {
                let (class, value_cost) = choose(row);
                decoded.push((value, class));
                cost += value_cost;
            }
            if slot.present < self.seen {
                let mut padding = self.total.clone();
                add_scaled(&mut padding, &slot.totals, -1.);
                let (class, value_cost) = choose(&padding);
                decoded.push((pad, class));
                cost += value_cost;
            }
            choices.insert(key, decoded);
            if cost < best_cost {
                selected = key;
                best_cost = cost;
            }
        }
        Oracle {
            selected,
            cost: best_cost,
            choices,
        }
    }
}
struct Oracle {
    selected: usize,
    cost: f64,
    choices: BTreeMap<usize, Vec<(usize, usize)>>,
}
struct Stored {
    items: Vec<(usize, usize)>,
    label: usize,
    residual: Residual,
}
#[derive(Default)]
struct WeightKey {
    choices: Vec<usize>,
    weight: u64,
    counts: Vec<BTreeMap<usize, u64>>,
}

pub struct MemoryTrainer {
    pub features: Aligned,
    pub scale: f64,
    memory: Vec<Option<Stored>>,
    gradient: Aggregate,
    weights: BTreeMap<usize, WeightKey>,
    total: u64,
    pub updates: usize,
    pub observations: usize,
}
impl MemoryTrainer {
    pub fn new(features: Aligned, scale: f64, records: usize) -> Self {
        assert!(scale > 0. && scale.is_finite());
        Self {
            features,
            scale,
            memory: (0..records).map(|_| None).collect(),
            gradient: Aggregate::default(),
            weights: BTreeMap::new(),
            total: 0,
            updates: 0,
            observations: 0,
        }
    }
    pub fn raw_logits(&self, items: &[(usize, usize)]) -> Vec<f64> {
        let a = self.features.storage();
        if self.total == 0 {
            return vec![1. / a.vocab as f64; a.vocab];
        }
        let mut z = vec![0.; a.vocab];
        for (&k, key) in &self.weights {
            if key.weight == 0 {
                continue;
            }
            let v = items
                .binary_search_by_key(&k, |x| x.0)
                .map(|i| items[i].1)
                .unwrap_or(a.pad());
            let mut default = key.weight;
            for (&c, &weight) in &key.counts[v] {
                z[c] += weight as f64 / self.total as f64;
                default -= weight;
            }
            z[0] += default as f64 / self.total as f64;
        }
        z
    }
    /// Subtract one common softmax background. MF2 proves that this adds only
    /// a constant to a feasible linear objective, so oracle decisions and gaps
    /// are unchanged. The shift also keeps uniform initialization sparse.
    fn residual(&self, sample: &Observation) -> (Residual, f64, f64) {
        let z = self.raw_logits(&sample.items);
        let minimum = z.iter().copied().fold(f64::INFINITY, f64::min);
        let shifted: Vec<_> = z.iter().map(|x| self.scale * (x - minimum)).collect();
        let denom: f64 = shifted.iter().map(|x| x.exp()).sum();
        let mut residual: Residual = shifted
            .iter()
            .enumerate()
            .filter_map(|(c, &x)| {
                let value = self.scale * x.exp_m1() / denom;
                (value != 0.).then_some((c, value))
            })
            .collect();
        *residual.entry(sample.label).or_default() -= self.scale;
        let dot = residual.iter().map(|(&c, &r)| r * z[c]).sum();
        (residual, denom.ln() - shifted[sample.label], dot)
    }
    /// Refresh only the current minibatch's records, then take one feasible FW
    /// step using the aggregate. A complete refresh is ordinary full-batch FW.
    pub fn update_batch(&mut self, batch: &[(usize, &Observation)]) {
        assert!(!batch.is_empty());
        for &(id, sample) in batch {
            assert_eq!(sample.query, 0);
            assert_eq!(sample.items.first(), Some(&(0, 0)));
            let (residual, _, _) = self.residual(sample);
            let mut delta = residual.clone();
            let fresh = self.memory[id].is_none();
            if let Some(old) = &self.memory[id] {
                assert_eq!(old.label, sample.label, "label changed for a retained ID");
                assert_eq!(
                    old.items, sample.items,
                    "features changed for a retained ID"
                );
                add_scaled(&mut delta, &old.residual, -1.);
            }
            self.gradient.add(sample, &delta, fresh);
            self.memory[id] = Some(Stored {
                items: sample.items.clone(),
                label: sample.label,
                residual,
            });
        }
        let oracle = self.gradient.oracle(self.features.storage().pad());
        let values = self.features.storage().values();
        for (k, rows) in oracle.choices {
            let key = self.weights.entry(k).or_insert_with(|| WeightKey {
                choices: vec![0; values],
                ..WeightKey::default()
            });
            for (v, c) in rows {
                key.choices[v] = c;
            }
        }
        self.updates += 1;
        let increment = self.updates as u64;
        self.total = self
            .total
            .checked_add(increment)
            .expect("FW integer total overflow");
        let key = self.weights.get_mut(&oracle.selected).unwrap();
        if key.counts.is_empty() {
            key.counts.resize_with(values, BTreeMap::new);
        }
        key.weight += increment;
        for (v, &c) in key.choices.iter().enumerate() {
            if c != 0 {
                *key.counts[v].entry(c).or_default() += increment;
            }
        }
        self.observations += batch.len();
    }
    /// Fresh full-data CE and FW gap; no stale memory enters this certificate.
    pub fn objective(&self, samples: &[Observation]) -> (f64, f64) {
        assert!(!samples.is_empty());
        let mut fresh = Aggregate::default();
        let (mut loss, mut dot) = (0., 0.);
        for sample in samples {
            let (residual, ce, inner) = self.residual(sample);
            fresh.add(sample, &residual, true);
            loss += ce;
            dot += inner;
        }
        let oracle = fresh.oracle(self.features.storage().pad());
        (
            loss / samples.len() as f64,
            (dot - oracle.cost) / samples.len() as f64,
        )
    }
    pub fn seen(&self) -> usize {
        self.gradient.seen
    }
    pub fn snapshot(&self) -> SharedModel {
        let a = self.features.storage();
        let mut block = channel::QueryChannel {
            classes: vec![],
            routes: BTreeMap::new(),
            rows: BTreeMap::new(),
        };
        if self.total == 0 {
            block.classes = (0..a.vocab).collect();
            block.routes.insert(0, 1.);
            for v in 0..a.values() {
                block.rows.insert(v, vec![1. / a.vocab as f64; a.vocab]);
            }
        } else {
            block.classes = self
                .weights
                .values()
                .flat_map(|k| k.counts.iter().flat_map(|r| r.keys().copied()))
                .chain([0])
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            for (&k, key) in &self.weights {
                if key.weight == 0 {
                    continue;
                }
                block
                    .routes
                    .insert(k, key.weight as f64 / self.total as f64);
                for (v, counts) in key.counts.iter().enumerate() {
                    if counts.is_empty() {
                        continue;
                    }
                    let mut row = vec![0.; block.classes.len()];
                    let mut default = key.weight;
                    for (&c, &count) in counts {
                        row[block.classes.binary_search(&c).unwrap()] =
                            count as f64 / self.total as f64;
                        default -= count;
                    }
                    row[0] = default as f64 / self.total as f64;
                    block.rows.insert(k * a.values() + v, row);
                }
            }
        }
        SharedModel {
            features: self.features,
            scale: self.scale,
            channel: channel::Channel {
                adapter: a,
                queries: [(0, block)].into(),
            },
        }
    }
}
impl metrics::Predictor for MemoryTrainer {
    fn vocab(&self) -> usize {
        self.features.base.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.raw_logits(&self.features.encode(prefix))
            .into_iter()
            .map(|z| self.scale * z)
            .collect()
    }
}

#[cfg(test)]
mod tests;
