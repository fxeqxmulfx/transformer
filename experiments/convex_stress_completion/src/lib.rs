//! MF5: fill missing selected-key rows without changing the oracle cost.
//! Port of adaptive-v1 (f7108742), with only zero-gradient row completion added.
//! All counts represent sums of feasible joint vertices, never trained factors.
pub use convex_stress_repair::{
    Aligned, Result, channel, data, features, hash, input, metrics, random,
};
use input::Observation;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

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
impl Oracle {
    /// Only rows absent from the query's retained observations are filled.
    /// Present rows include PAD whenever it occurred; numerical zero is not
    /// used as an absence test. MF5 proves the linear cost is unchanged.
    fn complete_missing(&mut self, pooled: &Oracle) {
        let rows = self.choices.get_mut(&self.selected).unwrap();
        let observed: BTreeSet<_> = rows.iter().map(|&(v, _)| v).collect();
        if let Some(fallback) = pooled.choices.get(&self.selected) {
            rows.extend(
                fallback
                    .iter()
                    .copied()
                    .filter(|(v, _)| !observed.contains(v)),
            );
            rows.sort_unstable_by_key(|row| row.0);
        }
    }
}
struct Stored {
    sample: Observation,
    residual: Residual,
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
struct WeightKey {
    // Unseen values retain class zero. Only consumed choices need storage.
    // This optimizer history is not required by a saved inference model.
    #[serde(skip)]
    choices: BTreeMap<usize, usize>,
    weight: u64,
    counts: Vec<BTreeMap<usize, u64>>,
}
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
struct Weights {
    keys: BTreeMap<usize, WeightKey>,
    active: BTreeSet<usize>,
    total: u64,
}
impl Weights {
    fn add(&mut self, oracle: Oracle, increment: u64, values: usize) {
        for (k, rows) in oracle.choices {
            let key = self.keys.entry(k).or_default();
            for (v, c) in rows {
                key.choices.insert(v, c);
            }
        }
        self.total = self
            .total
            .checked_add(increment)
            .expect("vertex total overflow");
        let key = self.keys.get_mut(&oracle.selected).unwrap();
        self.active.insert(oracle.selected);
        if key.counts.is_empty() {
            key.counts.resize_with(values, BTreeMap::new);
        }
        key.weight += increment;
        for (&v, &c) in &key.choices {
            if c != 0 {
                *key.counts[v].entry(c).or_default() += increment;
            }
        }
    }
    fn direct(block: &channel::QueryChannel, values: usize) -> Self {
        assert_eq!(block.routes.len(), 1);
        let (&key, &mass) = block.routes.first_key_value().unwrap();
        assert_eq!(mass, 1.);
        let choices = (0..values)
            .map(|v| {
                let class = block
                    .rows
                    .get(&(key * values + v))
                    .map(|r| {
                        let i = r.iter().position(|&x| x == 1.).unwrap();
                        assert!(r.iter().enumerate().all(|(j, &x)| j == i || x == 0.));
                        block.classes[i]
                    })
                    .unwrap_or(0);
                (v, class)
            })
            .collect();
        let mut result = Self::default();
        result.add(
            Oracle {
                selected: key,
                cost: 0.,
                choices: [(key, choices)].into(),
            },
            1,
            values,
        );
        result
    }
    fn validate(&self, a: features::Adapter) -> Result<()> {
        if self.active
            != self
                .keys
                .iter()
                .filter(|(_, key)| key.weight > 0)
                .map(|(&k, _)| k)
                .collect()
        {
            return Err("active routing index mismatch".into());
        }
        if self.keys.values().map(|k| k.weight).sum::<u64>() != self.total {
            return Err("routing counts do not sum to total".into());
        }
        for (&k, key) in &self.keys {
            if k >= a.keys()
                || key
                    .choices
                    .iter()
                    .any(|(&v, &c)| v >= a.values() || c >= a.vocab)
                || (key.weight > 0 && key.counts.len() != a.values())
            {
                return Err("invalid vertex dimensions".into());
            }
            for row in &key.counts {
                if row.keys().any(|&c| c == 0 || c >= a.vocab)
                    || row.values().sum::<u64>() > key.weight
                {
                    return Err("invalid joint counts".into());
                }
            }
        }
        Ok(())
    }
}

/// An integer-weighted sum of shared and query-specific feasible vertices.
/// Unseen queries get class-zero vertices for past unrestricted updates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveModel {
    pub features: Aligned,
    pub scale: f64,
    shared: Weights,
    local: BTreeMap<usize, Weights>,
    total: u64,
}
impl AdaptiveModel {
    fn raw_logits(&self, query: usize, items: &[(usize, usize)]) -> Vec<f64> {
        let a = self.features.storage();
        if self.total == 0 {
            return vec![1. / a.vocab as f64; a.vocab];
        }
        let local = self.local.get(&query);
        let mut z = vec![0.; a.vocab];
        let missing = self.total - self.shared.total - local.map_or(0, |w| w.total);
        z[0] = missing as f64 / self.total as f64;
        let keys: BTreeSet<_> = self
            .shared
            .active
            .iter()
            .chain(local.into_iter().flat_map(|w| w.active.iter()))
            .copied()
            .collect();
        for k in keys {
            let parts: Vec<_> = self
                .shared
                .keys
                .get(&k)
                .into_iter()
                .chain(local.and_then(|w| w.keys.get(&k)))
                .filter(|w| w.weight > 0)
                .collect();
            if parts.is_empty() {
                continue;
            }
            let v = items
                .binary_search_by_key(&k, |x| x.0)
                .map(|i| items[i].1)
                .unwrap_or(a.pad());
            let mut row = BTreeMap::<usize, u64>::new();
            let mut default: u64 = parts.iter().map(|w| w.weight).sum();
            for part in parts {
                for (&c, &count) in &part.counts[v] {
                    *row.entry(c).or_default() += count;
                    default -= count;
                }
            }
            z[0] += default as f64 / self.total as f64;
            for (c, count) in row {
                z[c] += count as f64 / self.total as f64;
            }
        }
        z
    }
    /// Exact integer invariants imply nonnegative entries and unit routing mass.
    /// Floating error is only the conversion of those same counts to logits.
    pub fn feasibility_error(&self) -> Result<f64> {
        let a = self.features.storage();
        self.shared.validate(a)?;
        if self.shared.total > self.total {
            return Err("excess shared mass".into());
        }
        for (&q, local) in &self.local {
            local.validate(a)?;
            if q >= a.vocab || self.shared.total + local.total > self.total {
                return Err("excess query mass".into());
            }
        }
        Ok(0.)
    }
}
impl metrics::Predictor for AdaptiveModel {
    fn vocab(&self) -> usize {
        self.features.base.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let query = *prefix.last().unwrap() as usize;
        self.raw_logits(query, &self.features.encode(prefix))
            .into_iter()
            .map(|z| self.scale * z)
            .collect()
    }
}

pub struct AdaptiveTrainer {
    model: AdaptiveModel,
    memory: Vec<Option<Stored>>,
    shared_gradient: Aggregate,
    local_gradient: BTreeMap<usize, Aggregate>,
    pub updates: usize,
    pub observations: usize,
    pub shared_updates: usize,
}
impl AdaptiveTrainer {
    pub fn new(features: Aligned, scale: f64, records: usize) -> Self {
        assert!(scale > 0. && scale.is_finite());
        Self {
            model: AdaptiveModel {
                features,
                scale,
                shared: Weights::default(),
                local: BTreeMap::new(),
                total: 0,
            },
            memory: (0..records).map(|_| None).collect(),
            shared_gradient: Aggregate::default(),
            local_gradient: BTreeMap::new(),
            updates: 0,
            observations: 0,
            shared_updates: 0,
        }
    }
    fn residual(&self, sample: &Observation) -> (Residual, f64, f64) {
        let z = self.model.raw_logits(sample.query, &sample.items);
        let minimum = z.iter().copied().fold(f64::INFINITY, f64::min);
        let shifted: Vec<_> = z.iter().map(|x| self.model.scale * (x - minimum)).collect();
        let denom: f64 = shifted.iter().map(|x| x.exp()).sum();
        let mut residual: Residual = shifted
            .iter()
            .enumerate()
            .filter_map(|(c, &x)| {
                let value = self.model.scale * x.exp_m1() / denom;
                (value != 0.).then_some((c, value))
            })
            .collect();
        *residual.entry(sample.label).or_default() -= self.model.scale;
        let dot = residual.iter().map(|(&c, &r)| r * z[c]).sum();
        (residual, denom.ln() - shifted[sample.label], dot)
    }
    pub fn update_batch(&mut self, batch: &[(usize, &Observation)]) {
        assert!(!batch.is_empty());
        for &(id, sample) in batch {
            assert_eq!(sample.items.first(), Some(&(0, 0)));
            let (residual, _, _) = self.residual(sample);
            let mut delta = residual.clone();
            let fresh = self.memory[id].is_none();
            if let Some(old) = &self.memory[id] {
                assert_eq!(old.sample.query, sample.query);
                assert_eq!(old.sample.label, sample.label);
                assert_eq!(old.sample.items, sample.items);
                add_scaled(&mut delta, &old.residual, -1.);
            }
            self.shared_gradient.add(sample, &delta, fresh);
            self.local_gradient
                .entry(sample.query)
                .or_default()
                .add(sample, &delta, fresh);
            self.memory[id] = Some(Stored {
                sample: sample.clone(),
                residual,
            });
        }
        let a = self.model.features.storage();
        let shared = self.shared_gradient.oracle(a.pad());
        let local: BTreeMap<_, _> = self
            .local_gradient
            .iter()
            .map(|(&q, g)| (q, g.oracle(a.pad())))
            .collect();
        let local_cost: f64 = local.values().map(|o| o.cost).sum();
        self.updates += 1;
        let increment = self.updates as u64;
        self.model.total = self
            .model
            .total
            .checked_add(increment)
            .expect("FW total overflow");
        if shared.cost <= local_cost {
            self.model.shared.add(shared, increment, a.values());
            self.shared_updates += 1;
        } else {
            for (q, mut oracle) in local {
                oracle.complete_missing(&shared);
                self.model
                    .local
                    .entry(q)
                    .or_default()
                    .add(oracle, increment, a.values());
            }
        }
        self.observations += batch.len();
    }
    /// Fresh full-training objective and unrestricted FW gap, not a memory gap.
    pub fn objective(&self, samples: &[Observation]) -> (f64, f64) {
        assert!(!samples.is_empty());
        let mut fresh = BTreeMap::<usize, Aggregate>::new();
        let (mut loss, mut dot) = (0., 0.);
        for sample in samples {
            let (residual, ce, inner) = self.residual(sample);
            fresh
                .entry(sample.query)
                .or_default()
                .add(sample, &residual, true);
            loss += ce;
            dot += inner;
        }
        let cost: f64 = fresh
            .values()
            .map(|g| g.oracle(self.model.features.storage().pad()).cost)
            .sum();
        (
            loss / samples.len() as f64,
            (dot - cost) / samples.len() as f64,
        )
    }
    pub fn seen(&self) -> usize {
        self.shared_gradient.seen
    }
    pub fn snapshot(&self) -> AdaptiveModel {
        let mut model = self.model.clone();
        model.shared.keys.retain(|_, key| key.weight > 0);
        for local in model.local.values_mut() {
            local.keys.retain(|_, key| key.weight > 0);
        }
        model
    }
}
impl metrics::Predictor for AdaptiveTrainer {
    fn vocab(&self) -> usize {
        self.model.vocab()
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.model.logits(prefix)
    }
}

/// Persistent forward fitting: shared if globally consistent, otherwise all
/// query-specific candidates. The fallback is still an exact CE fit when found.
pub struct Forward {
    features: Aligned,
    scale: f64,
    shared: convex_stress_minibatch::persistent::Persistent,
    local: convex_stress_minibatch::persistent::Persistent,
    pub updates: usize,
    pub observations: usize,
}
impl Forward {
    pub fn new(features: Aligned, scale: f64) -> Self {
        Self {
            features,
            scale,
            shared: convex_stress_minibatch::persistent::Persistent::new(features.storage()),
            local: convex_stress_minibatch::persistent::Persistent::new(features.storage()),
            updates: 0,
            observations: 0,
        }
    }
    pub fn update_batch(&mut self, samples: &[&Observation]) {
        let aligned: Vec<_> = samples
            .iter()
            .map(|s| Observation {
                query: 0,
                ..(*s).clone()
            })
            .collect();
        self.shared
            .update_batch(&aligned.iter().collect::<Vec<_>>());
        self.local.update_batch(samples);
        self.updates += 1;
        self.observations += samples.len();
    }
    pub fn snapshot(&self) -> std::result::Result<AdaptiveModel, Vec<usize>> {
        let values = self.features.storage().values();
        let mut model = AdaptiveModel {
            features: self.features,
            scale: self.scale,
            shared: Weights::default(),
            local: BTreeMap::new(),
            total: 1,
        };
        if let Ok(shared) = self.shared.snapshot() {
            model.shared = Weights::direct(&shared.queries[&0], values);
        } else {
            let local = self.local.snapshot()?;
            model.local = local
                .queries
                .iter()
                .map(|(&q, block)| (q, Weights::direct(block, values)))
                .collect();
        }
        Ok(model)
    }
}

#[cfg(test)]
mod tests;
