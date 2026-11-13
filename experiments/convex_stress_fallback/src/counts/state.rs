//! Saved integer joint coefficients from convex_stress_counts at9496a606.
//! The same categorical indicators have bound1. Normalization is fixed before
//! using this parent; neither its routing weights nor its decoder are trained.
use super::*;

const MAX_EXACT: u64 = 1 << 52;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Key {
    pub weight: u64,
    pub counts: Vec<BTreeMap<usize, u64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Weights {
    pub keys: BTreeMap<usize, Key>,
    pub active: BTreeSet<usize>,
    pub total: u64,
}

impl Weights {
    fn check(&self, layout: convex_stress_counts::counts::Layout) -> Result<(u64, u64)> {
        if self.total > MAX_EXACT || self.keys.len() > layout.keys() {
            return Err("count parent total or key count exceeds declared bound".into());
        }
        let active: BTreeSet<_> = self
            .keys
            .iter()
            .filter(|(_, k)| k.weight > 0)
            .map(|(&k, _)| k)
            .collect();
        if active != self.active {
            return Err("count parent active keys differ from joint weights".into());
        }
        let (mut mass, mut rows, mut stored) = (0u64, 0u64, 0u64);
        for (&k, key) in &self.keys {
            if k >= layout.keys()
                || key.weight > MAX_EXACT
                || (key.weight == 0 && !key.counts.is_empty())
                || (key.weight > 0 && key.counts.len() != layout.values())
            {
                return Err("count parent key/value dimensions differ".into());
            }
            mass = mass.checked_add(key.weight).ok_or("joint mass overflow")?;
            stored += 1;
            rows += key.counts.len() as u64;
            for row in &key.counts {
                let mut sum = 0u64;
                for (&c, &count) in row {
                    if c == 0 || c >= layout.vocab || count > key.weight {
                        return Err("invalid count parent joint coordinate".into());
                    }
                    sum = sum.checked_add(count).ok_or("joint row overflow")?;
                    stored += 1;
                }
                if sum > key.weight {
                    return Err("count parent row exceeds routing weight".into());
                }
            }
        }
        if mass != self.total {
            return Err("count parent routing mass differs from total".into());
        }
        Ok((rows, stored))
    }
}

impl CountParent {
    pub fn validate(&self) -> Result<Bounds> {
        let (v, context) = (self.features.base.vocab, self.features.base.context);
        if !(2..=1024).contains(&v)
            || !(1..=4096).contains(&context)
            || !self.scale.is_finite()
            || !(0.0..=64.0).contains(&self.scale)
            || self.total > MAX_EXACT
            || self.local.len() > v
        {
            return Err("invalid frozen count parent dimensions or scale".into());
        }
        let layout = self.features.storage();
        let (mut rows, mut stored) = self.shared.check(layout)?;
        if self.shared.total > self.total {
            return Err("count parent shared mass exceeds total".into());
        }
        for (&q, weights) in &self.local {
            let (r, s) = weights.check(layout)?;
            if q >= v || weights.total > self.total - self.shared.total {
                return Err("count parent query mass exceeds total".into());
            }
            rows += r;
            stored += s + 1;
        }
        // One constant uniform row and one missing-mass row per possible query
        // are charged even when their coefficients vanish. Each indicator row
        // has an entire vocabulary of complex coefficients, including zeros.
        rows += 1 + v as u64;
        let real_slots = 2 * rows * v as u64;
        stored += 3 + v as u64;
        // Summing all nonnegative class coefficients bounds any one column.
        // Shared mass<=T and each local mass<=T imply this explicit radius.
        let radius = self.scale * (v as f64 + layout.values() as f64 * (1.0 + v as f64));
        let buffer_bytes = 512 * 1024 * 1024 + 128 * (stored + rows);
        if buffer_bytes > BUFFER_LIMIT {
            return Err("frozen count parent exceeds buffer forecast".into());
        }
        Ok(Bounds {
            real_slots,
            stored_scalars: stored,
            radius,
            buffer_bytes,
        })
    }

    /// Sum every actual nonnegative normalized joint coefficient by class.
    /// Unstored class-zero coordinates are reconstructed from the saved mass.
    pub fn column_l1(&self) -> Result<Vec<f64>> {
        self.validate()?;
        let v = self.features.base.vocab;
        if self.total == 0 {
            return Ok(vec![self.scale / v as f64; v]);
        }
        let mut counts = vec![0u128; v];
        for q in 0..v {
            counts[0] += (self.total
                - self.shared.total
                - self.local.get(&q).map_or(0, |w| w.total)) as u128;
        }
        for weights in std::iter::once(&self.shared).chain(self.local.values()) {
            for key in weights.keys.values().filter(|k| k.weight > 0) {
                for row in &key.counts {
                    let mut zero = key.weight;
                    for (&c, &count) in row {
                        counts[c] += count as u128;
                        zero -= count;
                    }
                    counts[0] += zero as u128;
                }
            }
        }
        Ok(counts
            .into_iter()
            .map(|n| self.scale * (n as f64 / self.total as f64))
            .collect())
    }
}
