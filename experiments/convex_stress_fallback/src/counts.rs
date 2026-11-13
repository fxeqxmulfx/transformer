//! Actual sparse complex embedding of the frozen CFt1-CFt3 count parent.
//! Port of AdaptiveModel::raw_logits at9496a606. Joint integer coefficients
//! have imaginary part0. A fixed indicator selects each key/value row; fixed
//! normalization and scale are linear. This is the generic bounded-feature
//! case of CM38 Basic/Domain, not a unitary-code claim about count features.
use crate::*;
use std::collections::{BTreeMap, BTreeSet};
mod state;
use state::Weights;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CountParent {
    pub(crate) features: convex_stress_counts::Aligned,
    pub(crate) scale: f64,
    shared: Weights,
    local: BTreeMap<usize, Weights>,
    total: u64,
}

impl CountParent {
    pub fn from_saved(value: serde_json::Value) -> Result<Self> {
        let model: Self = serde_json::from_value(value)?;
        model.validate()?;
        Ok(model)
    }

    /// Complex accumulation keeps the original integer merging and f64
    /// division order. The52-bit count limit makes each integer coefficient
    /// and each intermediate merged count exactly representable as a real.
    pub fn complex_logits(&self, prefix: &[i64]) -> Result<Vec<C>> {
        check_prefix(prefix, self.features.base.vocab, self.features.base.context)?;
        let v = self.features.base.vocab;
        if self.total == 0 {
            return Ok(vec![C::new(1.0 / v as f64, 0.0) * self.scale; v]);
        }
        let q = *prefix.last().unwrap() as usize;
        let local = self.local.get(&q);
        let items = self.features.encode(prefix);
        let mut logits = vec![C::new(0.0, 0.0); v];
        let missing = self.total - self.shared.total - local.map_or(0, |w| w.total);
        logits[0] = C::new(missing as f64, 0.0) / self.total as f64;
        let keys: BTreeSet<_> = self
            .shared
            .active
            .iter()
            .chain(local.into_iter().flat_map(|w| w.active.iter()))
            .copied()
            .collect();
        for key in keys {
            let parts: Vec<_> = self
                .shared
                .keys
                .get(&key)
                .into_iter()
                .chain(local.and_then(|w| w.keys.get(&key)))
                .filter(|w| w.weight > 0)
                .collect();
            let value = items
                .binary_search_by_key(&key, |x| x.0)
                .map(|i| items[i].1)
                .unwrap_or(self.features.storage().pad());
            let mut row = BTreeMap::<usize, C>::new();
            let mut zero = C::new(parts.iter().map(|w| w.weight).sum::<u64>() as f64, 0.0);
            for part in parts {
                for (&class, &count) in &part.counts[value] {
                    let coefficient = C::new(count as f64, 0.0);
                    *row.entry(class).or_default() += coefficient;
                    zero -= coefficient;
                }
            }
            logits[0] += zero / self.total as f64;
            for (class, coefficient) in row {
                logits[class] += coefficient / self.total as f64;
            }
        }
        Ok(logits.into_iter().map(|z| z * self.scale).collect())
    }
}
