//! Unfactorized joint channel coefficients, stored only on active categorical rows.
//! Each row sums to its key's common routing mass; unobserved values output class 0.
//! Source: ConvexChannel.Defs, Basic and Oracle in the accompanying Lean development.
use crate::{Result, features::Adapter};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryChannel {
    pub classes: Vec<usize>,
    pub routes: BTreeMap<usize, f64>,
    pub rows: BTreeMap<usize, Vec<f64>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Channel {
    pub adapter: Adapter,
    pub queries: BTreeMap<usize, QueryChannel>,
}
impl Channel {
    pub fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let (q, features) = self.adapter.encode(prefix);
        let Some(channel) = self.queries.get(&q) else {
            return vec![1. / self.adapter.vocab as f64; self.adapter.vocab];
        };
        let mut logits = vec![0.; self.adapter.vocab];
        for (&key, &mass) in &channel.routes {
            let value = features
                .binary_search_by_key(&key, |x| x.0)
                .map(|i| features[i].1)
                .unwrap_or(self.adapter.pad());
            if let Some(row) = channel.rows.get(&(key * self.adapter.values() + value)) {
                for (&class, &weight) in channel.classes.iter().zip(row) {
                    logits[class] += weight;
                }
            } else {
                logits[0] += mass;
            }
        }
        logits
    }

    pub fn feasibility_error(&self) -> Result<f64> {
        let mut error = 0f64;
        for (&q, channel) in &self.queries {
            if q >= self.adapter.vocab
                || channel.classes.first() != Some(&0)
                || channel.classes.windows(2).any(|w| w[0] >= w[1])
                || channel
                    .classes
                    .last()
                    .is_none_or(|&c| c >= self.adapter.vocab)
            {
                return Err("invalid query or class indices".into());
            }
            error = error.max((channel.routes.values().sum::<f64>() - 1.).abs());
            for (&key, &mass) in &channel.routes {
                if key >= self.adapter.keys() || !mass.is_finite() {
                    return Err("invalid routing row".into());
                }
                error = error.max(-mass);
            }
            for (&feature, weights) in &channel.rows {
                let key = feature / self.adapter.values();
                let mass = channel.routes.get(&key).ok_or("row without routing mass")?;
                if weights.len() != channel.classes.len() || weights.iter().any(|x| !x.is_finite())
                {
                    return Err("invalid joint coefficient row".into());
                }
                error = error.max((weights.iter().sum::<f64>() - mass).abs());
                for &weight in weights {
                    error = error.max(-weight);
                }
            }
        }
        Ok(error)
    }
}

pub fn argmax(values: &[f64]) -> usize {
    (0..values.len())
        .max_by(|&a, &b| values[a].total_cmp(&values[b]).then(b.cmp(&a)))
        .unwrap()
}

pub fn cross_entropy(logits: &[f64], label: usize) -> f64 {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + logits.iter().map(|&x| (x - max).exp()).sum::<f64>().ln() - logits[label]
}
