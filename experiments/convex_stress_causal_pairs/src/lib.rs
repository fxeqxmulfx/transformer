//! CFt4 at7126517f: input-only paired causal features and a declared invariant prior.
//! Discovery is discrete. Subsequent fixed-feature complex readout CE is convex.
pub use convex_stress_tagged::{C, Result, linalg, metrics};
use serde::{Deserialize, Serialize};
pub mod binding;
pub mod discovery;
pub mod features;
pub mod model;
#[cfg(test)]
mod tests;
pub mod training;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub tokens: Vec<usize>,
    pub label: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub vocabulary: usize,
    pub context: usize,
    pub observations: Vec<Observation>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub feature_cells: usize,
    pub visits: usize,
    pub attempts: usize,
    pub states_per_query: usize,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            feature_cells: 32_000_000,
            visits: 200_000_000,
            attempts: 250_000,
            states_per_query: 512,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub input: Input,
    pub policy: Policy,
}
impl Request {
    pub fn validate(&self) -> Result<u64> {
        let x = &self.input;
        if !(2..=1024).contains(&x.vocabulary)
            || !(1..=4096).contains(&x.context)
            || x.observations.is_empty()
            || x.observations.len() > 200_000
            || self.policy.feature_cells > 32_000_000
            || self.policy.visits > 200_000_000
            || self.policy.attempts > 250_000
            || !(1..=512).contains(&self.policy.states_per_query)
        {
            return Err("invalid bounded causal discovery dimensions/policy".into());
        }
        let mut tokens = 0usize;
        for s in &x.observations {
            tokens = tokens
                .checked_add(s.tokens.len())
                .ok_or("token count overflow")?;
            if s.tokens.is_empty()
                || s.tokens.len() > x.context
                || s.label >= x.vocabulary
                || s.tokens.iter().any(|&t| t >= x.vocabulary)
            {
                return Err("invalid raw categorical observation".into());
            }
        }
        if tokens > 20_000_000 {
            return Err("raw prefix storage exceeds20M tokens".into());
        }
        // Original input copies, cached categorical features, compact traces,
        // and checked serialization. No dense pair-by-example matrix is made.
        Ok(512 * 1024 * 1024
            + 64 * tokens as u64
            + 32 * self.policy.feature_cells as u64
            + 1024 * self.policy.attempts as u64)
    }
}
