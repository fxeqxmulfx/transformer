//! Full CM64 joint decoder core after9048ce88 and the exact block source
//! afteraa1d0371. Native arithmetic/projections are audited computations, not
//! a claimed Lean refinement. Programs and raw data adapters live outside it.
use serde::{Deserialize, Serialize};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const MAX_BYTES: u128 = 48 * 1024u128.pow(3);
pub const MAX_WORK: u64 = 40_000_000_000_000;
pub mod domain;
pub mod head;
pub mod projection;
#[cfg(test)]
mod tests;
pub mod training;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shape {
    pub programs: usize,
    pub states: usize,
    pub classes: usize,
}
impl Shape {
    pub fn cells(self) -> usize {
        self.programs * self.states * self.classes
    }
    pub fn parameters(self) -> usize {
        self.programs + self.cells()
    }
    pub fn at(self, p: usize, s: usize, c: usize) -> usize {
        self.programs + (p * self.states + s) * self.classes + c
    }
    pub fn admit(self, records: usize) -> Result<u64> {
        if !(1..=65536).contains(&self.programs)
            || !(4..=4096).contains(&self.states)
            || !self.states.is_multiple_of(4)
            || !(2..=1024).contains(&self.classes)
            || records > 65536
        {
            return Err("invalid bounded joint shape or record count".into());
        }
        // Sixteen full parameter arrays cover current/proposed/gradient,
        // projection scratch and complete real/imaginary heads. Encoded
        // original records are charged separately, with 1GiB runtime slack.
        let bytes = 16 * self.parameters() as u128 * 8
            + records as u128 * (self.programs as u128 * 8 + 64)
            + 1024u128.pow(3);
        if bytes > MAX_BYTES {
            return Err("bounded joint buffers exceed48GiB before allocation".into());
        }
        Ok(bytes as u64)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    /// Shared masses first, followed by every gamma[p,state,raw_class].
    pub values: Vec<f64>,
}
impl Parameters {
    pub fn validate(&self, shape: Shape) -> Result<()> {
        shape.admit(0)?;
        if self.values.len() != shape.parameters()
            || self.values.iter().any(|x| !x.is_finite() || x.abs() > 1e12)
        {
            return Err("invalid complete joint parameter tensor".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub states: Vec<usize>,
    pub label: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub used: u64,
    pub limit: u64,
}
impl Work {
    pub fn new(limit: u64) -> Result<Self> {
        if limit == 0 || limit > MAX_WORK {
            return Err("invalid bounded joint operation limit".into());
        }
        Ok(Self { used: 0, limit })
    }
    pub fn charge(&mut self, amount: u128) -> Result<()> {
        if amount + self.used as u128 > self.limit as u128 {
            return Err(
                format!("joint work limit reached at{} of{}", self.used, self.limit).into(),
            );
        }
        self.used += amount as u64;
        Ok(())
    }
}

pub fn ce(logits: &[f64], label: usize) -> f64 {
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    logits.iter().map(|z| (z - top).exp()).sum::<f64>().ln() + (top - logits[label])
}
