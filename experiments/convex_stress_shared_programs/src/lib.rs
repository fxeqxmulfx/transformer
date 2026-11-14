//! CM52 at62b675fc: shared input programs, complete complex copy head and
//! convex ordinary-CE mixtures. Realizability is an explicit extra premise.
pub use convex_stress_sequences as sequences;
pub use sequences::complex::{C, Input, Observation, linalg};
pub use sequences::{Result, data, dataset, hash, metrics};
use serde::{Deserialize, Serialize};
pub mod binding;
pub mod discovery;
pub mod evaluation;
pub mod head;
pub mod model;
pub mod program;
pub mod runner;
#[cfg(test)]
mod tests;
pub mod training;

pub const PROTOCOL: &str = include_str!("../protocol.json");
pub fn argmax(z: &[f64]) -> usize {
    (1..z.len()).fold(0, |best, c| if z[c] > z[best] { c } else { best })
}
pub fn ce(z: &[f64], label: usize) -> f64 {
    let top = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    z.iter().map(|x| (x - top).exp()).sum::<f64>().ln() + (top - z[label])
}
