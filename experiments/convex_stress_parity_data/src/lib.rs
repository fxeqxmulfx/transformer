//! Data-only port of the original parity scratchpads, from synthetic/bits.py.
//! Answer rules belong to data construction and scoring, never model features.
pub use convex_stress_generative_data as generative;
pub use generative::{Random, Result, data, hash, scoring, sequences};
pub mod dataset;
mod generator;
pub mod inventory;
#[cfg(test)]
mod tests;
mod wire;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const CASES: [&str; 4] = [
    "suite-parity-none",
    "suite-parity-running",
    "suite-parity-running-hints",
    "suite-parity-ones-hints",
];
