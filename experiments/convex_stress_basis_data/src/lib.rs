//! Original Basis data protocols from the frozen165-case catalog. Oracles
//! and clean noise references belong only to data validation, never prediction.
pub use convex_stress_remaining_synthetic_data as previous;
pub use previous::{Random, Result, data, generative, hash, sequences};
pub mod dataset;
pub mod generator;
pub mod independent;
pub mod inventory;
pub mod noise;
pub mod wire;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const CASES: [&str; 8] = [
    "basis-recall-hard",
    "language-l1",
    "language-l2",
    "language-l3",
    "language-l4",
    "language-l5",
    "language-l6",
    "basis-parity-noise20",
];
#[cfg(test)]
mod tests;
