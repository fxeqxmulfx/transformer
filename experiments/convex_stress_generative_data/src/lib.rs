//! Data-only port of frozen synthetic Rewrite generators and generation scoring.
//! Source: task318's sequences.py, base.py, records.py and generation.py.
//! Dataset answer rules never form a learned model or a model-selection input.
pub use convex_stress_typed_data as typed;
pub use typed::{Random, Result, data, hash, sequences};
pub mod dataset;
mod generator;
pub mod inventory;
pub mod scoring;
#[cfg(test)]
mod tests;
mod wire;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const CASES: [&str; 6] = [
    "suite-copy-repeat",
    "suite-copy-unique",
    "suite-reverse-repeat",
    "suite-reverse-unique",
    "suite-sort-repeat",
    "suite-sort-unique",
];
