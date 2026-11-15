//! Data-only port of frequency tasks from frozen synthetic/sequences.py.
//! Answer rules belong to data construction and scoring, never model features.
pub use convex_stress_generative_data as generative;
pub use generative::{Random, Result, data, hash, sequences};
pub mod dataset;
mod generator;
pub mod inventory;
pub mod scoring;
#[cfg(test)]
mod tests;
mod wire;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const CASES: [&str; 7] = [
    "suite-histogram-bos",
    "suite-histogram-no-bos",
    "suite-histogram2",
    "suite-mode-none",
    "suite-mode-counts",
    "suite-mode-itemized",
    "suite-most_freq",
];
