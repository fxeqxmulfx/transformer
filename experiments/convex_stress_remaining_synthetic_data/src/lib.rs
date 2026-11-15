//! Data-only port of the remaining synthetic catalog cases. Target algorithms
//! and random labels stay in data construction, outside all learned predictors.
pub use convex_stress_generative_data as generative;
pub use generative::{Random, Result, data, hash, sequences};
pub mod addition;
pub mod crasp;
pub mod dataset;
mod generator;
pub mod inventory;
pub mod scoring;
pub mod spec;
#[cfg(test)]
mod tests;
mod wire;
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const CASES: [&str; 13] = [
    "suite-count",
    "suite-addition-forward-plain-standard",
    "suite-addition-forward-plain-balanced",
    "suite-addition-forward-hints-standard",
    "suite-addition-forward-hints-balanced",
    "suite-addition-reverse-plain-standard",
    "suite-addition-reverse-plain-balanced",
    "suite-addition-reverse-hints-standard",
    "suite-addition-reverse-hints-balanced",
    "suite-crasp-depth-1",
    "suite-crasp-depth-2",
    "suite-crasp-depth-3",
    "random-lm-control",
];
