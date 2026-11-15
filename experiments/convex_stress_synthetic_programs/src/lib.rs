//! Task343: fixed input-only operators with the joint CM57 complex head.
//! Data/oracles are confined to the runner. Program evaluation has no task
//! name, target sequence, selected formula seed or learned output decoder.
pub use convex_stress_coordinate_programs as joint;
pub use convex_stress_remaining_synthetic_data as dataset;
pub use joint::{Features, Parameters, Result};
pub mod adapter;
pub mod artifacts;
pub mod discovery;
pub mod evaluation;
pub mod fit;
mod independent;
#[cfg(test)]
mod integration_tests;
pub mod model;
pub mod observations;
pub mod prefix;
pub mod program;
pub mod runner;
pub mod search;
pub const PROTOCOL: &str = include_str!("../protocol.json");

pub fn from_data<T>(result: dataset::Result<T>) -> Result<T> {
    result.map_err(|e| e as Box<dyn std::error::Error>)
}
#[cfg(test)]
mod tests;
