//! Native adapters for the first additional cohort in reviewed task318.
//! Generator answer rules are confined to data modules, never model features.
pub use baseline::{Result, data, hash, metrics, random::Random};
pub use convex_stress_calibrated as baseline;
pub mod dataset;
mod prefix;
mod retrieval;
mod wire;
pub const CATALOG: &str = include_str!("../../convex_stress_catalog/protocol.json");
#[cfg(test)]
mod tests;
