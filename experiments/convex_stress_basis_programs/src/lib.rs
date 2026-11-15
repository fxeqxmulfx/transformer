//! Full Basis pools with CM57 joint heads, CM60 multiplicities and CM61
//! selected anchors. Program discovery is a discrete, training-only stage.
pub use convex_stress_basis_data as basis;
pub use convex_stress_synthetic_programs as previous;
pub use convex_stress_weighted_observations as weighted;
pub use joint::{Features, Parameters};
pub use previous::{Result, artifacts, dataset, joint, program::Atom};
pub mod audit;
pub mod discovery;
pub mod evaluation;
pub mod fit;
pub mod input;
pub mod model;
pub mod program;
pub mod runner;
pub const PROTOCOL: &str = include_str!("../protocol.json");

pub fn from_data<T>(r: basis::Result<T>) -> Result<T> {
    r.map_err(|e| e.to_string().into())
}

#[cfg(test)]
mod scoring_tests;
#[cfg(test)]
mod tests;
