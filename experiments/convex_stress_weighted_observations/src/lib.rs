//! CM60 at819d4547: exact original multiplicities for the complete CM57 head.
//! No sampled observations, approximate feature keys or clean-label oracles.
//! Real-arithmetic equivalence is proved in Lean; native summation differences
//! are checked against the separately retained ungrouped trainer.
pub use convex_stress_coordinate_programs as joint;
pub use joint::{Features, Parameters, Result};
pub mod corpus;
pub mod domain;
pub mod objective;
#[cfg(test)]
mod selected_tests;
#[cfg(test)]
mod tests;
pub mod training;
