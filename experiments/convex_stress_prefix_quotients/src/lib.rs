//! Task344: CM58 observed-prefix quotients for discrete input-program discovery.
//! No target generator, selected formula, task identity or heldout data is an
//! input to search. CM57's joint convex CE head is reused after this stage.
pub use convex_stress_synthetic_programs as prior;
pub use prior::{
    Result, observations::Observation, prefix::Formula, prefix::Term, program::Program,
};
pub mod bank;
pub mod deep;
pub mod grammar;
pub mod latent;
#[cfg(test)]
mod pair_tests;
pub mod pairs;
pub mod pool;
pub mod root;
pub mod search;
#[cfg(test)]
mod tests;
pub mod transitions;
