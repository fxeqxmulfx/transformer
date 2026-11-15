//! CM62 streaming joins after CM58/59 observed-prefix constraints. Discrete
//! program discovery remains outside the subsequent convex CE claim.
pub use convex_stress_prefix_quotients as previous;
pub use previous::{
    Formula, Observation, Program, Result, Term, bank, count_bounds, grammar, latent, pool, prior,
    transitions,
};
pub use previous::{
    adapter, artifacts, dataset, discovery, evaluation, from_data, joint, model, observations,
    prefix, program,
};
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub mod cache;
pub mod deep;
pub mod disk;
pub mod fit;
pub mod index;
pub mod runner;
pub mod search;
pub mod stream;
#[cfg(test)]
mod tests;
