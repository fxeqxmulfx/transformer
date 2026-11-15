//! CM62 streaming joins after CM58/59 observed-prefix constraints. Discrete
//! program discovery remains outside the subsequent convex CE claim.
pub use convex_stress_prefix_quotients as previous;
pub use previous::{
    Formula, Observation, Program, Result, Term, bank, count_bounds, grammar, latent, pool, prior,
    transitions,
};
pub mod cache;
pub mod deep;
pub mod index;
pub mod search;
pub mod stream;
#[cfg(test)]
mod tests;
