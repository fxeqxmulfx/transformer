//! Continue CM62's unchanged exact joins past the first observed fit. This
//! changes search scheduling and retained examples, not the logical tests or
//! cached-refusal rule. Smaller syntax is an inductive prior, not an unseen
//! target-identification theorem. No heldout model score is produced here.
pub use convex_stress_streaming_counts as original;
pub use original::{
    Formula, Observation, Program, Result, Term, bank, cache, count_bounds, grammar, index, latent,
    pool, prefix, previous, prior, stream, transitions,
};
pub mod deep;
pub mod retained;
pub mod search;
#[cfg(test)]
mod tests;
