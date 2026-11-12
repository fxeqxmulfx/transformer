//! Bounded anchor discovery for CM19, after Lean aef9d966 and solver b60aba76.
//! Every selected certificate is one complete, independently checked bank union.
//! No constraint from another anchor is silently added to its feasible domain.
pub use convex_stress_general::{self as general, RUN_MEMORY_BYTES, Result, propagation};
use serde::{Deserialize, Serialize};

pub mod cohort;
pub mod search;

/// A declared vocabulary-only family of four coarsenings of the cyclic fold.
/// This lookup table is a structural prior on latent states, not an input-token
/// arithmetic operation. It is identical for every task with this vocabulary.
pub fn decoder_bank(vocab: usize) -> Result<Vec<Vec<usize>>> {
    if !(2..=193).contains(&vocab) {
        return Err("unsupported bank vocabulary".into());
    }
    let q = 2 * vocab - 1;
    Ok((1..=4)
        .map(|width| (0..q).map(|z| z.min(q - z) / width).collect())
        .collect())
}

#[cfg(test)]
mod tests;
