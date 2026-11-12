//! CM18 after Lean 5c0fe9bc: a training-only certificate for decoded answers.
//!
//! `Certificate` and `check` implement `DecodedRecovery.Accepts` and `Pinned`.
//! An accepted certificate identifies covered outputs IF the true table has
//! phases in the declared fixed decoder family (`accepted_decoding`). This
//! structural assumption cannot be checked from training observations alone.
//! `discover` is a bounded native proposal, not an executable extraction of the
//! classical Lean `recover`, and a search failure does not prove inconsistency.
pub use convex_stress_paired::{Decoder, Mode, Model, Result, Sample, Shape};
use serde::{Deserialize, Serialize};

pub mod certificate;
pub mod cohort;
pub mod discovery;
pub mod endpoint;
pub mod solver;
pub mod training;

pub use certificate::{Certificate, Verified, check};
pub use discovery::{Discovery, Limits, SearchCosts, SearchStop, discover};
pub use training::Training;

/// User override, 2026-11-12: new runs have a 64 GiB address-space cap.
pub const RUN_MEMORY_BYTES: u64 = 64 * 1024 * 1024 * 1024;

/// The parent solver's conservative allowance, plus eight latent halves, four
/// decoded tables, certificate/search buffers, the maximum proposal trace with
/// geometric vector-growth allowance, and 64 MiB of additional overhead.
/// The size validation precedes every dense allocation in this crate.
pub fn memory_bound(shape: Shape) -> Result<u64> {
    // Check raw dimensions before the parent's products: serialized sizes are
    // untrusted and must not overflow even in a debug build.
    if !(2..=193).contains(&shape.side)
        || !(2..=97).contains(&shape.orbits)
        || !(2..=193).contains(&shape.vocab)
    {
        return Err("unsupported certified model dimensions".into());
    }
    let base = convex_stress_paired::memory_bound(shape)?;
    let cells = shape.cells() as u64;
    Ok(base
        + 8 * 8 * cells * shape.orbits as u64
        + 4 * 8 * cells * shape.vocab as u64
        + 128 * cells
        + 2 * 4_000_000 * std::mem::size_of::<discovery::Attempt>() as u64
        + 64 * 1024 * 1024)
}

pub fn preflight(shape: Shape, limit: u64) -> Result<u64> {
    let bytes = memory_bound(shape)?;
    if bytes > limit {
        return Err("certified model exceeds the requested memory limit".into());
    }
    Ok(bytes)
}

/// A categorical observation. Cells use `row + side * column`, as in CM16.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    pub cell: usize,
    pub label: usize,
}

#[cfg(test)]
mod solver_tests;
#[cfg(test)]
mod tests;
