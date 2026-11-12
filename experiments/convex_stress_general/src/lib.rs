//! CM19 after Lean 0807609b and 7d299b04: general fixed-decoder spectral updates
//! constrained by the certified output union of an entire declared bank.
//! Each solver optimizes one fixed decoder's convex domain, not the full bank
//! hull. Discrete decoder selection and numerical convergence are not proved.
pub use convex_stress_propagation as propagation;
pub use convex_stress_schatten::{C, gradient, linalg, loss};
pub use propagation::RUN_MEMORY_BYTES;
use serde::{Deserialize, Serialize};

pub mod endpoint;
pub mod fft;
pub mod model;
pub mod projection;
pub mod solver;
pub use model::{Model, Shape};
pub use solver::{Mode, Solver, State};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Costs {
    pub gradients: u64,
    pub row_projections: u64,
    pub spectral_projections: u64,
    pub decompositions: u64,
}

/// Joint explicit buffer forecast, including the separately checked certificate
/// and its replay, complex FFT buffers, state copies and serialized number space.
pub fn memory_bound(shape: Shape, observations: usize, certificate: u64) -> Result<u64> {
    shape.validate()?;
    if observations > 1_000_000 {
        return Err("too many observations".into());
    }
    let (cells, q, v, m) = (
        shape.cells() as u64,
        shape.states as u64,
        shape.vocab as u64,
        observations as u64,
    );
    let bytes = certificate
        .checked_mul(2)
        .and_then(|x| x.checked_add(64 * 1024 * 1024))
        .and_then(|x| x.checked_add(192 * cells * q))
        .and_then(|x| x.checked_add(64 * cells * v))
        .and_then(|x| x.checked_add(64 * m * q))
        .ok_or("memory bound overflow")?;
    if bytes > RUN_MEMORY_BYTES {
        return Err("general decoder exceeds the 64 GiB cap".into());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests;
