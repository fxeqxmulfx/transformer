//! CM21 (Lean 1885957b): convex density extraction in supplied Fourier subspaces.
//! Native finite-tolerance optimization is not an exact arithmetic certificate.
//! The solver receives features and observable rows, never a latent ordering.
pub use convex_stress_decode::{C, Model, Result, Sample, random};
pub mod affine;
pub mod artifact;
pub mod assignment_certificate;
pub mod audit;
pub mod comparison;
pub mod interval;
pub mod interval_audit;
pub mod interval_search;
pub mod matrix;
pub mod refinement;
pub mod refinement_audit;
pub mod refinement_search;
pub mod search;
pub mod solver;

pub fn memory_bound(n: usize, rank: usize) -> Result<u64> {
    if !(2..=193).contains(&n) || !(1..=8).contains(&rank) || rank > n {
        return Err("density dimensions exceed n=193, rank=8 limits".into());
    }
    // Includes feature/sketch buffers, the <=194 square affine Gram eigensolve,
    // dense matrix work and a bounded full trace. The process also has a 64 GiB cap.
    Ok(256 * (n as u64 + 1).pow(2) + 64 * 1024 * 1024)
}

#[cfg(test)]
mod interval_search_tests;
#[cfg(test)]
mod interval_tests;
#[cfg(test)]
mod refinement_tests;
#[cfg(test)]
mod search_tests;
#[cfg(test)]
mod tests;
