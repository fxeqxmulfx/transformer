//! CM25: categorical tags and repeated phases inside a convex spectral hull.
//! Native proposals are heuristic; raw audits remain numerical evidence.
pub use convex_stress_decode::{C, Result, Sample, linalg, loss, metrics, random};
pub mod model;
pub mod refinement;
pub use model::Model;
#[cfg(test)]
mod tests;

pub fn validate_observations(n: usize, vocab: usize, samples: &[Sample]) -> Result<()> {
    model::memory_bound(n, 2, vocab)?;
    if samples.is_empty() || samples.len() > n * n {
        return Err("invalid observed-cell count".into());
    }
    let mut seen = vec![false; n * n];
    for s in samples {
        if s.cell >= n * n || s.label >= vocab || seen[s.cell] {
            return Err("invalid or repeated categorical observation".into());
        }
        seen[s.cell] = true;
    }
    Ok(())
}
