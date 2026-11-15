//! CM57, CoordinatePrograms at 258dd055: fixed input-only feature families,
//! joint program masses and scaled coordinate decoders, complete complex heads
//! and ordinary CE. Numerical tolerances are checks, not Lean refinements.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub type Features = [f64; 5];
pub type Parameters = [f64; 6];
pub const COSTS: Parameters = [4., 4., 4., 4., 4., 2.];
pub const MAX_PROGRAMS: usize = 256;
pub const MAX_RECORDS: usize = 200_000;
pub const MAX_BYTES: u128 = 56 * 1024 * 1024 * 1024;

pub mod constraints;
pub mod data;
pub mod head;
pub mod projection;
pub mod training;

pub fn charge(theta: &[Parameters]) -> f64 {
    theta
        .iter()
        .flat_map(|t| t.iter().zip(COSTS))
        .map(|(x, w)| w * x.abs())
        .sum()
}

pub fn argmax(z: &[f64]) -> usize {
    assert!(!z.is_empty() && z.iter().all(|x| x.is_finite()));
    let mut best = 0;
    for c in 1..z.len() {
        if z[c] > z[best] {
            best = c;
        }
    }
    best
}

pub fn ce(z: &[f64], label: usize) -> f64 {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    m + z.iter().map(|x| (x - m).exp()).sum::<f64>().ln() - z[label]
}

#[cfg(test)]
mod tests;
