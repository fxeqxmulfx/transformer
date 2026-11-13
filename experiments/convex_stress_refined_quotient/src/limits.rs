//! Prospective dimension and scalar-work caps, before tensor allocation.
use crate::*;
pub const WORK_CAP: u64 = 32_000_000_000;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bounds {
    pub states: usize,
    pub phase_entries: usize,
    pub real_coefficients: usize,
    pub buffer_bytes: u64,
    pub audit_work: u64,
}
pub fn dimensions(n: usize, q: usize, vocabulary: usize) -> Result<()> {
    if !(2..=257).contains(&n) || !(2..=257).contains(&q) || !(2..=1024).contains(&vocabulary) {
        return Err("refined dimensions exceed n257/q257/vocabulary1024".into());
    }
    Ok(())
}
pub fn shape(vocabulary: usize, sizes: &[usize]) -> Result<Bounds> {
    if !(2..=1024).contains(&vocabulary)
        || !(2..=257).contains(&sizes.len())
        || sizes.iter().any(|&n| !(1..=513).contains(&n))
    {
        return Err("refined Fourier block exceeds vocabulary/group/state caps".into());
    }
    let states: usize = sizes.iter().sum();
    let phase_entries: usize = sizes.iter().map(|&n| n * n).sum();
    let real_coefficients = 2 * states * vocabulary;
    if states > 65536 || phase_entries > 4_000_000 || real_coefficients > 16_000_000 {
        return Err("refined model exceeds state/phase/coefficient caps".into());
    }
    // Includes three endpoints, independent replay, gradients, JSON buffers
    // and streamed block SVD workspaces. No global N*N matrix is allocated.
    let buffer_bytes =
        512 * 1024 * 1024 + 128 * phase_entries as u64 + 128 * real_coefficients as u64;
    let audit_work = sizes
        .iter()
        .map(|&n| 64 * (n as u64).pow(3) + 16 * (n * n * vocabulary) as u64)
        .sum::<u64>();
    if buffer_bytes > 56 * 1024 * 1024 * 1024 || audit_work > WORK_CAP {
        return Err("refined model exceeds memory or audit-work forecast".into());
    }
    Ok(Bounds {
        states,
        phase_entries,
        real_coefficients,
        buffer_bytes,
        audit_work,
    })
}
