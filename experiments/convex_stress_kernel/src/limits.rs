//! CM34 bounds for actual vector-range blocks; no ambient prime^dimension tensor.
//! Ported CM32 work accounting from refined_quotient at285dcdb6, enlarged only
//! for the finite actual range and charged categorical encoding verification.
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
pub fn dimensions(n: usize, vocabulary: usize) -> Result<()> {
    if !(2..=257).contains(&n) || !(2..=1024).contains(&vocabulary) {
        return Err("kernel head dimensions exceed n257/vocabulary1024".into());
    }
    Ok(())
}
pub fn shape(vocabulary: usize, sizes: &[usize]) -> Result<Bounds> {
    if !(2..=1024).contains(&vocabulary)
        || sizes.is_empty()
        || sizes.len() > 257 * 257
        || sizes.iter().any(|&n| !(1..=513).contains(&n))
    {
        return Err("kernel Fourier block exceeds vocabulary/group/state caps".into());
    }
    let states: usize = sizes.iter().sum();
    let phase_entries: usize = sizes.iter().map(|&n| n * n).sum();
    let real_coefficients = 2 * states * vocabulary;
    if states > 131072 || phase_entries > 4_000_000 || real_coefficients > 128_000_000 {
        return Err("kernel head exceeds state/phase/coefficient caps".into());
    }
    // Reserve for the initializer, three endpoints, two replay models, real and
    // complex derivatives, serialized numeric buffers and streamed SVD. Encoding
    // and the bounded original field certificate use the fixed 1 GiB reserve.
    // This is a conservative allocation forecast, not measured RSS or a Lean
    // theorem about the allocator. The process also has a hard 64 GiB limit.
    let buffer_bytes =
        1024 * 1024 * 1024 + 256 * phase_entries as u64 + 512 * real_coefficients as u64;
    let audit_work = sizes
        .iter()
        .map(|&n| 64 * (n as u64).pow(3) + 16 * (n * n * vocabulary) as u64)
        .sum::<u64>()
        + 64 * 257 * 257 * 8;
    if buffer_bytes > 56 * 1024 * 1024 * 1024 || audit_work > WORK_CAP {
        return Err("kernel head exceeds memory or audit-work forecast".into());
    }
    Ok(Bounds {
        states,
        phase_entries,
        real_coefficients,
        buffer_bytes,
        audit_work,
    })
}
