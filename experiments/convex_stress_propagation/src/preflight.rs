use crate::{Config, Preflight, Problem, RUN_MEMORY_BYTES, Result};

/// Validate before allocating domains, traces or output tables. The bound
/// includes three retained certificate-sized objects, both implementations'
/// working domains, answer sets and input copies, plus 8 MiB of overhead.
/// An OS address-space limit remains required for an actual run.
pub fn preflight(p: &Problem, c: Config) -> Result<Preflight> {
    if !(1..=4096).contains(&p.side)
        || !(1..=4096).contains(&p.states)
        || !(1..=65536).contains(&p.vocab)
        || !(1..=1024).contains(&p.decoders.len())
        || !(1..=1_000_000).contains(&p.observations.len())
        || c.steps > 1024
        || c.memory_limit > RUN_MEMORY_BYTES
    {
        return Err("unsupported dimensions, bank, observations, sweeps or memory cap".into());
    }
    if p.decoders
        .iter()
        .any(|d| d.len() != p.states || d.iter().any(|&y| y >= p.vocab))
        || p.observations
            .iter()
            .any(|t| t.row >= p.side || t.column >= p.side || t.label >= p.vocab)
        || !p.observations.contains(&c.anchor)
    {
        return Err("invalid decoder, observation or unobserved anchor".into());
    }
    let total_branches = p
        .decoders
        .iter()
        .map(|d| d.iter().filter(|&&y| y == c.anchor.label).count())
        .sum::<usize>();
    // Every scalar has been bounded above; checked arithmetic is still used
    // so extending those limits cannot turn overflow into a small allocation.
    let [n, q, v, k, b, s, t] = [
        p.side,
        p.states,
        p.vocab,
        p.decoders.len(),
        total_branches,
        c.steps,
        p.observations.len(),
    ]
    .map(|x| x as u64);
    let product = |xs: &[u64]| -> Result<u64> {
        xs.iter()
            .try_fold(1_u64, |a, &x| a.checked_mul(x))
            .ok_or_else(|| "memory bound overflow".into())
    };
    let terms = [
        8 * 1024 * 1024,
        product(&[3, 2, b, n, q, 8])?, // three retained domain tables
        product(&[3, b, n, 128])?,     // vector headers and per-allocation allowance
        product(&[3, b, s, 8])?,       // deletion counts
        product(&[3, b, 256])?,        // branch headers and allocation allowance
        product(&[3, n, n, v, 16])?,   // answer vector capacities
        product(&[3, n, n, 64])?,
        product(&[6, n, q, 8])?, // working domains in both algorithms
        product(&[2, n, n, v])?, // output membership arrays
        product(&[4, k, q, 8])?,
        product(&[4, k, 64])?,
        product(&[4, t, 24])?,
        product(&[6, t, 8])?, // incident-index capacities and independent replay
        product(&[n, 128])?,
    ];
    let memory_bound = terms
        .into_iter()
        .try_fold(0_u64, u64::checked_add)
        .ok_or("memory bound overflow")?;
    Ok(Preflight {
        memory_bound,
        total_branches,
    })
}
