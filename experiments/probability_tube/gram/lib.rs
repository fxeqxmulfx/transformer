//! QG1-QG3 in review/flat_spectrum.md and GradientEvidence/CEGram/*.lean.
//! Pure checkpoint inputs: per-query ceilings, executed features, fixed metric.
//! BLAS covariance entries have an explicit FP64 error allowance. Positive
//! reductions expand outward. This is not a verified BLAS/CUDA implementation.
mod covariance;
pub use covariance::{Covariance, single_thread};

pub type Result<T> = std::result::Result<T, String>;
pub(crate) fn require(ok: bool, why: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(why.to_owned()) }
}
pub(crate) fn add(a: f64, b: f64) -> f64 {
    (a + b).next_up()
}
pub(crate) fn mul(a: f64, b: f64) -> f64 {
    (a * b).next_up()
}
pub(crate) fn sqrt(a: f64) -> f64 {
    a.sqrt().next_up()
}
pub(crate) fn div(a: f64, b: f64) -> f64 {
    (a / b).next_up()
}

#[derive(Clone, Copy, Debug)]
pub struct OldBound {
    pub gram_energy_upper: f64,
    pub gram_coefficient: f64,
    pub trace_coefficient: f64,
    pub selected: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct Forecast {
    pub uniform: f64,
    pub row_trace: f64,
    pub transported_gram: f64,
    pub selected: f64,
    pub weighted_motion: f64,
}

/// QG3: bound the future coefficient for one known batch. `old` is the
/// operator-squared/batch-size cap for this same q, metric and old feature table.
/// Radii must cover executed features; M covers every future feature energy.
/// The wider query-set maximum Q is used only by the comparison baseline.
pub fn transport(
    old: f64,
    q: &[f64],
    radii: &[f64],
    rmax: f64,
    energy: f64,
    maximum_q: f64,
) -> Result<Forecast> {
    require(
        !q.is_empty() && q.len() == radii.len() && q.len() <= 1_000_000,
        "batch inventory",
    )?;
    require(
        [old, rmax, energy, maximum_q]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.)
            && old <= 1e100
            && rmax <= 1e12
            && energy <= 1e12
            && maximum_q <= 1.,
        "coefficient range",
    )?;
    require(
        q.iter()
            .all(|v| v.is_finite() && *v > 0. && *v <= maximum_q)
            && radii
                .iter()
                .all(|v| v.is_finite() && *v >= 0. && *v <= 1e12),
        "query ceiling or feature radius",
    )?;
    let n = q.len() as f64;
    let mean = div(q.iter().copied().fold(0., add), n).min(maximum_q);
    let motion = q
        .iter()
        .zip(radii)
        .fold(0., |acc, (&q, &s)| add(acc, mul(q, mul(s, s))));
    let weighted_motion = mul(rmax, div(motion, n));
    let sum = add(sqrt(old), sqrt(weighted_motion));
    let transported_gram = mul(sum, sum);
    let uniform = mul(mul(rmax, energy), maximum_q);
    let row_trace = mul(mul(rmax, energy), mean).min(uniform);
    let selected = transported_gram.min(row_trace);
    require(selected.is_finite(), "nonfinite forecast")?;
    Ok(Forecast {
        uniform,
        row_trace,
        transported_gram,
        selected,
        weighted_motion,
    })
}

/// Inclusive endpoint convention: H=0 is the current batch; H=1 includes
/// the next state's batch. State 10000 is terminal and has no following batch.
pub fn batch_window(
    anchor: usize,
    horizon: usize,
    updates: usize,
) -> Result<std::ops::Range<usize>> {
    require(
        anchor < updates && horizon <= updates - anchor,
        "window range",
    )?;
    Ok(anchor..(anchor + horizon).saturating_add(1).min(updates))
}

/// Read the exact source plan's indices, retaining the shorter epoch batch.
pub fn batch_indices<'a>(order: &'a [u32], boundaries: &[usize], step: usize) -> Result<&'a [u32]> {
    require(
        step < boundaries.len().saturating_sub(1),
        "missing batch boundary",
    )?;
    let (start, end) = (boundaries[step], boundaries[step + 1]);
    require(start < end && end <= order.len(), "invalid batch boundary")?;
    Ok(&order[start..end])
}
