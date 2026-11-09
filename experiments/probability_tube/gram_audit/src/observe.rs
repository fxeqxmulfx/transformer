//! Read-only real-CE and execution checks; used separately from future files.
use crate::{Result, batch, evidence::require, gpu, sources, state};
use serde_json::{Value, json};
pub const ENERGY: f64 = 128. * (1. + 5e-6);
pub fn readout(s: &crate::application::State) -> state::State {
    let cv = |x: &[f32]| x[..97 * 128].iter().map(|x| *x as f64).collect();
    state::State {
        p: cv(&s.parameters),
        m: cv(&s.first),
        v: cv(&s.second),
    }
}
pub fn probabilities(logits: &[f64], active: &[bool]) -> Vec<f64> {
    logits
        .as_chunks::<97>()
        .0
        .iter()
        .map(|row| {
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let den = row.iter().map(|x| (x - max).exp()).sum::<f64>();
            row.iter()
                .enumerate()
                .filter(|(c, _)| !active[*c])
                .map(|(_, x)| (x - max).exp() / den)
                .fold(0., f64::max)
        })
        .collect()
}
pub fn execution(p: &[f64], actual: &gpu::Snapshot, rows: &[u32]) -> Result<Value> {
    let ideal = batch::forward(p, rows);
    let features = ideal
        .features
        .as_chunks::<128>()
        .0
        .iter()
        .zip(actual.features.as_chunks::<128>().0)
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0., f64::max);
    let dots = crate::algebra::linear(
        &actual.features,
        actual.features.len() / 128,
        128,
        &p[..97 * 128],
        97,
    );
    let readout = dots
        .iter()
        .zip(&actual.logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let energy = actual
        .features
        .as_chunks::<128>()
        .0
        .iter()
        .map(|x| x.iter().map(|v| v * v).sum::<f64>())
        .fold(0., f64::max);
    require(
        features <= 1e-3 && readout <= 1e-3 && energy <= ENERGY,
        "execution allowance",
    )?;
    Ok(json!({"feature_error":features,"readout_error":readout,"maximum_feature_energy":energy}))
}
pub fn converted(forecast: &Value) -> Result<(Vec<f64>, Vec<f64>)> {
    let q = sources::array(&forecast["probability_ceilings"])?;
    let s = sources::array(&forecast["feature_radii"])?;
    require(
        q.len() == 4704
            && s.len() == 4704
            && q.iter().all(|q| q.is_finite() && *q > 0. && *q <= 1.)
            && s.iter().all(|s| s.is_finite() && *s >= 0.),
        "source forecast range",
    )?;
    let zero = forecast["horizon"] == 0;
    Ok((
        q.into_iter()
            .map(|q| probability_tube::ceiling(q, 2e-3))
            .collect(),
        s.into_iter()
            .map(|s| if zero { 0. } else { (s + 2e-3).next_up() })
            .collect(),
    ))
}
