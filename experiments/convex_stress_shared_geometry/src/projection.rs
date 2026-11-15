//! Euclidean projection onto sum_i w_i*abs(theta_i)<=radius. CM55 proves the
//! resulting domain's sufficiency; tests independently check numerical KKT.
use crate::*;
pub fn mass(theta: &[f64], weights: &[f64]) -> f64 {
    theta.iter().zip(weights).map(|(x, w)| w * x.abs()).sum()
}
pub fn project(theta: &mut [f64], weights: &[f64], radius: f64) -> Result<f64> {
    if theta.len() != weights.len()
        || !radius.is_finite()
        || radius < 0.
        || theta.iter().any(|x| !x.is_finite())
        || weights.iter().any(|w| !w.is_finite() || *w < 0.)
    {
        return Err("invalid weighted projection".into());
    }
    if mass(theta, weights) <= radius {
        return Ok(0.);
    }
    let mut active: Vec<_> = theta
        .iter()
        .zip(weights)
        .filter(|(_, w)| **w > 0.)
        .map(|(x, w)| (x.abs() / w, x.abs() * w, w * w))
        .collect();
    active.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut numerator, mut denominator, mut lambda) = (0., 0., 0.);
    for (i, &(_, weighted, square)) in active.iter().enumerate() {
        numerator += weighted;
        denominator += square;
        lambda = (numerator - radius) / denominator;
        if i + 1 == active.len() || lambda >= active[i + 1].0 {
            break;
        }
    }
    for (x, w) in theta.iter_mut().zip(weights) {
        *x = x.signum() * (x.abs() - lambda * w).max(0.);
    }
    if mass(theta, weights) > radius * (1. + 1e-10) + 1e-9 {
        return Err("weighted projection violated budget".into());
    }
    Ok(lambda)
}
