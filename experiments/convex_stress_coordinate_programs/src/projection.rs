//! Euclidean projections used in bounded Dykstra iterations. CM57 proves the
//! domains convex; these floating-point algorithms require numerical checks.
use crate::*;

/// Projection onto a weighted absolute-sum ball, by a scalar KKT multiplier.
pub fn weighted_l1(values: &[f64], weights: &[f64], radius: f64) -> Result<Vec<f64>> {
    if values.is_empty()
        || values.len() != weights.len()
        || values.len() > 6 * MAX_PROGRAMS
        || values.iter().any(|x| !x.is_finite() || x.abs() > 1e15)
        || weights
            .iter()
            .any(|x| !x.is_finite() || *x <= 0. || *x > 1e6)
        || !radius.is_finite()
        || radius < 0.
        || radius > 1e12
    {
        return Err("invalid weighted ball projection".into());
    }
    let mass = values
        .iter()
        .zip(weights)
        .map(|(x, w)| x.abs() * w)
        .sum::<f64>();
    if mass <= radius {
        return Ok(values.to_vec());
    }
    let mut lo = 0.;
    let mut hi = values
        .iter()
        .zip(weights)
        .map(|(x, w)| x.abs() / w)
        .fold(0., f64::max);
    for _ in 0..96 {
        let t = (lo + hi) / 2.;
        let mass = values
            .iter()
            .zip(weights)
            .map(|(x, w)| (x.abs() - t * w).max(0.) * w)
            .sum::<f64>();
        if mass > radius {
            lo = t;
        } else {
            hi = t;
        }
    }
    Ok(values
        .iter()
        .zip(weights)
        .map(|(x, w)| x.signum() * (x.abs() - hi * w).max(0.))
        .collect())
}

/// Local perspective cone: 4*sum(abs(theta[0..5])) <= (B-2)*gamma.
/// Its projection includes gamma>=0, including the polar-cone zero case.
pub fn perspective(values: Parameters, budget: f64) -> Result<Parameters> {
    if values.iter().any(|x| !x.is_finite() || x.abs() > 1e15)
        || !budget.is_finite()
        || !(6. ..=1e6).contains(&budget)
    {
        return Err("invalid perspective projection".into());
    }
    let slope = budget - 2.;
    let mass = 4. * values[..5].iter().map(|x| x.abs()).sum::<f64>();
    if values[5] >= 0. && mass <= slope * values[5] {
        return Ok(values);
    }
    let mut lo = 0.;
    let mut hi = values[..5]
        .iter()
        .map(|x| x.abs() / 4.)
        .fold(0., f64::max)
        .max((-values[5] / slope).max(0.))
        + 1.;
    for _ in 0..96 {
        let t = (lo + hi) / 2.;
        let left = 4.
            * values[..5]
                .iter()
                .map(|x| (x.abs() - 4. * t).max(0.))
                .sum::<f64>();
        if left > slope * (values[5] + slope * t) {
            lo = t;
        } else {
            hi = t;
        }
    }
    let mut out = values;
    for j in 0..5 {
        out[j] = values[j].signum() * (values[j].abs() - 4. * hi).max(0.);
    }
    out[5] = (values[5] + slope * hi).max(0.);
    Ok(out)
}

pub fn dot(a: &Parameters, b: &Parameters) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Two-pass modified Gram-Schmidt on observed constraint rows. Stored original
/// rows are checked separately, so a numerical rank decision cannot hide error.
pub fn basis(rows: &[Parameters]) -> Result<Vec<Parameters>> {
    let mut basis: Vec<Parameters> = vec![];
    for row in rows {
        let norm = dot(row, row).sqrt();
        if !norm.is_finite() {
            return Err("nonfinite observed anchor".into());
        }
        if norm == 0. {
            continue;
        }
        let mut v = row.map(|x| x / norm);
        for _ in 0..2 {
            for q in &basis {
                let c = dot(q, &v);
                for j in 0..6 {
                    v[j] -= c * q[j];
                }
            }
        }
        let residual = dot(&v, &v).sqrt();
        if residual > 1e-11 {
            if basis.len() == 6 {
                return Err("observed rank exceeds feature dimension".into());
            }
            basis.push(v.map(|x| x / residual));
        }
    }
    Ok(basis)
}

pub fn linear(values: &mut Parameters, basis: &[Parameters]) {
    for _ in 0..2 {
        for q in basis {
            let c = dot(q, values);
            for j in 0..6 {
                values[j] -= c * q[j];
            }
        }
    }
}
