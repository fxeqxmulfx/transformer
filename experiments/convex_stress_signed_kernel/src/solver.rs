//! Small deterministic SPD systems. Residual checks are numerical checks,
//! not exact certificates of a missing target or of real-arithmetic kernels.
use crate::Result;

pub const MAX_OBSERVATIONS: usize = 256;
pub const RESIDUAL_LIMIT: f64 = 1e-7;

pub fn inverse(matrix: &[f64], n: usize) -> Result<(Vec<f64>, f64)> {
    if n == 0 || n > MAX_OBSERVATIONS || matrix.len() != n * n {
        return Err("kernel system dimension refused".into());
    }
    if matrix.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite kernel system".into());
    }
    let mut l = vec![0.; n * n];
    for i in 0..n {
        for j in 0..=i {
            if (matrix[i * n + j] - matrix[j * n + i]).abs() > 1e-12 {
                return Err("nonsymmetric kernel system".into());
            }
            let mut value = matrix[i * n + j];
            for k in 0..j {
                value -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !value.is_finite() || value <= 1e-14 {
                    return Err("nonpositive or unstable Cholesky pivot".into());
                }
                l[i * n + i] = value.sqrt();
            } else {
                l[i * n + j] = value / l[j * n + j];
            }
        }
    }
    let mut inverse = vec![0.; n * n];
    for column in 0..n {
        let mut x = vec![0.; n];
        for i in 0..n {
            let mut value = f64::from(i == column);
            for j in 0..i {
                value -= l[i * n + j] * x[j];
            }
            x[i] = value / l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut value = x[i];
            for j in i + 1..n {
                value -= l[j * n + i] * x[j];
            }
            x[i] = value / l[i * n + i];
        }
        for i in 0..n {
            inverse[i * n + column] = x[i];
        }
    }
    let mut residual: f64 = 0.;
    for i in 0..n {
        for j in 0..n {
            let value: f64 = (0..n).map(|k| matrix[i * n + k] * inverse[k * n + j]).sum();
            if !value.is_finite() {
                return Err("nonfinite inverse product".into());
            }
            residual = residual.max((value - f64::from(i == j)).abs());
        }
    }
    if residual > RESIDUAL_LIMIT {
        return Err("kernel inverse residual refused".into());
    }
    Ok((inverse, residual))
}

/// Outward accumulation bounds the real absolute mass of the stored floats.
pub fn upper_mass(values: &[f64]) -> Result<f64> {
    let mut total = 0.;
    for &value in values {
        if !value.is_finite() {
            return Err("nonfinite signed weight".into());
        }
        if value != 0. {
            total = (total + value.abs()).next_up();
        }
    }
    if !total.is_finite() {
        return Err("signed absolute mass overflow".into());
    }
    Ok(total)
}

/// Numerical CM51 normalization. Downward quotient magnitudes and the final
/// check are explicit; no target label or uncharged output scale is involved.
pub fn normalize(values: &mut [f64]) -> Result<f64> {
    let mut scale = 1.;
    for _ in 0..4 {
        let mass = upper_mass(values)?;
        if mass <= 1. {
            return Ok(scale);
        }
        let divisor = mass.next_up();
        scale /= divisor;
        for value in values.iter_mut() {
            if *value != 0. {
                *value = (value.abs() / divisor).next_down().max(0.).copysign(*value);
            }
        }
    }
    if upper_mass(values)? > 1. {
        return Err("signed normalization refused".into());
    }
    Ok(scale)
}
