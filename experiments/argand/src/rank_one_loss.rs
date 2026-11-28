//! Half the mean squared distance of a real matrix to rank at most one.
//! At a tied leading singular value the returned residual is the gradient
//! of the quadratic with the selected best approximant held fixed.
use crate::{Result, svd};

pub(crate) fn value_gradient(x: &[f64], rows: usize, cols: usize) -> Result<(f64, Vec<f64>)> {
    if rows.checked_mul(cols) != Some(x.len()) || x.is_empty() || x.iter().any(|v| !v.is_finite()) {
        return Err("expected a nonempty finite rectangular matrix".into());
    }
    let s = svd::thin(x, rows, cols);
    let rank = s.values.len();
    let count = x.len() as f64;
    let mut loss = 0.0;
    let mut gradient = Vec::with_capacity(x.len());
    for (p, entry) in x.iter().enumerate() {
        let leading = s.values[0] * s.left[(p / cols) * rank] * s.right[p % cols];
        let residual = entry - leading;
        loss += residual * residual;
        gradient.push(residual / count);
    }
    Ok((loss / (2.0 * count), gradient))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangular_gradient_matches_differences_and_vanishes_on_unit_factors() -> Result<()> {
        let x = [2.0, -1.0, 3.0, -1.0, 4.0, 1.0];
        for (rows, cols) in [(2, 3), (3, 2)] {
            let (loss, grad) = value_gradient(&x, rows, cols)?;
            assert!(loss > 0.0);
            for i in 0..x.len() {
                let (mut hi, mut lo) = (x, x);
                hi[i] += 1e-5;
                lo[i] -= 1e-5;
                let numeric = (value_gradient(&hi, rows, cols)?.0
                    - value_gradient(&lo, rows, cols)?.0)
                    / 2e-5;
                assert!((numeric - grad[i]).abs() < 1e-8);
            }
        }
        let (loss, grad) = value_gradient(&[1., -1., -1., -1., 1., 1.], 2, 3)?;
        assert!(loss < 1e-28 && grad.iter().all(|v| v.abs() < 1e-14));
        Ok(())
    }
}
