//! RD1: actual partial derivative of CE with normalized features held fixed.
use crate::algebra::linear;

pub fn loss_gradient(
    features: &[f64],
    labels: &[usize],
    weights: &[f64],
    width: usize,
    classes: usize,
) -> (f64, Vec<f64>) {
    let n = labels.len();
    assert!(n > 0 && features.len() == n * width && weights.len() == classes * width);
    let logits = linear(features, n, width, weights, classes);
    let mut errors = vec![0.; classes * n];
    let mut loss = 0.;
    for (i, row) in logits.chunks_exact(classes).enumerate() {
        assert!(labels[i] < classes);
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row.iter().map(|v| (v - max).exp()).sum();
        loss += max - row[labels[i]] + sum.ln();
        for c in 0..classes {
            errors[c * n + i] = ((row[c] - max).exp() / sum - f64::from(c == labels[i])) / n as f64;
        }
    }
    let transposed: Vec<_> = (0..width)
        .flat_map(|k| (0..n).map(move |i| features[i * width + k]))
        .collect();
    (
        loss / n as f64,
        linear(&errors, classes, n, &transposed, width),
    )
}

pub fn energy(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_ce_gradient_matches_central_differences() {
        let x = [1., 2., -0.5, 3., 2., -1.];
        let y = [0, 2, 1];
        let w = [0.2, -0.4, 0.1, 0.3, -0.2, 0.5];
        let (_, g) = loss_gradient(&x, &y, &w, 2, 3);
        for j in 0..w.len() {
            let mut plus = w;
            let mut minus = w;
            plus[j] += 1e-5;
            minus[j] -= 1e-5;
            let finite = (loss_gradient(&x, &y, &plus, 2, 3).0
                - loss_gradient(&x, &y, &minus, 2, 3).0)
                / 2e-5;
            assert!((finite - g[j]).abs() < 1e-9);
        }
    }

    #[test]
    fn fixed_feature_nullspace_decays_under_actual_gradient_steps() {
        let x = [1., 0., 2., 0.];
        let mut w = [0.2, 3., -0.1, -2.];
        for n in 0..12 {
            assert!((w[1] - 3. * 0.98f64.powi(n)).abs() < 1e-12);
            assert!((w[3] + 2. * 0.98f64.powi(n)).abs() < 1e-12);
            let (_, g) = loss_gradient(&x, &[0, 1], &w, 2, 2);
            assert_eq!((g[1], g[3]), (0., 0.));
            for j in 0..4 {
                w[j] = 0.98 * w[j] - 0.1 * g[j];
            }
        }
    }

    #[test]
    fn actual_adam_first_step_lowers_loss_and_escapes_feature_span() {
        let (loss, g) = loss_gradient(&[1., 2.], &[0], &[0.; 4], 2, 2);
        assert_eq!(&g[..2], &[-0.5, -1.]);
        let eta = 0.1;
        let mut updated = [0.; 4];
        for j in 0..2 {
            let m = (1. - 0.9) * g[j];
            let v = (1. - 0.999) * g[j] * g[j];
            updated[j] = -eta * (m / (1. - 0.9)) / ((v / (1. - 0.999)).sqrt() + 1.);
        }
        // Only the first logit is trainable, matching RD5's binary example.
        assert!(loss_gradient(&[1., 2.], &[0], &updated, 2, 2).0 < loss);
        let residual = (2. * updated[0] - updated[1]).powi(2) / 5.;
        assert!((residual - eta * eta / 180.).abs() < 1e-15 && residual > 0.);
    }
}
