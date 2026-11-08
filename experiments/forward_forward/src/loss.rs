//! LF1/LF3 in review/flat_spectrum.md: directional, pairwise local goodness.
//! Deviation from Hinton 2212.13345 §§2–3.3: labels enter only the local
//! readout; positive/negative score differences replace binary thresholds.
pub fn softplus(x: f64) -> f64 {
    x.max(0.) + (-x.abs()).exp().ln_1p()
}

pub fn sigmoid(x: f64) -> f64 {
    let e = (-x.abs()).exp();
    if x >= 0. { 1. / (1. + e) } else { e / (1. + e) }
}

/// Mean over every competitor, with unbatched logit derivatives.
pub fn pairwise(scores: &[f64], target: usize, margin: f64) -> (f64, Vec<f64>) {
    assert!(scores.len() > 1 && target < scores.len());
    assert!(margin.is_finite() && scores.iter().all(|v| v.is_finite()));
    let k = (scores.len() - 1) as f64;
    let mut value = 0.;
    let mut gradient = vec![0.; scores.len()];
    for (c, &s) in scores.iter().enumerate() {
        if c != target {
            let d = margin + s - scores[target];
            value += softplus(d) / k;
            gradient[c] = sigmoid(d) / k;
            gradient[target] -= gradient[c];
        }
    }
    (value, gradient)
}

/// Analytic local head update. `features` are fixed data for this derivative;
/// callers must not relabel this as the total gradient of a tied encoder.
pub fn local_gradient(
    features: &[f64],
    width: usize,
    weights: &[f64],
    labels: &[usize],
    margin: f64,
) -> (f64, Vec<f64>, Vec<f64>) {
    assert!(width > 0 && !labels.is_empty());
    assert_eq!(features.len(), width * labels.len());
    assert_eq!(weights.len() % width, 0);
    let classes = weights.len() / width;
    let mut dw = vec![0.; weights.len()];
    let mut dh = vec![0.; features.len()];
    let mut total = 0.;
    for (row, &target) in labels.iter().enumerate() {
        let h = &features[row * width..(row + 1) * width];
        let scores: Vec<_> = weights.chunks_exact(width).map(|e| dot(e, h)).collect();
        let (value, q) = pairwise(&scores, target, margin);
        total += value / labels.len() as f64;
        for c in 0..classes {
            for k in 0..width {
                dw[c * width + k] += q[c] * h[k] / labels.len() as f64;
                dh[row * width + k] += q[c] * weights[c * width + k] / labels.len() as f64;
            }
        }
    }
    (total, dw, dh)
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 2e-8 * (1. + a.abs().max(b.abs())),
            "{a} != {b}"
        );
    }
    fn ce(z: &[f64], y: usize) -> (f64, Vec<f64>) {
        let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = z.iter().map(|v| (v - max).exp()).sum();
        let mut q: Vec<_> = z.iter().map(|v| (v - max).exp() / sum).collect();
        q[y] -= 1.;
        (max + sum.ln() - z[y], q)
    }
    #[test]
    fn corrected_distance_preserves_complex_direction_and_class_order() {
        use num_complex::Complex64;
        let h = [1., -2., 3., 4.];
        let a = [2., 1., -1., 0.];
        let b = [4., 3., 2., 5.];
        let good = |e: &[f64]| {
            let distance: f64 = h.iter().zip(e).map(|(x, y)| (x - y).powi(2)).sum();
            (dot(e, e) - distance) / 2.
        };
        close(good(&a) - good(&b), dot(&h, &a) - dot(&h, &b));
        let hermitian: f64 = h
            .as_chunks::<2>()
            .0
            .iter()
            .zip(a.as_chunks::<2>().0.iter())
            .map(|(x, y)| (Complex64::new(x[0], x[1]).conj() * Complex64::new(y[0], y[1])).re)
            .sum();
        close(hermitian, dot(&h, &a));
        assert_ne!(
            (Complex64::new(1., -2.) * Complex64::new(2., 1.)).re,
            dot(&h[..2], &a[..2])
        );
    }
    #[test]
    fn normalized_activity_saturates_without_losing_directional_score() {
        let eps = f32::EPSILON as f64;
        let x = [1., -2., 3., -4.];
        let q = dot(&x, &x);
        let scale = (q / x.len() as f64 + eps).sqrt();
        let h: Vec<_> = x.iter().map(|v| v / scale).collect();
        close(dot(&h, &h), x.len() as f64 * q / (q + x.len() as f64 * eps));
        close(
            dot(&h, &h.iter().map(|v| -v).collect::<Vec<_>>()),
            -dot(&h, &h),
        );
        assert!(dot(&h, &h) > 3.999);
    }
    #[test]
    fn extremes_and_score_shifts_are_stable() {
        for target in 0..3 {
            let (l, q) = pairwise(&[-1000., 0., 1000.], target, 1.);
            assert!(l.is_finite() && q.iter().all(|v| v.is_finite()));
            close(q.iter().sum(), 0.);
        }
        let (l, q) = pairwise(&[-3., 1., 4.], 1, 1.);
        let (shifted, qs) = pairwise(&[97., 101., 104.], 1, 1.);
        close(l, shifted);
        for (a, b) in q.iter().zip(qs) {
            close(*a, b);
        }
    }
    #[test]
    fn binary_zero_margin_equals_ce_but_multiclass_gradient_changes() {
        for y in 0..2 {
            let (a, ga) = pairwise(&[-2., 3.], y, 0.);
            let (b, gb) = ce(&[-2., 3.], y);
            close(a, b);
            for (a, b) in ga.iter().zip(gb) {
                close(*a, b);
            }
        }
        let z = [0., 0., 2f64.ln()];
        let (_, a) = pairwise(&z, 0, 0.);
        let (_, b) = ce(&z, 0);
        assert!((a[1] * b[2] - a[2] * b[1]).abs() > 0.04);
    }
    #[test]
    fn local_head_and_feature_derivatives_match_independent_differences() {
        let h = [1., -2., 0.4, 0.7];
        let w = [0.1, 0.3, -0.2, 0.6, 0.5, -0.4];
        let (_, dw, dh) = local_gradient(&h, 2, &w, &[0, 2], 1.);
        let step = 1e-5;
        for (k, &gradient) in dw.iter().enumerate() {
            let (mut plus, mut minus) = (w, w);
            plus[k] += step;
            minus[k] -= step;
            let delta = local_gradient(&h, 2, &plus, &[0, 2], 1.).0
                - local_gradient(&h, 2, &minus, &[0, 2], 1.).0;
            close(gradient, delta / (2. * step));
        }
        for (k, &gradient) in dh.iter().enumerate() {
            let (mut plus, mut minus) = (h, h);
            plus[k] += step;
            minus[k] -= step;
            let delta = local_gradient(&plus, 2, &w, &[0, 2], 1.).0
                - local_gradient(&minus, 2, &w, &[0, 2], 1.).0;
            close(gradient, delta / (2. * step));
        }
    }
    #[test]
    fn local_two_block_derivative_is_not_end_to_end_credit() {
        let first = |a: f64| softplus(-a);
        let terminal = |a: f64| softplus(-2. * a);
        let step = 1e-5;
        close((first(step) - first(-step)) / (2. * step), -0.5);
        close((terminal(step) - terminal(-step)) / (2. * step), -1.);
        // Holding the first feature fixed gives a different second-block derivative.
        let fixed = 0.3;
        let second = |b: f64| softplus(-b * fixed);
        close(
            (second(2. + step) - second(2. - step)) / (2. * step),
            -fixed * sigmoid(-2. * fixed),
        );
    }
}
