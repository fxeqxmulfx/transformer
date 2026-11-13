//! Euclidean column-L1 projection, ported from depth/head.rs at1b2ebc27.
pub(super) fn project(coefficients: &mut [f64], vocabulary: usize, radius: f64) {
    if radius == 0. {
        coefficients.fill(0.);
        return;
    }
    let mut scratch = Vec::with_capacity(coefficients.len() / vocabulary);
    for c in 0..vocabulary {
        scratch.clear();
        scratch.extend(
            coefficients
                .iter()
                .skip(c)
                .step_by(vocabulary)
                .map(|x| x.abs()),
        );
        if scratch.iter().sum::<f64>() <= radius {
            continue;
        }
        scratch.sort_unstable_by(|a, b| b.total_cmp(a));
        let (mut sum, mut threshold) = (0., 0.);
        for (j, &x) in scratch.iter().enumerate() {
            sum += x;
            let next = (sum - radius) / (j + 1) as f64;
            if x > next {
                threshold = next;
            }
        }
        for x in coefficients.iter_mut().skip(c).step_by(vocabulary) {
            *x = x.signum() * (x.abs() - threshold).max(0.);
        }
    }
}

/// Independent threshold by bisection, used only by the replay checker.
pub(super) fn reference(coefficients: &mut [f64], vocabulary: usize, radius: f64) {
    for c in 0..vocabulary {
        if coefficients
            .iter()
            .skip(c)
            .step_by(vocabulary)
            .map(|x| x.abs())
            .sum::<f64>()
            <= radius
        {
            continue;
        }
        let mut lower = 0.;
        let mut upper = coefficients
            .iter()
            .skip(c)
            .step_by(vocabulary)
            .map(|x| x.abs())
            .fold(0., f64::max);
        for _ in 0..100 {
            let mid = (lower + upper) / 2.;
            let sum: f64 = coefficients
                .iter()
                .skip(c)
                .step_by(vocabulary)
                .map(|x| (x.abs() - mid).max(0.))
                .sum();
            if sum > radius {
                lower = mid;
            } else {
                upper = mid;
            }
        }
        for x in coefficients.iter_mut().skip(c).step_by(vocabulary) {
            *x = x.signum() * (x.abs() - upper).max(0.);
        }
    }
}

#[test]
fn threshold_projection_matches_bisection_at_active_budgets() {
    for radius in [0., 0.1, 1., 10.] {
        let mut a: Vec<_> = (0..35).map(|k| ((17 * k + 9) % 31) as f64 - 15.).collect();
        let mut b = a.clone();
        project(&mut a, 5, radius);
        reference(&mut b, 5, radius);
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-12));
        for c in 0..5 {
            assert!(a.iter().skip(c).step_by(5).map(|x| x.abs()).sum::<f64>() <= radius + 1e-12);
        }
        let previous = a.clone();
        project(&mut a, 5, radius);
        assert!(a.iter().zip(&previous).all(|(x, y)| (x - y).abs() < 1e-12));
    }
}
