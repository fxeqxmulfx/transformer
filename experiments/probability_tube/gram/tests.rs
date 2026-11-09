use probability_gram::{Covariance, batch_indices, batch_window, transport};

fn largest_eigenvalue_2(a: f64, b: f64, d: f64) -> f64 {
    (a + d + ((a - d).powi(2) + 4. * b * b).sqrt()) / 2.
}
#[test]
fn exact_dyadic_covariance_and_nonuniform_metric() {
    // C = [[9/4, 1/2], [1/2, 3/2]]; weighting r=[1,4] gives [[9/4,1],[1,6]].
    let c = Covariance::new(&[1., 2., 2., 0., 0., 1.], &[0.25, 0.5, 0.5], 2).unwrap();
    let b = c.bound(&[1., 4.]).unwrap();
    let exact_energy = 2.25f64.powi(2) + 2. + 36.;
    let operator = largest_eigenvalue_2(2.25, 1., 6.) / 3.;
    assert!(b.gram_energy_upper >= exact_energy);
    assert!(b.gram_energy_upper < exact_energy * (1. + 1e-12));
    assert!(b.selected >= operator);
    assert!(b.selected < b.trace_coefficient);
    assert!(c.maximum_entry_error > 0. && c.maximum_entry_error < 1e-12);
}
#[test]
fn flat_covariance_improves_trace_at_source_batch_sizes() {
    for rows in [512, 96] {
        let mut features = vec![0.; rows * 128];
        for i in 0..rows {
            features[i * 128 + i % 128] = 2.;
        }
        let q: Vec<_> = (0..rows)
            .map(|i| if i % 2 == 0 { 0.25 } else { 0.5 })
            .collect();
        let c = Covariance::new(&features, &q, 128).unwrap();
        let b = c.bound(&[1.; 128]).unwrap();
        // Half of the nonzero eigenvalues are 1/rank, half are 2/rank.
        let rank = rows.min(128) as f64;
        let gram = (2.5 / rank).sqrt();
        assert!(b.gram_coefficient >= gram && b.gram_coefficient < gram + 1e-10);
        assert!(b.selected < 0.4 * b.trace_coefficient);
        let f = transport(b.selected, &q, &vec![0.; rows], 1., 4., 0.5).unwrap();
        assert!(f.selected >= gram && f.selected < gram + 1e-10);
        assert!(f.row_trace < f.uniform);
    }
}

fn probabilities(features: &[f64], e: &[f64]) -> Vec<[f64; 3]> {
    features
        .as_chunks::<2>()
        .0
        .iter()
        .map(|x| {
            let logits: [f64; 3] = std::array::from_fn(|c| e[2 * c] * x[0] + e[2 * c + 1] * x[1]);
            let max = logits.into_iter().fold(f64::NEG_INFINITY, f64::max);
            let exp = logits.map(|v| (v - max).exp());
            let total: f64 = exp.iter().sum();
            exp.map(|v| v / total)
        })
        .collect()
}
fn loss(features: &[f64], e: &[f64]) -> f64 {
    -probabilities(features, e)
        .iter()
        .map(|p| p[0].ln())
        .sum::<f64>()
        / (features.len() / 2) as f64
}
#[test]
fn actual_ce_derivative_and_moving_feature_operator_are_controlled() {
    let old = [1., 2., -2., 1., 1., -1.];
    let future = [1.125, 2., -2., 0.875, 1.125, -0.875];
    let e = [-0.5, 0.25, 0.2, -0.1, -0.4, 0.3];
    let r = [1., 3.];
    let p = probabilities(&future, &e);
    let q: Vec<_> = p.iter().map(|p| p[1].max(p[2]) + 0.02).collect();
    let gradient: Vec<_> = (0..6)
        .map(|j| {
            (0..3)
                .map(|i| (p[i][j / 2] - f64::from(j / 2 == 0)) * future[2 * i + j % 2])
                .sum::<f64>()
                / 3.
        })
        .collect();
    for j in 0..6 {
        let mut plus = e;
        plus[j] += 1e-6;
        let mut minus = e;
        minus[j] -= 1e-6;
        assert!(((loss(&future, &plus) - loss(&future, &minus)) / 2e-6 - gradient[j]).abs() < 1e-9);
    }
    let radii: Vec<_> = (0..3)
        .map(|i| {
            ((future[2 * i] - old[2 * i]).powi(2) + (future[2 * i + 1] - old[2 * i + 1]).powi(2))
                .sqrt()
        })
        .collect();
    let b = Covariance::new(&old, &q, 2).unwrap().bound(&r).unwrap();
    let f = transport(b.selected, &q, &radii, 3., 6., 1.).unwrap();
    let c: Vec<_> = (0..4)
        .map(|j| {
            (0..3)
                .map(|i| q[i] * future[2 * i + j / 2] * future[2 * i + j % 2])
                .sum::<f64>()
                * r[j / 2].sqrt()
                * r[j % 2].sqrt()
                / 3.
        })
        .collect();
    assert!(largest_eigenvalue_2(c[0], c[1], c[3]) <= f.selected);
    let dg = [gradient[2] - gradient[4], gradient[3] - gradient[5]];
    let radial = (e[2] - e[4]) * dg[0] + (e[3] - e[5]) * dg[1];
    let square = r[0] * dg[0] * dg[0] + r[1] * dg[1] * dg[1];
    assert!(radial > 0. && square > 0. && square <= f.selected * radial);
    assert!(f.weighted_motion > 0.);
}

#[test]
fn known_plan_keeps_tail_batches_and_inclusive_endpoints() {
    let order: Vec<_> = (0..4704).rev().collect();
    let mut boundaries: Vec<_> = (0..10).map(|i| i * 512).collect();
    boundaries.push(4704);
    assert_eq!(batch_indices(&order, &boundaries, 0).unwrap().len(), 512);
    let last = batch_indices(&order, &boundaries, 9).unwrap();
    assert_eq!(last.len(), 96);
    assert_eq!(last.first(), Some(&95));
    assert_eq!(last.last(), Some(&0));
    assert_eq!(batch_window(9500, 0, 10000).unwrap(), 9500..9501);
    assert_eq!(batch_window(9500, 1, 10000).unwrap(), 9500..9502);
    assert_eq!(batch_window(9500, 500, 10000).unwrap(), 9500..10000);
    assert!(batch_window(9500, 501, 10000).is_err());
    assert!(batch_window(10000, 0, 10000).is_err());
    assert!(batch_indices(&order, &boundaries, 10).is_err());
}
#[test]
fn invalid_numerical_inputs_fail_closed() {
    assert!(Covariance::new(&[], &[], 2).is_err());
    assert!(Covariance::new(&[1.], &[0.], 1).is_err());
    assert!(Covariance::new(&[f64::NAN], &[0.5], 1).is_err());
    assert!(Covariance::new(&[1e7], &[0.5], 1).is_err());
    let c = Covariance::new(&[1.], &[0.5], 1).unwrap();
    assert!(c.bound(&[-1.]).is_err());
    assert!(transport(1., &[0.5], &[-1.], 1., 2., 1.).is_err());
    assert!(transport(1., &[0.5], &[1.], 1., 2., 0.4).is_err());
    assert!(batch_indices(&[0], &[0, 2], 0).is_err());
    assert!(batch_indices(&[0], &[0, 1], usize::MAX).is_err());
}
