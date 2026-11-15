use super::*;

#[test]
fn individual_projections_match_independent_kkt_and_cone_solutions() -> Result<()> {
    let x = projection::weighted_l1(&[3., -2., 0.1], &[2., 1., 4.], 2.)?;
    for (a, b) in x.iter().zip([0.6, -0.8, 0.]) {
        assert!((a - b).abs() < 1e-12);
    }
    let lambda = (3. - x[0]) / 2.;
    assert!(((-2. - x[1]) / -1. - lambda).abs() < 1e-12);
    assert!(0.1 / 4. <= lambda);
    let cone = projection::perspective([3., 0., 0., 0., 0., -1.], 6.)?;
    for (a, b) in cone.iter().zip([1., 0., 0., 0., 0., 1.]) {
        assert!((a - b).abs() < 1e-12);
    }
    let zero = projection::perspective([0.5, 0., 0., 0., 0., -1.], 6.)?;
    assert!(zero.iter().all(|x| x.abs() < 1e-12));
    // Exhaustive independent two-dimensional feasible grid, not an invocation
    // of the projection algorithm under test.
    let optimum: f64 = x[..2]
        .iter()
        .zip([3., -2.])
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    for i in -50..=50 {
        for j in -100..=100 {
            let (a, b) = (i as f64 / 50., j as f64 / 50.);
            if 2. * a.abs() + b.abs() <= 2. + 1e-12 {
                assert!(optimum <= (a - 3.).powi(2) + (b + 2.).powi(2) + 1e-10);
            }
        }
    }
    Ok(())
}

#[test]
fn joint_projection_satisfies_all_anchors_and_independent_variational_checks() -> Result<()> {
    let d = domain()?;
    let input = vec![
        [3., -2., 1., -1., 4., -2.],
        [-1., 2., -3., 1., -2., 4.],
        [2., 1., -1., 3., 2., 5.],
    ];
    let result = d.project(&input, 20000, 1e-10)?;
    assert!(result.normalized_violation <= 1e-10);
    assert!(result.normalized_step <= 1e-10);
    assert!(result.theta[2].iter().all(|x| x.abs() < 1e-8));
    let again = d.project(&result.theta, 20000, 1e-10)?;
    assert!(
        again
            .theta
            .iter()
            .flatten()
            .zip(result.theta.iter().flatten())
            .all(|(a, b)| (a - b).abs() < 1e-8)
    );
    let beta0 = [0.5, 0.25, 0., 1., 1., 1.];
    let beta1 = [0.25, 0.5, 0., 1., 1., 1.];
    let mut tested = 0;
    // Analytically described feasible rays give independent comparison points.
    for a in 0..=39 {
        for b in 0..=39 {
            let (a, b) = (a as f64 / 4., b as f64 / 4.);
            if a + b < 1. || 13. * (a + b) > 128. {
                continue;
            }
            let y = [beta0.map(|v| v * a), beta1.map(|v| v * b), [0.; 6]];
            let distance: f64 = input
                .iter()
                .flatten()
                .zip(y.iter().flatten())
                .map(|(u, v)| (u - v).powi(2))
                .sum();
            assert!(result.squared_distance <= distance + 1e-6);
            let normal: f64 = input
                .iter()
                .flatten()
                .zip(result.theta.iter().flatten())
                .zip(y.iter().flatten())
                .map(|((u, x), y)| (u - x) * (y - x))
                .sum();
            assert!(normal < 1e-6, "projection variational residual{normal}");
            tested += 1;
        }
    }
    assert!(tested > 500);
    assert!(d.span_residual(&features(2))?.iter().all(|x| *x < 1e-12));
    Ok(())
}
