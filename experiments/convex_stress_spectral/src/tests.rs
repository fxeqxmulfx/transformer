use super::*;

#[test]
fn complex_svd_reconstructs_and_clips_known_singular_values() {
    let x = vec![C::new(3., 4.), C::default(), C::default(), C::new(0., -2.)];
    let decomposition = linalg::svd(&x, 2).unwrap();
    assert!((decomposition.values[0] - 5.).abs() < 1e-12);
    assert!((decomposition.values[1] - 2.).abs() < 1e-12);
    let clipped = decomposition.reconstruct(3.);
    assert!((clipped[0] - C::new(1.8, 2.4)).norm() < 1e-12);
    assert!((clipped[3] - C::new(0., -2.)).norm() < 1e-12);
    let x: Vec<_> = (0..25)
        .map(|i| C::new((i as f64).sin(), (1.3 * i as f64).cos()))
        .collect();
    let reconstructed = linalg::svd(&x, 5).unwrap().reconstruct(f64::INFINITY);
    assert!(linalg::frobenius_distance(&x, &reconstructed) < 1e-12);
}

#[test]
fn ordinary_ce_gradients_match_independent_forward_differences() {
    for vocab in [2, 3, 17, 97] {
        let code = codes(vocab);
        for label in [0, vocab / 2, vocab - 1] {
            for z in [C::default(), C::new(0.3, -0.7), C::new(-1.2, 1.4)] {
                let g = gradient(z, label, &code);
                let fd = forward_gradient(z, label, &code, 1e-5);
                assert!((g - fd).norm() < 2e-9, "{vocab} {label} {z} {g} {fd}");
                let logits: Vec<_> = code.iter().map(|p| p.re * z.re + p.im * z.im).collect();
                let expected = logits.iter().map(|l| l.exp()).sum::<f64>().ln() - logits[label];
                assert!((loss(z, label, &code) - expected).abs() < 2e-14);
            }
        }
    }
}

#[test]
fn dykstra_projection_has_feasibility_and_variational_checks() {
    let x = vec![
        C::new(2., -1.),
        C::new(1., 1.),
        C::new(3., 2.),
        C::new(0., 4.),
    ];
    let (p, report) = linalg::project(&x, 2, 1.6, 2048, 1e-12).unwrap();
    assert!(!report.capped);
    assert!(linalg::max_norm(&p) <= 1. + 1e-12);
    assert!(linalg::svd(&p, 2).unwrap().values[0] <= 1.6 + 1e-12);
    let (pp, _) = linalg::project(&p, 2, 1.6, 256, 1e-12).unwrap();
    assert!(linalg::frobenius_distance(&p, &pp) < 1e-11);
    for k in 0..64 {
        let mut y: Vec<_> = (0..4)
            .map(|j| C::new(((j + 7 * k) as f64).sin(), ((3 * j + k) as f64).cos()))
            .collect();
        let divisor = linalg::max_norm(&y)
            .max(linalg::svd(&y, 2).unwrap().values[0] / 1.6)
            .max(1.);
        for z in &mut y {
            *z /= divisor;
        }
        let inner: f64 = x
            .iter()
            .zip(&p)
            .zip(&y)
            .map(|((a, b), c)| ((a - b).conj() * (c - b)).re)
            .sum();
        assert!(inner <= 1e-9, "projection variational inequality {inner}");
    }
    let (_, capped) = linalg::project(&x, 2, 1.6, 1, 1e-12).unwrap();
    assert!(capped.capped);
}

#[test]
fn tight_spectral_bound_repairs_the_xe15_unseen_sign_in_all_three_modes() {
    let observed = [
        Sample { cell: 0, label: 0 },
        Sample { cell: 2, label: 0 },
        Sample { cell: 1, label: 1 },
    ];
    let code = codes(2);
    let mut endpoints = vec![];
    for arm in 0..3 {
        let mut model = Model::new(2, 2_f64.sqrt());
        for t in 0..128 {
            let chosen = if arm == 1 { vec![t % 3] } else { vec![0, 1, 2] };
            step(
                &mut model,
                &observed,
                &chosen,
                (arm == 2).then_some(1e-5),
                1.,
                256,
                1e-10,
            )
            .unwrap();
        }
        model.validate().unwrap();
        assert!(
            model.entries[3].re > 0.5,
            "arm={arm}, missing={}",
            model.entries[3]
        );
        for s in &observed {
            assert!(loss(model.entries[s.cell], s.label, &code) < 0.25);
        }
        endpoints.push(model);
    }
    assert!(linalg::frobenius_distance(&endpoints[0].entries, &endpoints[2].entries) < 1e-7);
    let mut loose = Model::new(2, 2.);
    for _ in 0..128 {
        step(&mut loose, &observed, &[0, 1, 2], None, 1., 256, 1e-10).unwrap();
    }
    assert!(loose.entries[3].norm() < 1e-12);
}

#[test]
fn malformed_or_infeasible_saved_matrices_are_rejected() {
    let mut model = Model::new(2, 1.);
    model.entries[0] = C::new(2., 0.);
    assert!(model.validate().is_err());
    model.entries[0] = C::new(f64::NAN, 0.);
    assert!(model.validate().is_err());
    assert!(linalg::svd(&[], 0).is_err());
    assert!(linalg::svd(&[C::default()], 2).is_err());
    assert!(linalg::project(&[C::default()], 1, -1., 10, 1e-9).is_err());
}
