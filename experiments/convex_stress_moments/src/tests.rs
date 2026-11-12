use super::*;
use projection::{Projector, direct_moments, distance, simplex};
use solver::{Mode, Solver};

#[test]
fn ce_is_ordinary_logsumexp_and_gradient_matches_scalar_differences() {
    let logits = [-0.3, 1.2, 0.7, -1.];
    let expected = logits.iter().map(|x: &f64| x.exp()).sum::<f64>().ln() - logits[2];
    assert!((loss(&logits, 2) - expected).abs() < 1e-14);
    let mut g = [0.; 4];
    gradient(&logits, 2, &mut g);
    for c in 0..4 {
        let (mut a, mut b) = (logits, logits);
        a[c] += 1e-5;
        b[c] -= 1e-5;
        assert!((g[c] - (loss(&a, 2) - loss(&b, 2)) / 2e-5).abs() < 1e-10);
    }
    assert!((g.iter().sum::<f64>()).abs() < 1e-15);
    for n in [2, 5, 97] {
        let mut p = vec![0.; n];
        p[n - 1] = 1.;
        assert!((loss(&p, n - 1) - ideal_loss(n)).abs() < 1e-14);
    }
}

#[test]
fn fft_matches_direct_complex_codes_roundtrips_and_preserves_energy() -> Result<()> {
    for n in [2, 3, 4, 7] {
        let p: Vec<_> = (0..n * n * n)
            .map(|i| ((i * 17 + 3) as f64).sin())
            .collect();
        let mut fft = fft::Fourier::new(n)?;
        let m = fft.forward(&p)?;
        let direct = direct_moments(&p, n);
        assert!(m.iter().zip(direct).all(|(a, b)| (*a - b).norm() < 1e-13));
        assert!(distance(&fft.inverse(&m)?, &p) < 1e-13);
        let energy = p.iter().map(|x| x * x).sum::<f64>() * n as f64;
        assert!((m.iter().map(|x| x.norm_sqr()).sum::<f64>() - energy).abs() < 1e-10);
    }
    assert!(fft::Fourier::new(0).is_err());
    Ok(())
}

#[test]
fn simplex_projection_satisfies_variational_inequality_and_idempotence() {
    for shift in 0..40 {
        let x: Vec<_> = (0..7)
            .map(|i| ((i * 17 + shift) as f64).sin() * 3.)
            .collect();
        let mut p = [0.; 7];
        simplex(&x, &mut p);
        assert!(p.iter().all(|p| *p >= 0.));
        assert!((p.iter().sum::<f64>() - 1.).abs() < 1e-14);
        let mut pp = [0.; 7];
        simplex(&p, &mut pp);
        assert!(distance(&p, &pp) < 1e-14);
        // Linear maximization on the simplex is achieved at a vertex, so these
        // inequalities test every competitor, not just a sampled distribution.
        for c in 0..7 {
            let dot: f64 = (0..7)
                .map(|i| (x[i] - p[i]) * (f64::from(i == c) - p[i]))
                .sum();
            assert!(dot < 1e-13);
        }
    }
}

#[test]
fn complex_ball_projection_and_uniform_repair_enforce_every_frequency() -> Result<()> {
    for n in [3, 4] {
        let mut projector = Projector::new(n, 2)?;
        let x: Vec<_> = (0..n * n * n)
            .map(|i| ((i * 17 + 3) as f64).cos())
            .collect();
        let projected = projector.ball(&x, 0.9)?;
        assert!(
            projector
                .norms(&projected)?
                .iter()
                .all(|s| *s < 0.9 + 1e-12)
        );
        assert!(distance(&projector.ball(&projected, 0.9)?, &projected) < 1e-12);
        for (a, b) in x.chunks_exact(n).zip(projected.chunks_exact(n)) {
            assert!((a.iter().sum::<f64>() - b.iter().sum::<f64>()).abs() < 1e-12);
        }
        for trial in 0..12 {
            let y: Vec<_> = (0..x.len())
                .map(|i| ((i * 7 + trial + 2) as f64).sin())
                .collect();
            let norm = projector.norms(&y)?.into_iter().fold(0., f64::max);
            let y: Vec<_> = y.iter().map(|v| v * 0.8 / norm).collect();
            let dot: f64 = x
                .iter()
                .zip(&projected)
                .zip(y)
                .map(|((&a, &b), c)| (a - b) * (c - b))
                .sum();
            assert!(dot <= 1e-11, "ball projection variational defect {dot}");
        }
        let mut p = vec![0.; x.len()];
        for (a, b) in x.chunks_exact(n).zip(p.chunks_exact_mut(n)) {
            simplex(a, b);
        }
        let (model, alpha, _) = projector.feasible(&p, 0.9)?;
        assert!(alpha > 0. && alpha < 1.);
        model.validate(&mut projector)?;
        for (a, b) in p.chunks_exact(n).zip(model.probabilities.chunks_exact(n)) {
            for c in 0..n {
                for d in 0..n {
                    assert_eq!(a[c].total_cmp(&a[d]), b[c].total_cmp(&b[d]));
                }
            }
        }
    }
    Ok(())
}

#[test]
fn all_three_modes_recover_the_proved_unseen_binary_sign() -> Result<()> {
    // Column-major K=[[1,1],[-1,1]], with (1,1) held out; no target in Solver.
    let samples = vec![
        Sample { cell: 0, label: 0 },
        Sample { cell: 1, label: 1 },
        Sample { cell: 2, label: 0 },
    ];
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let model = Model::new(2, 2_f64.sqrt())?;
        let mut solver = Solver::new(&model, samples.clone(), mode, 1)?;
        for step in 0..1024 {
            solver.step(&[step % 3])?;
        }
        let (out, _, _) = solver.snapshot()?;
        out.validate(&mut solver.projector)?;
        assert!(
            out.probabilities[6] > 1. - 1e-8,
            "missing p={}",
            out.probabilities[6]
        );
        assert!(solver.fit_defect(&out) < 1e-8);
        let residual = solver.full_residual()?;
        assert!(residual < 1e-8);
        let bytes = serde_json::to_vec(solver.diagnostic_state())?;
        let mut replay = Solver::new(&model, samples.clone(), mode, 1)?;
        replay.restore_diagnostic_state(serde_json::from_slice(&bytes)?)?;
        assert_eq!(out.probabilities, replay.snapshot()?.0.probabilities);
        assert!((replay.full_residual()? - residual).abs() < 1e-14);
        assert!(replay.restore_diagnostic_state(vec![0.]).is_err());
    }
    Ok(())
}

#[test]
fn minibatch_does_not_read_unselected_labels_and_loose_bounds_do_not_recover() -> Result<()> {
    let model = Model::new(3, 3.)?;
    let make = |label| vec![Sample { cell: 0, label: 0 }, Sample { cell: 1, label }];
    let mut a = Solver::new(&model, make(0), Mode::Mini, 1)?;
    let mut b = Solver::new(&model, make(2), Mode::Mini, 1)?;
    for _ in 0..4 {
        a.step(&[0])?;
        b.step(&[0])?;
    }
    assert_eq!(a.shadow, b.shadow);
    a.step(&[1])?;
    b.step(&[1])?;
    assert!(distance(&a.shadow, &b.shadow) > 0.1);
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let mut s = Solver::new(&model, make(2), mode, 1)?;
        for step in 0..50 {
            s.step(&[step % 2])?;
        }
        let (out, _, _) = s.snapshot()?;
        assert!(
            out.probabilities[6..9]
                .iter()
                .all(|p| (*p - 1. / 3.).abs() < 1e-12)
        );
    }
    Ok(())
}

#[test]
fn invalid_models_and_duplicate_observations_are_rejected() -> Result<()> {
    let good = Model::new(3, 1.)?;
    for p in [vec![], vec![f64::NAN; 27], vec![-1.; 27], vec![1.; 27]] {
        let mut bad = good.clone();
        bad.probabilities = p;
        assert!(bad.validate_shape().is_err());
    }
    let duplicate = vec![Sample { cell: 0, label: 0 }, Sample { cell: 0, label: 1 }];
    assert!(Solver::new(&good, duplicate, Mode::Full, 1).is_err());
    let mut bad = good;
    for p in bad.probabilities.as_chunks_mut::<3>().0 {
        p.copy_from_slice(&[1., 0., 0.]);
    }
    assert!(bad.validate(&mut Projector::new(3, 1)?).is_err());
    Ok(())
}
