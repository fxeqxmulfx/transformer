use super::*;
use projection::{Projector, distance, simplex, singular_budget};
use solver::{Mode, Solver};

#[test]
fn singular_projection_has_known_solutions_and_variational_property() -> Result<()> {
    for (r, b, expected) in [
        (2., 3., vec![2., 1., 0.]),
        (2., 6., vec![2., 2., 1.]),
        (10., 3., vec![2.5, 0.5, 0.]),
    ] {
        assert!(distance(&singular_budget(&[4., 2., 1.], r, b)?, &expected) < 1e-13);
    }
    for trial in 0..40 {
        let s: Vec<_> = (0..7)
            .map(|i| ((i * 7 + trial) as f64).sin().abs() * 4.)
            .collect();
        let p = singular_budget(&s, 1.7, 3.1)?;
        assert!(p.iter().all(|x| *x >= 0. && *x <= 1.7));
        assert!(p.iter().sum::<f64>() <= 3.1 + 1e-13);
        assert!(distance(&p, &singular_budget(&p, 1.7, 3.1)?) < 1e-13);
        for other in 0..30 {
            let t: Vec<_> = (0..7)
                .map(|i| ((i * 13 + other) as f64).cos().abs() * 5.)
                .collect();
            let q = singular_budget(&t, 1.7, 3.1)?;
            let dot: f64 = s
                .iter()
                .zip(&p)
                .zip(q)
                .map(|((&s, &p), q)| (s - p) * (q - p))
                .sum();
            assert!(dot < 1e-10);
        }
    }
    assert!(singular_budget(&[1., f64::NAN], 1., 1.).is_err());
    Ok(())
}

#[test]
fn combined_complex_projection_obeys_both_budgets_and_variational_inequality() -> Result<()> {
    for n in [2, 3, 4] {
        let mut projector = Projector::new(n, 2)?;
        let x: Vec<_> = (0..n * n * n)
            .map(|i| ((i * 17 + 2) as f64).sin())
            .collect();
        let p = projector.ball(&x, 1., 1.5)?;
        assert!(
            projector
                .norms(&p)?
                .iter()
                .all(|s| s.operator <= 1. + 1e-12 && s.nuclear <= 1.5 + 1e-12)
        );
        assert!(distance(&p, &projector.ball(&p, 1., 1.5)?) < 1e-12);
        for (a, b) in x.chunks_exact(n).zip(p.chunks_exact(n)) {
            assert!((a.iter().sum::<f64>() - b.iter().sum::<f64>()).abs() < 1e-12);
        }
        for trial in 0..12 {
            let mut q: Vec<_> = (0..x.len())
                .map(|i| ((i * 13 + trial + 1) as f64).cos())
                .collect();
            let factor = projector
                .norms(&q)?
                .iter()
                .map(|s| s.operator.max(s.nuclear / 1.5))
                .fold(1., f64::max);
            for q in &mut q {
                *q /= factor;
            }
            let dot: f64 = x
                .iter()
                .zip(&p)
                .zip(q)
                .map(|((&x, &p), q)| (x - p) * (q - p))
                .sum();
            assert!(dot <= 1e-10, "matrix variational defect {dot}");
        }
        let mut simplexes = vec![0.; x.len()];
        for (a, b) in x.chunks_exact(n).zip(simplexes.chunks_exact_mut(n)) {
            simplex(a, b);
        }
        let (model, alpha, _) = projector.feasible(&simplexes, 1., 1.5)?;
        assert!(alpha > 0. && alpha <= 1.);
        model.validate(&mut projector)?;
    }
    Ok(())
}

#[test]
fn redundant_nuclear_budget_preserves_original_operator_projection() -> Result<()> {
    for n in [2, 3, 4, 7] {
        let x: Vec<_> = (0..n * n * n)
            .map(|i| ((i * 7 + 1) as f64).cos() * 2.)
            .collect();
        let mut old = convex_stress_moments::projection::Projector::new(n, 1)?;
        let mut new = Projector::new(n, 1)?;
        assert!(distance(&old.ball(&x, 1.3)?, &new.ball(&x, 1.3, n as f64 * 1.3)?) < 1e-12);
    }
    Ok(())
}

#[test]
fn rank_one_recovery_from_three_observations_works_in_all_modes() -> Result<()> {
    let samples = vec![
        Sample { cell: 0, label: 0 },
        Sample { cell: 1, label: 0 },
        Sample { cell: 2, label: 0 },
    ];
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let model = Model::new(2, 2., 2.)?;
        let mut solver = Solver::new(&model, samples.clone(), mode, 1)?;
        for step in 0..4096 {
            solver.step(&[step % 3])?;
        }
        let (out, _, _) = solver.snapshot()?;
        out.validate(&mut solver.projector)?;
        assert!(
            out.probabilities[6] > 0.98,
            "unseen target coordinate {}",
            out.probabilities[6]
        );
        assert!(solver.fit_defect(&out) < 1e-3);
        let bytes = serde_json::to_vec(solver.diagnostic_state())?;
        let mut replay = Solver::new(&model, samples.clone(), mode, 1)?;
        replay.restore_diagnostic_state(serde_json::from_slice(&bytes)?)?;
        assert_eq!(replay.snapshot()?.0.probabilities, out.probabilities);
    }
    Ok(())
}

#[test]
fn full_rank_binary_recovery_is_retained_and_unconstrained_control_stays_uniform() -> Result<()> {
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let model = Model::new(2, 2_f64.sqrt(), 2. * 2_f64.sqrt())?;
        let samples = vec![
            Sample { cell: 0, label: 0 },
            Sample { cell: 1, label: 1 },
            Sample { cell: 2, label: 0 },
        ];
        let mut solver = Solver::new(&model, samples, mode, 1)?;
        for step in 0..1024 {
            solver.step(&[step % 3])?;
        }
        let (out, _, _) = solver.snapshot()?;
        assert!(out.probabilities[6] > 1. - 1e-8);
        // The bound contains the reflected points too (largest norm <= 2 * golden
        // ratio), so projection cannot transfer information to the unseen cell.
        // R=2 contains every probability table, but need not contain those points.
        let model = Model::new(2, 4., 8.)?;
        let samples = vec![
            Sample { cell: 0, label: 0 },
            Sample { cell: 1, label: 0 },
            Sample { cell: 2, label: 0 },
        ];
        let mut loose = Solver::new(&model, samples, mode, 1)?;
        for step in 0..64 {
            loose.step(&[step % 3])?;
        }
        let unseen = loose.snapshot()?.0.probabilities[6];
        assert!(
            (unseen - 0.5).abs() < 1e-12,
            "loose unseen coordinate: {unseen}"
        );
    }
    Ok(())
}

#[test]
fn selection_uses_training_certificate_and_fixed_candidate_order() -> Result<()> {
    assert_eq!(choose_by_train(&[3.8, 3.0], 3., 1e-8)?, 1);
    assert_eq!(choose_by_train(&[3., 3.8], 3., 1e-8)?, 0);
    assert_eq!(choose_by_train(&[3. + 1e-9, 3.], 3., 1e-8)?, 0);
    assert_eq!(choose_by_train(&[3.8, 3.6], 3., 1e-8)?, 1);
    assert!(choose_by_train(&[f64::NAN], 3., 1e-8).is_err());
    assert!(choose_by_train(&[2.99], 3., 1e-8).is_err());
    Ok(())
}

#[test]
fn nuclear_only_violation_and_invalid_budgets_are_rejected() -> Result<()> {
    assert!(Model::new(3, 1., 0.).is_err());
    assert!(Model::new(3, 1., f64::NAN).is_err());
    let mut model = Model::new(3, 1.1, 2.4)?;
    for j in 0..3 {
        model.probabilities[(j + 3 * j) * 3..(j + 3 * j + 1) * 3].copy_from_slice(&[1., 0., 0.]);
    }
    model.validate_shape()?;
    assert!(model.validate(&mut Projector::new(3, 1)?).is_err());
    Ok(())
}

#[test]
fn ordinary_ce_gradient_and_unselected_label_independence_are_retained() -> Result<()> {
    let p = [0.2, 0.6, 0.2];
    let mut g = [0.; 3];
    gradient(&p, 1, &mut g);
    for c in 0..3 {
        let (mut a, mut b) = (p, p);
        a[c] += 1e-5;
        b[c] -= 1e-5;
        assert!((g[c] - (loss(&a, 1) - loss(&b, 1)) / 2e-5).abs() < 1e-10);
    }
    let model = Model::new(3, 3., 3.)?;
    let sample = |label| vec![Sample { cell: 0, label: 0 }, Sample { cell: 1, label }];
    let mut a = Solver::new(&model, sample(0), Mode::Mini, 1)?;
    let mut b = Solver::new(&model, sample(2), Mode::Mini, 1)?;
    for _ in 0..5 {
        a.step(&[0])?;
        b.step(&[0])?;
    }
    assert_eq!(a.shadow, b.shadow);
    a.step(&[1])?;
    b.step(&[1])?;
    assert!(distance(&a.shadow, &b.shadow) > 0.1);
    Ok(())
}
