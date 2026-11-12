use crate::{endpoint::*, solver::*, *};
use convex_stress_paired::{C, linalg, projection::Projector, weight};

fn partial() -> Result<(Training, Certificate)> {
    let shape = Shape {
        side: 4,
        orbits: 3,
        vocab: 3,
    };
    let mut t = Training::new(shape, Decoder::fold(3))?;
    let c = Certificate {
        row: vec![0, 1, 2, 0],
        col: vec![0, 4, 3, 0],
        top: 0,
        bot: 1,
        left: 0,
        right: 2,
    };
    let mut observations = vec![];
    for j in 0..3 {
        for i in 0..3 {
            if (i, j) != (2, 1) {
                observations.push(Observation {
                    cell: i + 4 * j,
                    label: t.decoder().class_of_state((c.row[i] + c.col[j]) % 5),
                });
            }
        }
    }
    observations.push(Observation { cell: 15, label: 2 });
    t.observe(&observations)?;
    Ok((t, c))
}

fn ce(t: &Training, m: &Model) -> f64 {
    t.unique_observations()
        .iter()
        .map(|s| {
            let row = &m.output[s.cell * m.shape.vocab..(s.cell + 1) * m.shape.vocab];
            row.iter().map(|x| x.exp()).sum::<f64>().ln() - row[s.label]
        })
        .sum::<f64>()
        / t.unique_observations().len() as f64
}

/// A direct DFT of the reflected coordinates, independent of the cosine bank
/// used for projection and endpoint selection. LAPACK SVD is shared.
fn raw_check(e: &Endpoint) -> Result<()> {
    let m = &e.model;
    let q = m.shape.latent_states();
    for f in 1..q {
        let matrix: Vec<_> = (0..m.shape.cells())
            .map(|cell| {
                (0..q)
                    .map(|z| {
                        let angle = std::f64::consts::TAU * ((f * z) % q) as f64 / q as f64;
                        C::from_polar(m.latent_half[cell * m.shape.orbits + z.min(q - z)], angle)
                    })
                    .sum::<C>()
            })
            .collect();
        let s = linalg::svd(&matrix, m.shape.side)?.values;
        assert!(s[0] <= m.radius * (1. + 2e-10));
        assert!(s.iter().sum::<f64>() <= m.nuclear_budget * (1. + 2e-10));
        let reported = &e.norms[f.min(q - f) - 1];
        assert!((s[0] - reported.operator).abs() < 1e-10);
        assert!((s.iter().sum::<f64>() - reported.nuclear).abs() < 1e-10);
    }
    Ok(())
}

#[test]
fn all_modes_exclude_the_lean_wrong_global_minimum() -> Result<()> {
    let mut t = Training::new(
        Shape {
            side: 3,
            orbits: 2,
            vocab: 2,
        },
        Decoder::fold(2),
    )?;
    let observations: Vec<_> = (1..9)
        .map(|cell| Observation {
            cell,
            label: usize::from(cell % 3 != cell / 3),
        })
        .collect();
    t.observe(&observations)?;
    let v = discover(&t, Limits::default())?.verified.unwrap();
    let builder = EndpointBuilder::new(&t, v.certificate())?;
    let mut wrong = builder.anchor().latent_half.clone();
    wrong[..2].fill(1. / 3.);
    assert!(builder.build(&wrong, 24).is_err());
    // Uniform mixing was previously used for feasibility; it erases fixed rows.
    let mixed: Vec<_> = builder
        .anchor()
        .latent_half
        .iter()
        .map(|x| x / 2. + 1. / 6.)
        .collect();
    assert!(builder.build(&mixed, 24).is_err());
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let mut solver = Solver::new(&t, v.certificate(), mode)?;
        for step in 0..16 {
            solver.step(&[step % observations.len()])?;
            let e = solver.snapshot(12)?;
            check_equalities(&e.model, &v)?;
            assert_eq!(e.model.output[0], 1.);
            assert_eq!(e.model.output[1], 0.);
            assert!((ce(&t, &e.model) - ((std::f64::consts::E + 1.).ln() - 1.)).abs() < 1e-12);
            raw_check(&e)?;
        }
        let costs = solver.state().costs;
        let saved = solver.state();
        let (residual, diagnostic) = solver.full_residual()?;
        assert!(residual.is_finite() && residual >= 0.);
        assert_eq!(
            diagnostic.gradient_evaluations,
            if mode == Mode::Forward { 0 } else { 8 }
        );
        assert_eq!(solver.state(), saved);
        assert_eq!(costs.row_projections, 32);
        assert_eq!(costs.spectral_projections, 16);
        assert_eq!(
            costs.gradient_evaluations,
            match mode {
                Mode::Full => 128,
                Mode::Mini => 16,
                Mode::Forward => 0,
            }
        );
    }
    Ok(())
}

#[test]
fn endpoint_repair_preserves_partial_certificate_and_measures_returned_table() -> Result<()> {
    let (t, c) = partial()?;
    let builder = EndpointBuilder::new(&t, &c)?;
    let mut half = builder.anchor().latent_half.clone();
    for cell in 0..16 {
        if builder.verified().labels()[cell].is_none() {
            let orbit = (7 * cell + cell / 4) % 3;
            half[cell * 3..cell * 3 + 3].fill(0.);
            half[cell * 3 + orbit] = 1. / weight(orbit);
        }
    }
    let norms = Projector::new(t.shape())?.norms(&half)?;
    assert!(norms.iter().any(|s| s.nuclear > 4.01));
    let zero = builder.build(&half, 0)?;
    assert_eq!(zero.iterate_weight, 0.);
    assert_eq!(zero.model, *builder.anchor());
    let repaired = builder.build(&half, 24)?;
    assert!(repaired.iterate_weight < 1.);
    assert_eq!(repaired.norm_evaluations, 25);
    check_equalities(&repaired.model, builder.verified())?;
    raw_check(&repaired)?;
    assert!(builder.build(&half, 41).is_err());
    Ok(())
}

#[test]
fn partial_certificate_leaves_uncovered_observations_to_the_solver() -> Result<()> {
    let (t, c) = partial()?;
    let v = check(&t, &c)?;
    assert_eq!(v.labels()[15], None);
    let anchor = v.anchor(&t)?;
    assert_eq!(anchor.output[45 + 2], 0.);
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let mut solver = Solver::new(&t, &c, mode)?;
        let index = solver.sample_index(15).unwrap();
        assert_eq!(solver.sample_index(6), None);
        for _ in 0..256 {
            solver.step(&[index])?;
        }
        let e = solver.snapshot(32)?;
        check_equalities(&e.model, &v)?;
        raw_check(&e)?;
        assert!(
            ce(&t, &e.model) < ce(&t, &anchor) - 0.05,
            "CE {} vs {}, weight {}",
            ce(&t, &e.model),
            ce(&t, &anchor),
            e.iterate_weight
        );
        assert!(
            e.model.output[47] > 0.9,
            "mass {}, weight {}",
            e.model.output[47],
            e.iterate_weight
        );
    }
    Ok(())
}

#[test]
fn complete_state_roundtrips_and_replays_each_update_mode() -> Result<()> {
    let (t, c) = partial()?;
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let mut first = Solver::new(&t, &c, mode)?;
        first.step(&[1, 2])?;
        first.step(&[3, 4])?;
        let saved = serde_json::to_vec(&first.state())?;
        let mut replay = Solver::new(&t, &c, mode)?;
        replay.restore(serde_json::from_slice(&saved)?)?;
        assert_eq!(first.state(), replay.state());
        assert_eq!(first.residual(), replay.residual());
        for batch in [vec![5, 6], vec![7, 0], vec![8], vec![]] {
            first.step(&batch)?;
            replay.step(&batch)?;
            assert_eq!(first.state(), replay.state());
            let a = first.snapshot(12)?;
            let b = replay.snapshot(12)?;
            assert_eq!(a.model, b.model);
            assert_eq!(a.iterate_weight, b.iterate_weight);
        }
    }
    Ok(())
}

#[test]
fn invalid_batches_and_mismatched_saved_instances_fail_without_mutation() -> Result<()> {
    let (t, c) = partial()?;
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let mut solver = Solver::new(&t, &c, mode)?;
        solver.step(&[0])?;
        let original = solver.state();
        assert!(solver.step(&[0, 999]).is_err());
        assert_eq!(original, solver.state());
        for mutation in 0..9 {
            let mut bad = original.clone();
            match mutation {
                0 => bad.certificate.row[0] += 1,
                1 => bad.observations[0].label += 1,
                2 => bad.decoder.orbit_class.swap(0, 1),
                3 => bad.mode = "different_mode".into(),
                4 => bad.updates += 1,
                5 => bad.z[0] = f64::NAN,
                6 => bad.gradients.pop().map(|_| ()).unwrap(),
                7 => bad.costs.row_projections += 1,
                _ => bad.residual = -1.,
            }
            assert!(solver.restore(bad).is_err(), "mutation {mutation}");
            assert_eq!(original, solver.state());
        }
    }
    Ok(())
}
