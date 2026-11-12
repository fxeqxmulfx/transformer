use crate::*;
use propagation::{Config, Observation, Problem, Verified};

fn obs(row: usize, column: usize, label: usize) -> Observation {
    Observation { row, column, label }
}
fn checked(p: Problem, anchor: Observation, steps: usize) -> Verified {
    let c = Config::new(anchor, steps);
    let propagation::Outcome::Complete(cert) = propagation::propagate(&p, c).unwrap() else {
        panic!("refused fixture")
    };
    propagation::verify(&p, c, &cert).unwrap()
}
fn wide(steps: usize) -> Verified {
    checked(
        Problem {
            side: 2,
            states: 4,
            vocab: 2,
            decoders: vec![vec![0, 0, 0, 1]],
            observations: vec![obs(0, 0, 1), obs(0, 1, 1), obs(1, 0, 0)],
        },
        obs(1, 0, 0),
        steps,
    )
}
fn ce(model: &Model, v: &Verified) -> f64 {
    let p = &v.certificate().problem;
    p.observations
        .iter()
        .map(|t| {
            let cell = model.shape.cell(t.row, t.column);
            loss(&model.output[cell * p.vocab..(cell + 1) * p.vocab], t.label)
        })
        .sum::<f64>()
        / p.observations.len() as f64
}
fn direct(shape: Shape, latent: &[f64], f: usize) -> Vec<C> {
    latent
        .chunks_exact(shape.states)
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(z, &p)| {
                    C::from_polar(
                        p,
                        std::f64::consts::TAU * (f * z) as f64 / shape.states as f64,
                    )
                })
                .sum()
        })
        .collect()
}
fn raw_audit(model: &Model) {
    model.validate().unwrap();
    for f in 0..model.shape.states {
        let s = linalg::svd(&direct(model.shape, &model.latent, f), model.shape.side)
            .unwrap()
            .values;
        assert!(s[0] <= model.shape.side as f64 + 1e-8);
        assert!(s.iter().sum::<f64>() <= model.shape.side as f64 + 1e-8);
    }
}

#[test]
fn full_fourier_geometry_has_independent_odd_even_dimensions() {
    for q in [2, 3, 4, 5, 8] {
        let shape = Shape {
            side: 3,
            states: q,
            vocab: 4,
        };
        let latent: Vec<_> = (0..shape.entries())
            .map(|i| ((i * 17 % 31) as f64 - 15.) / 31.)
            .collect();
        let mut fft = fft::Fourier::new(3, q).unwrap();
        let moments = fft.forward(&latent).unwrap();
        for f in 0..q {
            for (got, want) in moments[f * 9..(f + 1) * 9]
                .iter()
                .zip(direct(shape, &latent, f))
            {
                assert!((*got - want).norm() < 1e-12);
            }
        }
        let restored = fft.inverse(&moments).unwrap();
        assert!(
            restored
                .iter()
                .zip(&latent)
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
        let mut projector = projection::Projector::new(shape).unwrap();
        let (bounded, _) = projector.ball(&latent).unwrap();
        for f in 1..q {
            let values = linalg::svd(&direct(shape, &bounded, f), 3).unwrap().values;
            assert!(values[0] <= 3. + 1e-9 && values.iter().sum::<f64>() <= 3. + 1e-9);
        }
    }
}

#[test]
fn decoded_gradient_matches_finite_differences_with_unused_classes() {
    let decoder = [0, 0, 0, 1];
    let p = [0.1, 0.2, 0.3, 0.4];
    let g = solver::latent_gradient(&decoder, 4, &p, 1).unwrap();
    let objective = |x: &[f64]| {
        let mut y = vec![0.; 4];
        for (&d, &a) in decoder.iter().zip(x) {
            y[d] += a;
        }
        loss(&y, 1)
    };
    for z in 0..4 {
        let mut plus = p;
        let mut minus = p;
        plus[z] += 1e-6;
        minus[z] -= 1e-6;
        assert!((g[z] - (objective(&plus) - objective(&minus)) / 2e-6).abs() < 1e-9);
    }
    assert_eq!(g[0], g[1]);
    assert_eq!(g[1], g[2]);
    assert!((g[0] - solver::latent_gradient(&decoder, 2, &p, 1).unwrap()[0]).abs() > 0.1);
}

#[test]
fn every_mode_keeps_certified_wide_fiber_outputs_and_real_spectra() {
    let verified = wide(2);
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let mut solver = Solver::new(&verified, 0, mode).unwrap();
        for step in 0..12 {
            solver.step(&[step % 3]).unwrap();
            let endpoint = solver.snapshot(12).unwrap();
            endpoint.model.certified(&verified).unwrap();
            raw_audit(&endpoint.model);
            assert!(
                (ce(&endpoint.model, &verified) - ((std::f64::consts::E + 1.).ln() - 1.)).abs()
                    < 1e-10
            );
        }
        let cost = solver.state().costs;
        assert_eq!(
            cost.gradients,
            match mode {
                Mode::Full => 36,
                Mode::Mini => 12,
                Mode::Forward => 0,
            }
        );
    }
}

#[test]
fn endpoint_keeps_ambiguous_latent_mass_without_replacing_covered_rows() {
    let verified = wide(2);
    let mut builder = endpoint::Builder::new(&verified, 0).unwrap();
    let cert = verified.certificate();
    let mut alternate = [0.; 16];
    let branch = &cert.branches[1];
    for i in 0..2 {
        for j in 0..2 {
            let z = (branch.domains.rows[i][0] + branch.domains.columns[j][0]) % 4;
            alternate[(i + 2 * j) * 4 + z] = 1.;
        }
    }
    let mixed: Vec<_> = alternate
        .iter()
        .zip(&builder.anchor.latent)
        .map(|(&x, &a)| 0.5 * x + 0.5 * a)
        .collect();
    let endpoint = builder.build(&mixed, 12).unwrap();
    assert_eq!(endpoint.iterate_weight, 1.);
    assert_eq!(endpoint.model.latent, mixed);
    assert!(endpoint.model.latent.contains(&0.5));
    raw_audit(&endpoint.model);
}

fn partial_identity() -> Verified {
    let observations = (0..3)
        .flat_map(|i| {
            (0..3).filter_map(move |j| ((i, j) != (0, 0)).then_some(obs(i, j, (i + j) % 3)))
        })
        .collect();
    checked(
        Problem {
            side: 3,
            states: 3,
            vocab: 3,
            decoders: vec![vec![0, 1, 2]],
            observations,
        },
        obs(0, 1, 1),
        0,
    )
}

#[test]
fn partial_coverage_learns_uncovered_observations_in_all_modes() {
    let verified = partial_identity();
    let ideal = (std::f64::consts::E + 2.).ln() - 1.;
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let mut solver = Solver::new(&verified, 0, mode).unwrap();
        let initial = solver.snapshot(16).unwrap();
        let before = ce(&initial.model, &verified);
        assert!(
            before > ideal + 1e-4,
            "initial endpoint already fitted: {mode:?} {before}"
        );
        for step in 0..240 {
            solver.step(&[step % 8]).unwrap();
        }
        let endpoint = solver.snapshot(24).unwrap();
        let after = ce(&endpoint.model, &verified);
        eprintln!(
            "partial coverage {mode:?}: CE {before:.9} -> {after:.9}; weight {:.9}; gradients {}",
            endpoint.iterate_weight,
            solver.state().costs.gradients
        );
        assert!(after < before - 0.05, "{mode:?}: {before} -> {after}");
        endpoint.model.certified(&verified).unwrap();
        raw_audit(&endpoint.model);
        if mode == Mode::Forward {
            assert_eq!(solver.state().costs.gradients, 0);
        }
    }
}

#[test]
fn state_roundtrip_replays_every_mode_and_rejects_changed_instances() {
    let verified = partial_identity();
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let mut a = Solver::new(&verified, 0, mode).unwrap();
        for i in 0..5 {
            a.step(&[i % 8]).unwrap();
        }
        let state: State =
            serde_json::from_slice(&serde_json::to_vec(&a.state()).unwrap()).unwrap();
        let mut b = Solver::restore(&verified, 0, mode, &state).unwrap();
        for i in 5..16 {
            a.step(&[i % 8]).unwrap();
            b.step(&[i % 8]).unwrap();
            assert_eq!(a.state(), b.state());
        }
        assert_eq!(a.snapshot(8).unwrap().model, b.snapshot(8).unwrap().model);
        let unchanged = a.state();
        assert!(a.step(&[usize::MAX]).is_err());
        assert_eq!(a.state(), unchanged);
        let mut bad = state.clone();
        bad.gradient_sum[0] += 1.;
        assert!(Solver::restore(&verified, 0, mode, &bad).is_err());
        bad = state.clone();
        bad.z[0] = f64::NAN;
        assert!(Solver::restore(&verified, 0, mode, &bad).is_err());
        bad = state.clone();
        bad.shadow[0] += 0.1;
        assert!(Solver::restore(&verified, 0, mode, &bad).is_err());
        assert!(Solver::restore(&wide(2), 0, mode, &state).is_err());
    }
}

#[test]
fn wrong_global_minimum_is_spectrally_feasible_but_rejected_by_certificates() {
    let observations = (0..3)
        .flat_map(|i| {
            (0..3).filter_map(move |j| ((i, j) != (0, 0)).then_some(obs(i, j, usize::from(i != j))))
        })
        .collect();
    let problem = Problem {
        side: 3,
        states: 3,
        vocab: 2,
        decoders: vec![vec![0, 1, 1]],
        observations,
    };
    // Lean CM19 Anchor.lean: the diagonal anchor is a fixed-point refusal,
    // although the off-diagonal anchor identifies the missing answer.
    for steps in [1, 4, 20] {
        let poor = checked(problem.clone(), obs(1, 1, 0), steps);
        assert_eq!(poor.certificate().answers[0], vec![0, 1]);
    }
    let verified = checked(problem, obs(0, 1, 1), 3);
    assert_eq!(verified.answer(0, 0), Some(0));
    let shape = Shape {
        side: 3,
        states: 3,
        vocab: 2,
    };
    let mut latent = vec![0.; 27];
    for i in 0..3 {
        for j in 0..3 {
            let first = shape.cell(i, j) * 3;
            if i == j {
                latent[first] = 1.;
            } else {
                latent[first + 1] = 0.5;
                latent[first + 2] = 0.5;
            }
        }
    }
    latent[..3].fill(1. / 3.);
    let wrong = Model::new(shape, vec![0, 1, 1], latent).unwrap();
    raw_audit(&wrong);
    assert!((ce(&wrong, &verified) - ((std::f64::consts::E + 1.).ln() - 1.)).abs() < 1e-12);
    assert!(wrong.certified(&verified).is_err());
    let mut builder = endpoint::Builder::new(&verified, 0).unwrap();
    assert!(builder.build(&wrong.latent, 16).is_err());
}

#[test]
fn memory_and_decoder_refusals_precede_large_allocations() {
    assert!(
        memory_bound(
            Shape {
                side: usize::MAX,
                states: 4,
                vocab: 2
            },
            3,
            0
        )
        .is_err()
    );
    assert!(
        memory_bound(
            Shape {
                side: 2,
                states: 4,
                vocab: 2
            },
            3,
            u64::MAX
        )
        .is_err()
    );
    assert!(Solver::new(&wide(2), 99, Mode::Full).is_err());
    let mut builder = endpoint::Builder::new(&wide(2), 0).unwrap();
    assert!(builder.build(&builder.anchor.latent.clone(), 41).is_err());
    assert!(builder.build(&[], 1).is_err());
    let mut fft = fft::Fourier::new(2, 4).unwrap();
    assert!(fft.forward(&[f64::MAX; 16]).is_err());
    assert!(fft.inverse(&[C::new(f64::MAX, 0.); 16]).is_err());
}

#[test]
fn decoder_without_singleton_fibers_works_in_every_mode() {
    let d = [0, 0, 1, 1, 0];
    let observations = (0..4)
        .flat_map(|i| {
            (0..4).filter_map(move |j| ((i, j) != (0, 0)).then_some(obs(i, j, d[(i + 5 - j) % 5])))
        })
        .collect();
    let verified = checked(
        Problem {
            side: 4,
            states: 5,
            vocab: 2,
            decoders: vec![d.to_vec()],
            observations,
        },
        obs(0, 2, 1),
        4,
    );
    assert_eq!(verified.answer(0, 0), Some(0));
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let mut solver = Solver::new(&verified, 0, mode).unwrap();
        for i in 0..30 {
            solver.step(&[i % 15]).unwrap();
        }
        let endpoint = solver.snapshot(20).unwrap();
        endpoint.model.certified(&verified).unwrap();
        raw_audit(&endpoint.model);
        assert!(
            (ce(&endpoint.model, &verified) - ((std::f64::consts::E + 1.).ln() - 1.)).abs() < 1e-10
        );
        if mode == Mode::Forward {
            assert_eq!(solver.state().costs.gradients, 0);
        }
    }
}

#[test]
fn conflicting_bank_keeps_both_candidates_without_identifying_a_winner() {
    let verified = checked(
        Problem {
            side: 2,
            states: 4,
            vocab: 4,
            decoders: vec![vec![0, 1, 2, 3], vec![0, 1, 3, 2]],
            observations: vec![obs(0, 0, 0), obs(0, 1, 1), obs(1, 0, 2)],
        },
        obs(0, 0, 0),
        2,
    );
    assert_eq!(verified.certificate().answers[3], vec![0, 3]);
    assert_eq!(verified.answer(1, 1), None);
    for index in 0..2 {
        let model = Model::anchor(&verified, index).unwrap();
        model.certified(&verified).unwrap();
        raw_audit(&model);
        assert_eq!(model.output[3 * 4 + [3, 0][index]], 1.);
        for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
            let mut solver = Solver::new(&verified, index, mode).unwrap();
            solver.step(&[0, 1, 2]).unwrap();
            let endpoint = solver.snapshot(10).unwrap();
            endpoint.model.certified(&verified).unwrap();
            raw_audit(&endpoint.model);
            assert_eq!(solver.state().certificate.problem.decoders.len(), 2);
        }
    }
}

#[test]
fn repeated_observations_keep_gradient_weight_and_batch_accounting() {
    let base = partial_identity();
    let mut p = base.certificate().problem.clone();
    let repeated = obs(2, 0, 2);
    p.observations.push(repeated);
    let verified = checked(p, obs(0, 1, 1), 0);
    let p = &verified.certificate().problem;
    let indices: Vec<_> = (0..p.observations.len()).collect();
    let mut full = Solver::new(&verified, 0, Mode::Full).unwrap();
    let mut mini = Solver::new(&verified, 0, Mode::Mini).unwrap();
    full.step(&[]).unwrap();
    mini.step(&indices).unwrap();
    let a = full.state();
    let b = mini.state();
    assert_eq!(a.gradients, b.gradients);
    assert_eq!(a.gradient_sum, b.gradient_sum);
    assert_eq!(a.z, b.z);
    assert_eq!(a.costs, b.costs);
    let shape = Shape {
        side: 3,
        states: 3,
        vocab: 3,
    };
    let first = shape.cell(2, 0) * 3;
    let copies: Vec<_> = p
        .observations
        .iter()
        .enumerate()
        .filter_map(|(index, &t)| (t == repeated).then_some(index))
        .collect();
    assert_eq!(copies.len(), 2);
    for z in 0..3 {
        assert_eq!(
            a.gradient_sum[first + z],
            2. * a.gradients[copies[0] * 3 + z]
        );
    }
    let mut once = Solver::new(&verified, 0, Mode::Mini).unwrap();
    let mut twice = Solver::new(&verified, 0, Mode::Mini).unwrap();
    once.step(&[copies[0]]).unwrap();
    twice.step(&[copies[0], copies[0]]).unwrap();
    assert_eq!(once.state().gradient_sum, twice.state().gradient_sum);
    assert_eq!(once.state().z, twice.state().z);
    assert_eq!(once.state().costs.gradients, 1);
    assert_eq!(twice.state().costs.gradients, 2);
}

#[test]
fn arbitrary_row_column_and_output_renaming_commutes_with_updates() {
    let original = partial_identity();
    let rows = [2, 0, 1];
    let cols = [1, 2, 0];
    let labels = [2, 0, 1];
    let rename = |t: Observation| obs(rows[t.row], cols[t.column], labels[t.label]);
    let mut problem = original.certificate().problem.clone();
    for d in &mut problem.decoders {
        for y in d {
            *y = labels[*y];
        }
    }
    problem.observations = problem.observations.into_iter().map(rename).collect();
    let renamed = checked(problem, rename(obs(0, 1, 1)), 0);
    let mapping: Vec<_> = original
        .certificate()
        .problem
        .observations
        .iter()
        .map(|&t| {
            renamed
                .certificate()
                .problem
                .observations
                .iter()
                .position(|&u| u == rename(t))
                .unwrap()
        })
        .collect();
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let mut a = Solver::new(&original, 0, mode).unwrap();
        let mut b = Solver::new(&renamed, 0, mode).unwrap();
        for step in 0..40 {
            a.step(&[step % 8]).unwrap();
            b.step(&[mapping[step % 8]]).unwrap();
        }
        let x = a.snapshot(24).unwrap().model;
        let y = b.snapshot(24).unwrap().model;
        raw_audit(&x);
        raw_audit(&y);
        assert_eq!(a.state().costs, b.state().costs);
        for (i, &renamed_row) in rows.iter().enumerate() {
            for (j, &renamed_col) in cols.iter().enumerate() {
                for (c, &renamed_class) in labels.iter().enumerate() {
                    let before = x.output[x.shape.cell(i, j) * 3 + c];
                    let after =
                        y.output[y.shape.cell(renamed_row, renamed_col) * 3 + renamed_class];
                    assert!(
                        (before - after).abs() < 1e-7,
                        "{mode:?}: {i},{j},{c}: {before} != {after}"
                    );
                }
            }
        }
    }
}
