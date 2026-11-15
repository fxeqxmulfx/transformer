use crate::{
    constraints::Domain,
    data::Record,
    head::Head,
    training::{Mode, Policy},
    *,
};

mod projection_checks;

fn coords() -> Vec<f64> {
    (0..5).map(|c| c as f64 / 4.).collect()
}
fn features(x: usize) -> Vec<Features> {
    (0..3)
        .map(|p| {
            if x < 3 {
                let q = match p {
                    0 => x as f64 / 2.,
                    1 => x as f64 - 1.,
                    _ => 0.,
                };
                [q, 1., 0., 0., 0.]
            } else {
                let mut h = [0.; 5];
                h[x - 1] = 1.;
                h
            }
        })
        .collect()
}
fn records() -> Vec<Record> {
    [0, 0, 1, 3, 4]
        .into_iter()
        .enumerate()
        .map(|(i, x)| Record {
            id: 100 + i,
            features: features(x),
            label: [1, 2, 3, 0, 4, 4][x],
        })
        .collect()
}
fn theta() -> Vec<Parameters> {
    vec![[2., 1., 0., 4., 4., 4.], [0.5, 1., 0., 2., 2., 2.], [0.; 6]]
}
fn domain() -> Result<Domain> {
    Domain::new(&records(), &coords(), vec![16.; 3], 128., 1.)
}
fn policy(mode: Mode) -> Policy {
    Policy {
        mode,
        epochs: if mode == Mode::Forward { 0 } else { 4 },
        batch: 2,
        learning_rate: 8.,
        initial_total_mass: 12.,
        projection_cycles: 10000,
        projection_tolerance: 1e-10,
    }
}

#[test]
fn complete_products_match_complex_multiplication_and_the_lean_witness() -> Result<()> {
    let head = Head::new(coords(), theta())?;
    assert_eq!(head.coefficients.len(), 3 * 16 * 5);
    assert_eq!(head.coefficients[3 * 5 + 4], 3.);
    assert_eq!(charge(&head.theta), 78.);
    assert!(head.max_column_charge() <= charge(&head.theta));
    let mut inputs: Vec<_> = (0..6).map(features).collect();
    inputs.push(vec![[-3., 2., 1., -1., 0.5]; 3]);
    for h in inputs {
        let z = head.logits(&h)?;
        let independent = head.independent_logits(&h)?;
        assert!(head::features(&h).iter().flatten().all(|x| x.abs() <= 1.));
        for (c, (&a, &b)) in z.iter().zip(&independent).enumerate() {
            let e = head.coordinates[c];
            let direct: f64 = h
                .iter()
                .zip(&head.theta)
                .map(|(h, t)| {
                    2. * e * h.iter().zip(t).map(|(x, w)| x * w).sum::<f64>() - t[5] * e * e
                })
                .sum::<f64>()
                / head::normalizer(&h);
            assert!((a - b).abs() < 1e-12 && (a - direct).abs() < 1e-12);
        }
    }
    assert_eq!(argmax(&head.logits(&features(2))?), 3);
    assert_eq!(argmax(&head.logits(&features(5))?), 4);
    assert!(records().iter().all(|r| r.label != 3));
    let mut changed = head.clone();
    changed.coefficients[0] += 1.;
    assert!(changed.validate().is_err());
    Ok(())
}

#[test]
fn every_joint_ce_derivative_matches_finite_differences_and_true_tail_weighting() -> Result<()> {
    let rows = records();
    let head = Head::new(coords(), theta())?;
    let (loss, gradient) = training::gradient(&head, &rows)?;
    assert_eq!(loss, training::score(&head, &rows)?);
    for p in 0..3 {
        for j in 0..6 {
            let mut plus = theta();
            plus[p][j] += 1e-5;
            let mut minus = theta();
            minus[p][j] -= 1e-5;
            let finite = (training::score(&Head::new(coords(), plus)?, &rows)?
                - training::score(&Head::new(coords(), minus)?, &rows)?)
                / 2e-5;
            assert!((gradient[p][j] - finite).abs() < 5e-10, "parameter{p}/{j}");
        }
    }
    let mut weighted = [[0.; 6]; 3];
    for batch in rows.chunks(2) {
        let (_, g) = training::gradient(&head, batch)?;
        for p in 0..3 {
            for j in 0..6 {
                weighted[p][j] += g[p][j] * batch.len() as f64 / rows.len() as f64;
            }
        }
    }
    for (a, b) in weighted.iter().flatten().zip(gradient.iter().flatten()) {
        assert!((a - b).abs() < 1e-15);
    }
    let alternative: Vec<_> = theta()
        .into_iter()
        .enumerate()
        .map(|(p, t)| t.map(|x| x * if p == 0 { 0.5 } else { 2. }))
        .collect();
    let mix: Vec<_> = theta()
        .iter()
        .zip(&alternative)
        .map(|(a, b)| std::array::from_fn(|j| 0.37 * a[j] + 0.63 * b[j]))
        .collect();
    assert!(
        training::score(&Head::new(coords(), mix)?, &rows)?
            <= 0.37 * loss
                + 0.63 * training::score(&Head::new(coords(), alternative)?, &rows)?
                + 1e-12
    );
    Ok(())
}

#[test]
fn all_modes_replay_preserve_ids_and_distinguish_projection_from_gradients() -> Result<()> {
    let rows = records();
    for mode in [Mode::Forward, Mode::Full, Mode::Mini] {
        let out = training::train(&rows, coords(), vec![16.; 3], 128., 1., policy(mode))?;
        training::replay(&out, &rows)?;
        assert!(out.final_ce <= out.initial_ce + 1e-9);
        assert!(out.domain.violation(&out.head.theta) <= 1e-10);
        assert!(out.head.theta[2].iter().all(|x| x.abs() < 1e-8));
        assert_eq!(argmax(&out.head.logits(&features(2))?), 3);
        assert_eq!(argmax(&out.head.logits(&features(5))?), 4);
        if mode == Mode::Forward {
            assert_eq!(out.work.gradient_batches, 0);
            assert_eq!(out.work.gradient_records, 0);
            assert_eq!(out.work.effective_updates, 0);
            assert!(out.work.projection_cycles > 0);
        } else {
            assert_eq!(out.work.gradient_records, 20);
            assert!(out.work.effective_updates > 0);
        }
        if mode == Mode::Mini {
            assert_eq!(out.steps[0].record_ids, [100, 101]);
            assert_eq!(out.steps[2].record_ids, [104]);
        }
        let encoded = serde_json::to_vec(&out)?;
        let decoded: training::Outcome = serde_json::from_slice(&encoded)?;
        assert_eq!(out, decoded);
        let mut corrupt = decoded;
        corrupt.initial_ce += 1.;
        assert!(training::replay(&corrupt, &rows).is_err());
    }
    Ok(())
}

#[test]
fn unsupported_role_stays_ambiguous_and_copy_coverage_is_explicit() -> Result<()> {
    let rows: Vec<_> = [0, 1]
        .into_iter()
        .map(|x| Record {
            id: x,
            features: vec![features(x)[0]],
            label: x + 1,
        })
        .collect();
    let d = Domain::new(&rows, &coords(), vec![32.], 128., 1.)?;
    assert!(d.span_residual(&[features(3)[0]])?[0] > 0.99);
    assert!(d.span_residual(&[features(2)[0]])?[0] < 1e-12);
    assert!(d.span_residual(&[features(5)[0]])?[0] < 1e-12);
    for (r, winner) in [(0., 0), (4., 4)] {
        let t = vec![[2., 1., r, 0., 4., 4.]];
        assert!(d.violation(&t) < 1e-12);
        let h = Head::new(coords(), t)?;
        assert_eq!(argmax(&h.logits(&[features(3)[0]])?), winner);
    }
    Ok(())
}

#[test]
fn bounded_shapes_and_corrupt_constraints_are_rejected() -> Result<()> {
    assert!(data::forecast(MAX_RECORDS, MAX_PROGRAMS, 1024).is_err());
    let mut rows = records();
    rows[1].id = rows[0].id;
    assert!(Domain::new(&rows, &coords(), vec![16.; 3], 128., 1.).is_err());
    assert!(Head::new(vec![0., 0.], theta()).is_err());
    let mut d = domain()?;
    d.basis[0][0][0] += 0.1;
    assert!(d.validate().is_err());
    assert!(domain()?.project(&theta(), 0, 1e-10).is_err());
    assert!(domain()?.project(&[[1e14; 6]; 3], 1, 1e-12).is_err());
    assert!(domain()?.violation(&[[f64::NAN; 6]; 3]).is_infinite());
    assert!(domain()?.violation(&[]).is_infinite());
    Ok(())
}

#[test]
fn contradictory_identical_inputs_cannot_be_declared_a_positive_mass_solution() -> Result<()> {
    let rows = vec![
        Record {
            id: 0,
            features: features(0),
            label: 1,
        },
        Record {
            id: 1,
            features: features(0),
            label: 2,
        },
    ];
    let d = Domain::new(&rows, &coords(), vec![16.; 3], 128., 1.)?;
    assert!(d.project(&[[0.; 6]; 3], 128, 1e-10).is_err());
    let mut p = policy(Mode::Forward);
    p.projection_cycles = 128;
    assert!(training::train(&rows, coords(), vec![16.; 3], 128., 1., p).is_err());
    Ok(())
}
