use super::*;
use metrics::Predictor;

// This finite fixture defines withheld answers for tests only. Production
// proposal/inference accepts categorical observations and has no answer rule.
const LABELS: [usize; 25] = [
    3, 0, 4, 1, 2, 0, 4, 1, 2, 3, 4, 1, 2, 3, 0, 1, 2, 3, 0, 4, 2, 3, 0, 4, 1,
];
fn observed() -> Vec<Sample> {
    LABELS
        .iter()
        .enumerate()
        .filter(|(cell, _)| *cell != 11)
        .map(|(cell, &label)| Sample { cell, label })
        .collect()
}

#[test]
fn learns_a_bijective_code_from_categorical_rectangles_and_rejects_ambiguity() -> Result<()> {
    let proposal = coding::discover(5, &observed())?;
    assert_eq!(proposal.reason, "unique_bijective_phase_code");
    assert_eq!(proposal.rank_including_constant_gauge, 4);
    assert!(proposal.constraints > 0);
    let code = proposal.coding.unwrap();
    code.validate(5)?;
    // Check even the held-out rectangle after learning from training only.
    for row in 0..5 {
        for col in 0..5 {
            assert_eq!(
                (code.encode[LABELS[row + 5 * col]] + code.encode[LABELS[0]]) % 5,
                (code.encode[LABELS[row]] + code.encode[LABELS[5 * col]]) % 5
            );
        }
    }
    let ambiguous: Vec<_> = (0..25)
        .map(|cell| Sample {
            cell,
            label: cell % 5,
        })
        .collect();
    assert_eq!(
        coding::discover(5, &ambiguous)?.reason,
        "ambiguous_phase_code"
    );
    let missing: Vec<_> = (0..25).map(|cell| Sample { cell, label: 0 }).collect();
    assert_eq!(
        coding::discover(5, &missing)?.reason,
        "unobserved_output_classes"
    );
    let composite: Vec<_> = (0..16)
        .map(|cell| Sample {
            cell,
            label: cell % 4,
        })
        .collect();
    assert_eq!(
        coding::discover(4, &composite)?.reason,
        "composite_vocabulary_unsupported"
    );
    Ok(())
}

#[test]
fn output_renaming_changes_only_the_phase_gauge_of_a_unique_proposal() -> Result<()> {
    let raw = observed();
    let original = coding::discover(5, &raw)?.coding.unwrap();
    let rename = Coding {
        encode: vec![2, 4, 1, 0, 3],
    };
    let renamed = coding::discover(5, &rename.samples(&raw)?)?.coding.unwrap();
    // All nonzero rescalings and shifts name the same set of Fourier phases.
    assert!((1..5).any(|a| (0..5).any(|b| {
        (0..5).all(|c| renamed.encode[rename.encode[c]] == (a * original.encode[c] + b) % 5)
    })));
    let mut reordered = raw.clone();
    reordered.reverse();
    assert_eq!(coding::discover(5, &raw)?, coding::discover(5, &reordered)?);
    Ok(())
}

#[test]
fn ordinary_full_vocabulary_ce_and_gradient_are_covariant() -> Result<()> {
    let code = coding::discover(5, &observed())?.coding.unwrap();
    let original = [-1.7, 0.1, 2.3, -0.4, 0.2];
    let mut encoded = [0.; 5];
    for c in 0..5 {
        encoded[code.encode[c]] = original[c];
    }
    for target in 0..5 {
        assert!((loss(&original, target) - loss(&encoded, code.encode[target])).abs() < 1e-14);
        let (mut a, mut b) = ([0.; 5], [0.; 5]);
        gradient(&original, target, &mut a);
        gradient(&encoded, code.encode[target], &mut b);
        for c in 0..5 {
            assert!((a[c] - b[code.encode[c]]).abs() < 1e-14);
        }
    }
    Ok(())
}

#[test]
fn every_mode_recovers_withheld_fixture_and_replays_saved_state() -> Result<()> {
    let raw = observed();
    let code = coding::discover(5, &raw)?.coding.unwrap();
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let initial = Table::new(5, 5., 5.)?;
        let mapped = code.samples(&raw)?;
        let mut solver = Solver::new(&initial, mapped.clone(), mode, 1)?;
        let mut order: Vec<_> = (0..raw.len()).collect();
        let mut rng = random::Random::new(10_000);
        for step in 0..128 {
            if mode == Mode::Mini && step % 5 == 0 {
                rng.shuffle(&mut order);
            }
            let batch = if mode == Mode::Mini {
                let start = step % 5 * 5;
                &order[start..(start + 5).min(order.len())]
            } else {
                &order
            };
            solver.step(batch)?;
        }
        let (table, _, _) = solver.snapshot()?;
        table.validate(&mut solver.projector)?;
        let model = Model {
            coding: code.clone(),
            table,
        };
        model.validate_shape()?;
        let pred = model.logits(&[2, 1]); // cell 11 was withheld above.
        let best = (0..5).max_by(|&a, &b| pred[a].total_cmp(&pred[b])).unwrap();
        assert_eq!(best, LABELS[11]);
        for s in &raw {
            let logits = model.logits(&[(s.cell / 5) as i64, (s.cell % 5) as i64]);
            let start = s.cell * 5;
            assert!(
                (loss(&logits, s.label)
                    - loss(
                        &model.table.probabilities[start..start + 5],
                        code.encode[s.label]
                    ))
                .abs()
                    < 1e-14
            );
        }
        let bytes = serde_json::to_vec(&model)?;
        let saved: Model = serde_json::from_slice(&bytes)?;
        saved.validate_shape()?;
        assert_eq!(saved.coding, code);
        let mut replay = Solver::new(&saved.table, mapped, mode, 1)?;
        replay.restore_diagnostic_state(solver.diagnostic_state().to_vec())?;
        assert_eq!(
            replay.snapshot()?.0.probabilities,
            saved.table.probabilities
        );
    }
    Ok(())
}

#[test]
fn forward_state_is_independent_of_batch_indices_and_validates_corruption() -> Result<()> {
    let code = coding::discover(5, &observed())?.coding.unwrap();
    let table = Table::new(5, 5., 5.)?;
    let mut a = Solver::new(&table, code.samples(&observed())?, Mode::Forward, 1)?;
    let mut b = Solver::new(&table, code.samples(&observed())?, Mode::Forward, 1)?;
    for _ in 0..4 {
        a.step(&[])?;
        b.step(&[2, 4, 1])?;
    }
    assert_eq!(a.diagnostic_state(), b.diagnostic_state());
    assert!(
        Coding {
            encode: vec![0, 0, 1, 2, 3]
        }
        .validate(5)
        .is_err()
    );
    assert!(
        Coding {
            encode: vec![0, 1, 2, 3, 5]
        }
        .validate(5)
        .is_err()
    );
    let mut duplicate = observed();
    duplicate.push(duplicate[0].clone());
    assert!(coding::discover(5, &duplicate).is_err());
    assert!(coding::discover(5, &[Sample { cell: 25, label: 0 }]).is_err());
    assert!(coding::discover(5, &[Sample { cell: 0, label: 5 }]).is_err());
    Ok(())
}

#[test]
fn explicit_memory_forecast_rejects_oversize_before_allocating_tables() -> Result<()> {
    for n in [2, 5, 97, 193] {
        assert!(memory_bound(n)? <= 2 * 1024_u64.pow(3));
    }
    assert!(memory_bound(194).is_err());
    assert!(memory_bound(usize::MAX).is_err());
    assert!(coding::discover(usize::MAX, &[]).is_err());
    Ok(())
}
