use crate::*;
use convex_stress_paired::projection::Projector;

fn from_rows(values: &[&[usize]], vocab: usize, omitted: &[(usize, usize)]) -> Training {
    let n = values.len();
    let mut training = Training::new(
        Shape {
            side: n,
            orbits: vocab,
            vocab,
        },
        Decoder::fold(vocab),
    )
    .unwrap();
    let samples: Vec<_> = (0..n)
        .flat_map(|j| {
            (0..n).filter_map(move |i| {
                (!omitted.contains(&(i, j))).then_some(Observation {
                    cell: i + n * j,
                    label: values[i][j],
                })
            })
        })
        .collect();
    training.observe(&samples).unwrap();
    training
}

fn held() -> Training {
    from_rows(&[&[0, 1, 2], &[1, 0, 1], &[2, 1, 0]], 3, &[(2, 1)])
}

fn held_certificate() -> Certificate {
    Certificate {
        row: vec![0, 1, 2],
        col: vec![0, 4, 3],
        top: 0,
        bot: 1,
        left: 0,
        right: 2,
    }
}

#[test]
fn lean_folded_witness_is_discovered_and_independently_checked() -> Result<()> {
    let training = held();
    let checked = check(&training, &held_certificate())?;
    assert_eq!(checked.covered_cells(), 9);
    assert_eq!(checked.labels()[5], Some(1));
    assert_eq!(training.at(2, 1), None);
    let found = discover(&training, Limits::default())?;
    assert_eq!(found.stop, SearchStop::CompleteCoverage);
    let found = found.verified.unwrap();
    assert_eq!(found.labels(), checked.labels());
    assert!(found.full_training_fit());
    let serialized = serde_json::to_string(found.certificate())?;
    let replay: Certificate = serde_json::from_str(&serialized)?;
    assert_eq!(found, check(&training, &replay)?);
    let anchor = found.anchor(&training)?;
    anchor.validate(&Projector::new(training.shape())?)?;
    assert_eq!(anchor.output[5 * 3 + 1], 1.);
    assert_eq!(anchor.latent_half.len(), 27);
    Ok(())
}

#[test]
fn lean_wrong_minimum_has_ideal_training_ce_but_violates_the_certificate() -> Result<()> {
    let training = from_rows(&[&[0, 1, 1], &[1, 0, 1], &[1, 1, 0]], 2, &[(0, 0)]);
    let checked = discover(&training, Limits::default())?.verified.unwrap();
    assert_eq!(checked.labels()[0], Some(0));
    let anchor = checked.anchor(&training)?;
    let mut half = anchor.latent_half.clone();
    half[..2].copy_from_slice(&[1. / 3., 1. / 3.]);
    let wrong = Model::new(training.shape(), training.decoder().clone(), 3., 3., half)?;
    let norms = wrong.validate(&Projector::new(training.shape())?)?;
    assert!((norms[0].nuclear - 3.).abs() < 1e-12);
    assert_eq!(wrong.output[1], 2. / 3.);
    assert!(wrong.output[1] > wrong.output[0]);
    let ce = |model: &Model| {
        training
            .unique_observations()
            .iter()
            .map(|s| {
                let row = &model.output[s.cell * 2..s.cell * 2 + 2];
                row.iter().map(|v| v.exp()).sum::<f64>().ln() - row[s.label]
            })
            .sum::<f64>()
    };
    assert_eq!(ce(&wrong), ce(&anchor));
    assert_eq!(anchor.output[0], 1.);
    Ok(())
}

#[test]
fn missing_rectangle_refuses_both_lean_ambiguous_completions() -> Result<()> {
    let training = from_rows(&[&[0, 1], &[1, 0]], 2, &[(1, 1)]);
    let result = discover(&training, Limits::default())?;
    assert_eq!(result.stop, SearchStop::SearchExhausted);
    assert!(result.verified.is_none());
    // The OR and XOR completions of CM16 agree on precisely these samples.
    for col in [vec![0, 1], vec![0, 2]] {
        let cert = Certificate {
            row: vec![0, 1],
            col,
            top: 0,
            bot: 1,
            left: 0,
            right: 1,
        };
        assert!(check(&training, &cert).is_err());
    }
    Ok(())
}

#[test]
fn checker_rejects_invalid_phases_reached_labels_and_unobserved_anchors() {
    let training = held();
    for mutation in 0..7 {
        let mut c = held_certificate();
        match mutation {
            0 => c.row[0] = 5,
            1 => c.row[1] = c.row[0],
            2 => c.col[2] = c.col[0],
            3 => c.col[1] = 0,
            4 => {
                c.top = 2;
                c.left = 1;
            }
            5 => c.right = 3,
            _ => {
                c.col.pop();
            }
        }
        assert!(check(&training, &c).is_err(), "mutation {mutation}");
    }
}

#[test]
fn partial_coverage_does_not_claim_unreached_answers_or_a_global_fit() -> Result<()> {
    let mut training = Training::new(
        Shape {
            side: 4,
            orbits: 3,
            vocab: 3,
        },
        Decoder::fold(3),
    )?;
    let mut samples: Vec<_> = held()
        .unique_observations()
        .iter()
        .map(|s| Observation {
            cell: s.cell % 3 + 4 * (s.cell / 3),
            label: s.label,
        })
        .collect();
    samples.push(Observation { cell: 15, label: 2 });
    training.observe(&samples)?;
    let mut c = held_certificate();
    c.row.push(0);
    c.col.push(0);
    let checked = check(&training, &c)?;
    assert_eq!(checked.covered_cells(), 9);
    assert_eq!(checked.labels()[2 + 4], Some(1));
    assert_eq!(checked.labels()[15], None);
    assert!(!checked.full_training_fit());
    let anchor = checked.anchor(&training)?;
    anchor.validate(&Projector::new(training.shape())?)?;
    assert_eq!(anchor.output[15 * 3], 1.);
    assert_eq!(
        discover(&training, Limits::default())?
            .verified
            .unwrap()
            .labels(),
        checked.labels()
    );
    Ok(())
}

#[test]
fn retained_observations_are_invariant_to_order_batches_and_repetition() -> Result<()> {
    let complete = held();
    let reference = discover(&complete, Limits::default())?;
    let mut observations = complete.unique_observations();
    observations.reverse();
    for size in 1..=observations.len() {
        let mut streamed = Training::new(complete.shape(), complete.decoder().clone())?;
        for batch in observations.chunks(size) {
            streamed.observe(batch)?;
            streamed.observe(batch)?;
        }
        assert_eq!(streamed.exposures(), 16);
        assert_eq!(
            streamed.unique_observations(),
            complete.unique_observations()
        );
        assert_eq!(discover(&streamed, Limits::default())?, reference);
    }
    let mut conflict = complete.clone();
    let before = conflict.unique_observations();
    assert!(
        conflict
            .observe(&[
                Observation { cell: 5, label: 1 },
                Observation { cell: 0, label: 2 }
            ])
            .is_err()
    );
    assert_eq!(conflict.unique_observations(), before);
    assert_eq!(conflict.exposures(), 8);
    Ok(())
}

#[test]
fn categorical_input_permutations_and_decoder_output_renaming_preserve_answers() -> Result<()> {
    let original = held();
    let rows = [2, 0, 1];
    let cols = [1, 2, 0];
    let classes = [2, 0, 1];
    let mut renamed = Training::new(
        original.shape(),
        Decoder {
            orbit_class: classes.to_vec(),
        },
    )?;
    let observations: Vec<_> = original
        .unique_observations()
        .iter()
        .map(|s| Observation {
            cell: rows[s.cell % 3] + 3 * cols[s.cell / 3],
            label: classes[s.label],
        })
        .collect();
    renamed.observe(&observations)?;
    let baseline = discover(&original, Limits::default())?.verified.unwrap();
    let result = discover(&renamed, Limits::default())?.verified.unwrap();
    for (j, &column) in cols.iter().enumerate() {
        for (i, &row) in rows.iter().enumerate() {
            assert_eq!(
                result.labels()[row + 3 * column],
                baseline.labels()[i + 3 * j].map(|x| classes[x])
            );
        }
    }
    Ok(())
}

#[test]
fn unsupported_decoders_malformed_samples_and_memory_overflow_are_refused() -> Result<()> {
    let shape = Shape {
        side: 3,
        orbits: 3,
        vocab: 4,
    };
    assert!(
        Training::new(
            shape,
            Decoder {
                orbit_class: vec![0, 1, 1]
            }
        )
        .is_err()
    );
    let mut training = Training::new(
        shape,
        Decoder {
            orbit_class: vec![3, 0, 2],
        },
    )?;
    assert!(
        training
            .observe(&[Observation { cell: 0, label: 1 }])
            .is_err()
    );
    assert!(
        training
            .observe(&[Observation { cell: 9, label: 0 }])
            .is_err()
    );
    assert!(
        training
            .observe(&[Observation { cell: 0, label: 4 }])
            .is_err()
    );
    assert_eq!(training.exposures(), 0);
    for huge in [
        Shape {
            side: usize::MAX,
            ..shape
        },
        Shape {
            orbits: usize::MAX,
            ..shape
        },
    ] {
        assert!(memory_bound(huge).is_err());
        assert!(Training::new(huge, Decoder::fold(3)).is_err());
    }
    let bytes = memory_bound(shape)?;
    assert_eq!(preflight(shape, bytes)?, bytes);
    assert!(preflight(shape, bytes - 1).is_err());
    assert!(bytes < RUN_MEMORY_BYTES);
    assert_eq!(RUN_MEMORY_BYTES, 68_719_476_736);
    Ok(())
}

#[test]
fn limits_count_work_without_claiming_inconsistency() -> Result<()> {
    let training = held();
    let result = discover(
        &training,
        Limits {
            rectangles: 0,
            candidates: 0,
        },
    )?;
    assert_eq!(result.stop, SearchStop::SearchLimit);
    assert!(result.verified.is_none());
    assert_eq!(result.costs.rectangles, 0);
    assert_eq!(result.costs.candidates, 0);
    assert!(
        discover(
            &training,
            Limits {
                rectangles: u64::MAX,
                candidates: 0
            }
        )
        .is_err()
    );
    Ok(())
}

/// Enumerate all phase completions after removing only the harmless global shift.
/// This reference does not use the proposal, rectangle conditions or reach code.
fn all_tables(n: usize, q: usize) -> Vec<Vec<usize>> {
    (0..q.pow((2 * n - 1) as u32))
        .map(|mut index| {
            let mut values = vec![0; 2 * n];
            for value in &mut values[1..] {
                *value = index % q;
                index /= q;
            }
            (0..n * n)
                .map(|cell| {
                    let z = (values[cell % n] + values[n + cell / n]) % q;
                    z.min(q - z)
                })
                .collect()
        })
        .collect()
}

#[test]
fn exhaustive_three_state_completions_validate_every_accepted_missing_answer() -> Result<()> {
    let tables = all_tables(3, 3);
    assert_eq!(tables.len(), 243);
    let mut accepted = 0;
    let mut refused = 0;
    for target in &tables {
        for withheld in 0..9 {
            let shape = Shape {
                side: 3,
                orbits: 2,
                vocab: 2,
            };
            let mut training = Training::new(shape, Decoder::fold(2))?;
            let observations: Vec<_> = target
                .iter()
                .enumerate()
                .filter_map(|(cell, &label)| {
                    (cell != withheld).then_some(Observation { cell, label })
                })
                .collect();
            training.observe(&observations)?;
            let result = discover(&training, Limits::default())?;
            if let Some(checked) = result.verified {
                let compatible: Vec<_> = tables
                    .iter()
                    .filter(|candidate| observations.iter().all(|s| candidate[s.cell] == s.label))
                    .collect();
                assert!(!compatible.is_empty());
                for (cell, label) in checked.labels().iter().enumerate() {
                    if let Some(label) = label {
                        assert!(compatible.iter().all(|candidate| candidate[cell] == *label));
                    }
                }
                accepted += usize::from(checked.labels()[withheld].is_some());
            } else {
                refused += 1;
            }
        }
    }
    assert!(accepted > 0 && refused > 0);
    Ok(())
}
