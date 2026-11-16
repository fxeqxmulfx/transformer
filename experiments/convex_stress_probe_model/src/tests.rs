use super::*;
use features::{Alphabet, Program, Relation};
use input::{PROCESS, Prepared};
use joint::training::Mode;
use probes::boundary::{Predictor, Training, TrainingRow};
const CALLER: u128 = 1024 * 1024;
fn work() -> Work {
    Work::new(2_000_000_000_000).unwrap()
}
fn training(rows: &[(Vec<i64>, Vec<i64>)], vocabulary: usize) -> Training<'_> {
    Training {
        vocabulary,
        rows: rows
            .iter()
            .enumerate()
            .map(|(original_index, (tokens, targets))| TrainingRow {
                original_index,
                tokens,
                targets,
            })
            .collect(),
    }
}
fn ordered(n: usize) -> Vec<(Vec<i64>, Vec<i64>)> {
    let pairs: Vec<_> = (1..=4)
        .flat_map(|a| (1..=4).filter(move |&b| a != b).map(move |b| (a, b)))
        .collect();
    (0..n)
        .map(|i| {
            let (a, b) = pairs[i % pairs.len()];
            (vec![a, b], vec![-100, if a < b { 6 } else { 7 }])
        })
        .collect()
}
#[test]
fn interpreters_match_every_member_on_varied_visible_prefixes() -> Result<()> {
    let catalog = features::catalog();
    assert_eq!(catalog.len(), 176);
    let mut w = work();
    for symbols in [vec![3, 5], vec![2, 3, 4, 5]] {
        let a = Alphabet {
            vocabulary: 70,
            symbols,
        };
        let mut rows = vec![
            vec![1, 3, 5, 5, 3, 18, 42, 43, 44],
            vec![1, 3, 7, 4, 8, 3],
            vec![4, 4],
            vec![1],
        ];
        let mut random = 42u64;
        for length in 2..40 {
            let mut row = vec![1];
            for _ in 0..length {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                row.push(2 + (random >> 40) as i64 % 16);
            }
            row.push(18);
            row.extend([45, 46, 47, 48]);
            rows.push(row);
        }
        for row in rows {
            for n in 1..=row.len() {
                for &p in &catalog {
                    let s = features::state(p, &a, &row[..n], &mut w)?;
                    assert!(s < features::STATES);
                    assert_eq!(
                        s,
                        reference::state(p, &a, &row[..n], &mut w)?,
                        "{p:?}, {:?}",
                        &row[..n]
                    );
                }
            }
        }
    }
    Ok(())
}
#[test]
fn chunks_context_end_and_walk_have_independent_expected_values() -> Result<()> {
    let a = Alphabet {
        vocabulary: 70,
        symbols: vec![3, 5],
    };
    let mut w = work();
    let p = Program::Chunk {
        count: 2,
        interleaved: false,
        context: Some(0),
    };
    assert_eq!(features::state(p, &a, &[1, 3, 5, 5, 3, 18], &mut w)?, 5); // digits0,1,0
    assert_eq!(
        features::state(p, &a, &[1, 3, 5, 5, 3, 18, 61], &mut w)?,
        11
    ); //1,0,0
    assert_eq!(
        features::state(p, &a, &[1, 3, 5, 5, 3, 18, 0, 0], &mut w)?,
        1
    );
    assert_eq!(features::state(p, &a, &[1, 3, 5, 5, 3], &mut w)?, 0);
    let prefix = [1, 3, 6, 3, 8, 6, 9, 3];
    let walk = |hops, latest| Program::Walk {
        hops,
        offset: 1,
        latest,
    };
    assert_eq!(features::state(walk(1, false), &a, &prefix, &mut w)?, 8);
    assert_eq!(features::state(walk(1, true), &a, &prefix, &mut w)?, 10);
    assert_eq!(features::state(walk(2, false), &a, &prefix, &mut w)?, 5);
    assert_eq!(features::state(walk(2, true), &a, &prefix, &mut w)?, 8);
    Ok(())
}
#[test]
fn all_compatible_programs_and_original_duplicate_ids_survive() -> Result<()> {
    let rows = ordered(130);
    let mut w = work();
    let input = Prepared::new(training(&rows, 8), CALLER, PROCESS, &mut w)?;
    let expected: Vec<_> = input
        .programs
        .iter()
        .enumerate()
        .filter_map(|(p, program)| {
            matches!(
                program,
                Program::Relation {
                    relation: Relation::Less | Relation::Compare,
                    ..
                }
            )
            .then_some(p)
        })
        .collect();
    assert_eq!(input.partition.retained, expected);
    assert_eq!(input.observations().len(), 130);
    for (i, obs) in input.observations().iter().enumerate() {
        assert_eq!((obs.row, obs.position), (i, 1));
    }
    for (p, decision) in input.partition.decisions.iter().enumerate() {
        if let partition::Decision::Conflict(c) = decision {
            assert_ne!(c.first.label, c.second.label);
            for original in [c.first, c.second] {
                let prefix = &rows[original.row].0[..=original.position];
                assert_eq!(
                    reference::state(input.programs[p], &input.alphabet, prefix, &mut w)?,
                    c.state
                );
                assert_eq!(
                    rows[original.row].1[original.position] as usize,
                    original.label
                );
            }
        }
    }
    let compiled = model::compile(&input, &mut w)?.unwrap();
    assert_eq!(compiled.records.len(), 130);
    assert_eq!(compiled.domain.shape.programs, 4);
    assert_eq!(compiled.domain.shape.states, 512);
    assert_eq!(compiled.domain.unknown.len(), 4 * 510);
    for i in 0..130 {
        assert_eq!(compiled.records.label(i)?, input.observations()[i].label);
    }
    Ok(())
}
#[test]
fn alphabet_uses_input_tokens_and_is_invariant_to_answer_permutation() -> Result<()> {
    let first = vec![(vec![1, 3, 5, 18, 61], vec![-100, -100, -100, 61, 17])];
    let second = vec![(vec![1, 3, 5, 18, 62], vec![-100, -100, -100, 62, 19])];
    let mut w = work();
    let a = Prepared::new(training(&first, 70), CALLER, PROCESS, &mut w)?;
    let b = Prepared::new(training(&second, 70), CALLER, PROCESS, &mut w)?;
    assert_eq!(a.alphabet.symbols, vec![3, 5]);
    assert_eq!(a.alphabet, b.alphabet);
    assert_eq!(a.programs, b.programs);
    assert_eq!(a.partition.retained, b.partition.retained);
    Ok(())
}
#[test]
fn conflicting_duplicate_prefix_refuses_the_whole_family() -> Result<()> {
    let rows = vec![(vec![2, 3], vec![-100, 6]), (vec![2, 3], vec![-100, 7])];
    let mut w = work();
    let p = Prepared::new(training(&rows, 8), CALLER, PROCESS, &mut w)?;
    assert!(p.partition.retained.is_empty());
    assert!(model::compile(&p, &mut w)?.is_none());
    assert_eq!(p.partition.decisions.len(), 176);
    Ok(())
}
#[test]
fn actual_modes_keep_tail_and_all_complex_coordinates() -> Result<()> {
    let rows = ordered(130);
    let mut w = work();
    let p = Prepared::new(training(&rows, 8), CALLER, PROCESS, &mut w)?;
    let c = model::compile(&p, &mut w)?.unwrap();
    let mut first = None;
    for mode in [Mode::Forward, Mode::Mini] {
        let t = c.train(mode, &mut w)?;
        assert!(!t.target_realizability_certified && !t.exact_real_projection_certified);
        assert_eq!(t.audit.logical_coefficients, 4 * 512 * 8 * 2);
        assert_eq!(t.parameters.values().count(), 4 + 4 * 510 * 8);
        assert!(t.audit.maximum_inverse_residual <= 1e-7);
        assert!(
            t.audit
                .column_l1
                .iter()
                .all(|&x| x <= 2. * 64. * 512. + 1e-7)
        );
        if mode == Mode::Forward {
            assert!(t.steps.is_empty());
            assert_eq!(t.initial_ce.to_bits(), t.final_ce.to_bits());
            first = Some(t.final_ce);
        } else {
            assert_eq!(t.steps.len(), 2);
            assert_eq!(
                (t.steps[0].first_record, t.steps[0].end_record_exclusive),
                (0, 127)
            );
            assert_eq!(
                (t.steps[1].first_record, t.steps[1].end_record_exclusive),
                (127, 130)
            );
            assert!(t.steps.iter().any(|s| s.parameter_change_l2 > 0.));
            assert!(t.final_ce < first.unwrap());
        }
        let mut forward = model::Forward::new(&p, &c.domain, &t.head, &mut w)?;
        for prefix in [vec![1, 4], vec![4, 1]] {
            let expected = if prefix[0] < prefix[1] { 6 } else { 7 };
            let logits = forward.logits(&prefix).map_err(|error| error.to_string())?;
            assert_eq!(
                forward.coverage(&prefix)?,
                model::Coverage::Agreement(expected)
            );
            assert!(
                logits
                    .iter()
                    .enumerate()
                    .all(|(i, &x)| i == expected || x < logits[expected])
            );
        }
        assert_eq!(forward.coverage(&[3, 3])?, model::Coverage::Unseen);
        let mut stream = vec![];
        let digest = joint::persistence::write(&c.domain, &t.parameters, &mut stream, &mut w)?;
        let (d, q) =
            joint::persistence::read(&stream[..], digest, stream.len() as u64, PROCESS, &mut w)?;
        assert_eq!(d, c.domain);
        assert!(
            q.values()
                .zip(t.parameters.values())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
    assert!(model::policy(Mode::Full).is_err());
    Ok(())
}
#[test]
fn unknown_padding_rows_admit_arbitrary_nonuniform_completions() -> Result<()> {
    let rows = ordered(12);
    let mut w = work();
    let p = Prepared::new(training(&rows, 8), CALLER, PROCESS, &mut w)?;
    let c = model::compile(&p, &mut w)?.unwrap();
    let initial = joint::Parameters::initial(&c.domain, 8., PROCESS, &mut w)?;
    let mut tensor = reference_model::Parameters {
        values: initial.values().copied().collect(),
    };
    let at = c.domain.offsets[511];
    assert_ne!(at, usize::MAX);
    tensor.values[at..at + 8].fill(0.);
    tensor.values[at + 5] = 2.;
    let changed = joint::Parameters::from_legacy(&c.domain, &tensor, PROCESS, &mut w)?;
    changed.check(&c.domain, 1e-8, &mut w)?;
    let a = joint::head::Head::build(&c.domain, &initial, PROCESS, &mut w)?;
    let b = joint::head::Head::build(&c.domain, &changed, PROCESS, &mut w)?;
    b.audit(&c.domain, &changed, &mut w)?;
    assert_ne!(
        a.logits(&[511, 511, 511, 511], &mut w)?,
        b.logits(&[511, 511, 511, 511], &mut w)?
    );
    assert!(b.coefficient(0, 509, 5)?.1.abs() > 0.);
    Ok(())
}
#[test]
fn bounds_reject_missing_reservations_invalid_inputs_and_exhausted_work() {
    let rows = ordered(12);
    assert!(Prepared::new(training(&rows, 8), 0, PROCESS, &mut work()).is_err());
    assert!(Prepared::new(training(&rows, 8), PROCESS as u128, PROCESS, &mut work()).is_err());
    let mut invalid = training(&rows, 8);
    invalid.rows[1].original_index = 0;
    assert!(Prepared::new(invalid, CALLER, PROCESS, &mut work()).is_err());
    let invalid = vec![(vec![8], vec![1])];
    assert!(Prepared::new(training(&invalid, 8), CALLER, PROCESS, &mut work()).is_err());
    assert!(
        Prepared::new(
            training(&rows, 8),
            CALLER,
            PROCESS,
            &mut Work::new(1).unwrap()
        )
        .is_err()
    );
    assert!(features::validate_prefix(&[], 8).is_err());
    assert!(features::validate_prefix(&[1, -1], 8).is_err());
}
