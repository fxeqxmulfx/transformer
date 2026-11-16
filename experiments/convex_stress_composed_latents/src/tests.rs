//! Boundary controls ported from e5ef378a; grammar-specific checks are new.
use super::*;
use family::{Evaluator, Program, programs};
mod family_controls;

fn work() -> Work {
    Work::new(MAX_WORK).unwrap()
}
fn observed() -> Training {
    Training {
        vocabulary: 3,
        rows: vec![
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 2],
            [1, 1, 0],
            [2, 0, 1],
            [0, 0, 0],
        ],
    }
}

#[test]
fn full_partition_matches_independent_pairwise_collisions() {
    let data = observed();
    let selected = selection::select(&data, &mut work()).unwrap();
    let evaluator = Evaluator::new(data.vocabulary).unwrap();
    let bank = programs();
    let mut surviving = vec![false; PROGRAMS];
    for &i in &selected.retained {
        surviving[i] = true;
    }
    assert_eq!(selected.retained.len() + selected.conflicts.len(), PROGRAMS);
    for (id, &tree) in bank.iter().enumerate() {
        let states: Vec<_> = data
            .rows
            .iter()
            .map(|x| evaluator.reference(tree, x[0] as usize, x[1] as usize))
            .collect();
        let compatible = (0..data.rows.len())
            .all(|i| (0..i).all(|j| states[i] != states[j] || data.rows[i][2] == data.rows[j][2]));
        assert_eq!(surviving[id], compatible);
    }
    for witness in &selected.conflicts {
        let a = data.rows[witness.first_record];
        let b = data.rows[witness.second_record];
        assert_ne!(a[2], b[2]);
        assert_eq!(
            evaluator.reference(bank[witness.program], a[0] as usize, a[1] as usize),
            witness.state
        );
        assert_eq!(
            evaluator.reference(bank[witness.program], b[0] as usize, b[1] as usize),
            witness.state
        );
    }
    assert_eq!(
        selected.audit.independently_checked_conflicts,
        selected.conflicts.len()
    );
    assert_eq!(
        selected.audit.independently_checked_states,
        (selected.retained.len() * data.rows.len()) as u64
    );
}

#[test]
fn raw_label_renaming_and_duplicate_records_preserve_the_complete_support() {
    let data = observed();
    let a = selection::select(&data, &mut work()).unwrap();
    let mut renamed = data.clone();
    for row in &mut renamed.rows {
        row[2] = (row[2] + 1) % 3;
    }
    let b = selection::select(&renamed, &mut work()).unwrap();
    assert_eq!(a.retained, b.retained);
    assert_eq!(a.conflicts, b.conflicts);
    assert_eq!(
        a.audit.program_major_states_sha256,
        b.audit.program_major_states_sha256
    );
    assert_ne!(
        a.original_ordered_training_sha256,
        b.original_ordered_training_sha256
    );
    let mut duplicated = data.clone();
    duplicated.rows.push(duplicated.rows[1]);
    let c = selection::select(&duplicated, &mut work()).unwrap();
    assert_eq!(a.retained, c.retained);
    assert_eq!(a.conflicts, c.conflicts);
    assert_eq!(c.records, data.rows.len() + 1);
    assert_ne!(
        a.original_ordered_training_sha256,
        c.original_ordered_training_sha256
    );
    assert_eq!(a.audit.dictionary_sha256, c.audit.dictionary_sha256);
    assert_eq!(
        c.audit.independently_checked_states,
        c.retained.len() as u64 * c.records as u64
    );
}

#[test]
fn all_compatible_candidates_survive_and_forged_subsets_are_rejected() {
    let data = Training {
        vocabulary: 3,
        rows: vec![[0, 0, 2], [1, 2, 2]],
    };
    let a = selection::select(&data, &mut work()).unwrap();
    assert_eq!(a.retained, (0..PROGRAMS).collect::<Vec<_>>());
    assert!(a.conflicts.is_empty());
    let mut missing = a.clone();
    missing.retained.pop();
    assert!(encoding::encode(&data, &missing, &mut work()).is_err());
    assert!(encoding::encode_packed(&data, &missing, &mut work()).is_err());
    let mut bad = a.clone();
    bad.audit.unknown_rows -= 1;
    assert!(encoding::encode(&data, &bad, &mut work()).is_err());
    assert!(encoding::encode_packed(&data, &bad, &mut work()).is_err());
    let encoded = serde_json::to_vec(&a).unwrap();
    let loaded: selection::Selection = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(loaded, a);
    let mut conflict = data;
    conflict.rows.push([0, 0, 1]);
    let refused = selection::select(&conflict, &mut work()).unwrap();
    assert!(refused.retained.is_empty());
    assert_eq!(refused.conflicts.len(), PROGRAMS);
    assert!(
        refused
            .audit
            .admission_refusal
            .unwrap()
            .contains("empty compatible support")
    );
}

#[test]
fn streamed_audit_matches_the_complete_materialized_domain_and_actual_training() {
    let data = Training {
        vocabulary: 3,
        rows: (0..3)
            .flat_map(|a| (0..3).map(move |b| [a, b, u32::from((a + b + a * b) % 3 != 0)]))
            .collect(),
    };
    let selected = selection::select(&data, &mut work()).unwrap();
    assert!(!selected.retained.is_empty());
    assert!(selected.retained.iter().any(|&i| i >= pairs::PROGRAMS));
    let (domain, records) = encoding::encode(&data, &selected, &mut work()).unwrap();
    let (packed_domain, packed) = encoding::encode_packed(&data, &selected, &mut work()).unwrap();
    assert_eq!(domain, packed_domain);
    let source = joint::records::Source::Packed(&packed);
    let mut index = 0;
    source
        .visit(0..records.len(), &mut work(), |row, _| {
            assert_eq!(row, &records[index]);
            index += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(index, records.len());
    assert_eq!(records.len(), data.rows.len());
    assert_eq!(domain.unknown.len(), selected.audit.unknown_rows);
    assert!(domain.unknown.len() >= domain.shape.programs); // The padded state remains free.
    use joint::training::Mode;
    for mode in [Mode::Forward, Mode::Full, Mode::Mini] {
        let mut config = policy(mode);
        config.batch_size = 4;
        config.learning_rate = 3.;
        let fitted =
            joint::training::train(&domain, &records, config.clone(), &mut work()).unwrap();
        let packed_fit =
            joint::training::train_source(&packed_domain, source, config, &mut work()).unwrap();
        assert_eq!(
            serde_json::to_vec(&fitted.steps).unwrap(),
            serde_json::to_vec(&packed_fit.steps).unwrap()
        );
        assert_eq!(fitted.initial_ce.to_bits(), packed_fit.initial_ce.to_bits());
        assert_eq!(fitted.final_ce.to_bits(), packed_fit.final_ce.to_bits());
        assert_eq!(
            fitted
                .parameters
                .values
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            packed_fit
                .parameters
                .values
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(fitted.audit, packed_fit.audit);
        for p in 0..domain.shape.programs {
            for s in 0..domain.shape.states {
                for c in 0..domain.shape.classes {
                    let (a, b) = fitted.head.coefficient(p, s, c).unwrap();
                    let (x, y) = packed_fit.head.coefficient(p, s, c).unwrap();
                    assert_eq!((a.to_bits(), b.to_bits()), (x.to_bits(), y.to_bits()));
                }
            }
        }
        assert_eq!(fitted.audit.logical_coefficients, domain.shape.cells() * 2);
        assert!(!fitted.target_realizability_certified);
        if mode == Mode::Forward {
            assert!(fitted.steps.is_empty());
        } else {
            assert!(fitted.final_ce < fitted.initial_ce);
        }
        if mode == Mode::Mini {
            assert_eq!(
                fitted
                    .steps
                    .iter()
                    .map(|s| (s.first_record, s.end_record_exclusive))
                    .collect::<Vec<_>>(),
                vec![(0, 4), (4, 8), (8, 9)]
            );
        }
    }
    let mut changed = data.clone();
    changed.rows.reverse();
    assert!(encoding::encode(&changed, &selected, &mut work()).is_err());
    assert!(encoding::encode_packed(&changed, &selected, &mut work()).is_err());
}

#[test]
fn impossible_model_buffers_are_refused_after_a_bounded_streamed_audit() {
    let data = Training {
        vocabulary: 1021,
        rows: vec![[0, 0, 0]],
    };
    let selected = selection::select(&data, &mut work()).unwrap();
    assert_eq!(selected.retained.len(), PROGRAMS);
    assert_eq!(selected.audit.known_rows, PROGRAMS);
    assert_eq!(selected.audit.unknown_rows, PROGRAMS * 1023);
    assert!(selected.audit.complete_buffer_forecast.is_none());
    assert!(selected.audit.admission_refusal.unwrap().contains("96 GB"));
}

#[test]
fn work_and_original_input_admission_do_not_silently_truncate_the_bank() {
    assert!(selection::select(&observed(), &mut Work::new(1).unwrap()).is_err());
    assert!(Evaluator::new(4).is_err());
    assert!(Evaluator::new(1025).is_err());
    let mut bad = observed();
    bad.rows[0][0] = 3;
    assert!(selection::select(&bad, &mut work()).is_err());
    assert!(
        serde_json::from_str::<Training>(r#"{"vocabulary":3,"rows":[[0,0,0]],"target":"hidden"}"#)
            .is_err()
    );
    let protocol: serde_json::Value = serde_json::from_str(PROTOCOL).unwrap();
    assert_eq!(protocol["fixed_family"]["total"], PROGRAMS);
    assert_eq!(protocol["process_cap_bytes"], joint::PROCESS_BYTES);
    assert_eq!(
        protocol["buffer_cap_bytes"].as_u64().unwrap() as u128,
        joint::MAX_BYTES
    );
}
