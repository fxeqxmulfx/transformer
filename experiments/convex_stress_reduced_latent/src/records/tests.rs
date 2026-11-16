use super::*;
use crate::{
    head::Head,
    training::{self, Config, Mode},
};
const BUDGET: u64 = 40_000_000_000_000;
fn records() -> (Shape, Vec<Record>) {
    let shape = Shape {
        programs: 3,
        states: 8,
        classes: 3,
    };
    let rows = (0..130)
        .map(|i| {
            let s = i % 3;
            Record {
                states: vec![s, s + 3, (2 * s) % 8],
                label: s,
            }
        })
        .collect();
    (shape, rows)
}
#[test]
fn every_word_and_original_index_survives_and_only_one_row_is_decoded() {
    let (shape, expanded) = records();
    let mut work = Work::new(BUDGET).unwrap();
    let packed = Packed::from_records(shape, &expanded, &mut work).unwrap();
    assert_eq!(packed.len(), 130);
    assert_eq!(packed.stored_bytes(), 2 * 130 * 4);
    assert_eq!(packed.scratch_bytes(), 24);
    let mut first_pointer = None;
    let mut index = 0;
    Source::Packed(&packed)
        .visit(0..130, &mut work, |row, _| {
            assert_eq!(row, &expanded[index]);
            if let Some(pointer) = first_pointer {
                assert_eq!(pointer, row.states.as_ptr());
            }
            first_pointer = Some(row.states.as_ptr());
            for p in 0..shape.programs {
                assert_eq!(packed.state(index, p).unwrap(), row.states[p]);
            }
            assert_eq!(packed.label(index).unwrap(), row.label);
            index += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(index, 130);
    let mut tail = vec![];
    Source::Packed(&packed)
        .visit(127..130, &mut work, |r, _| {
            tail.push(r.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(tail, expanded[127..130]);
    assert!(packed.state(130, 0).is_err() && packed.state(0, 3).is_err());
    assert!(packed.label(130).is_err());
    assert!(
        Source::Packed(&packed)
            .visit(0..131, &mut work, |_, _| Ok(()))
            .is_err()
    );
}
#[test]
fn domain_heads_gradients_and_all_training_modes_match_expanded_bits() {
    let (shape, expanded) = records();
    let mut work = Work::new(BUDGET).unwrap();
    let packed = Packed::from_records(shape, &expanded, &mut work).unwrap();
    let domain = Domain::new(shape, &expanded, 1., 32., &mut work).unwrap();
    let again = Domain::from_source(shape, Source::Packed(&packed), 1., 32., &mut work).unwrap();
    assert_eq!(domain, again);
    assert!(!domain.unknown.is_empty());
    let initial = domain.initial(8., &mut work).unwrap();
    let head = Head::build(&domain, &initial, &mut work).unwrap();
    for range in [0..130, 0..127, 127..130] {
        let a = training::gradient(&domain, &head, &expanded[range.clone()], &mut work).unwrap();
        let b =
            training::gradient_source(&domain, &head, Source::Packed(&packed), range, &mut work)
                .unwrap();
        assert_eq!(a.0.to_bits(), b.0.to_bits());
        assert_eq!(
            a.1.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.1.values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    for mode in [Mode::Forward, Mode::Full, Mode::Mini] {
        let config = Config {
            mode,
            epochs: match mode {
                Mode::Forward => 0,
                Mode::Full => 4,
                Mode::Mini => 1,
            },
            batch_size: 127,
            learning_rate: 1_000_000.,
            initial_mass: 8.,
        };
        let a = training::train(&domain, &expanded, config.clone(), &mut work).unwrap();
        let b = training::train_source(&again, Source::Packed(&packed), config, &mut work).unwrap();
        assert_eq!(
            serde_json::to_vec(&a.steps).unwrap(),
            serde_json::to_vec(&b.steps).unwrap()
        );
        assert_eq!(a.initial_ce.to_bits(), b.initial_ce.to_bits());
        assert_eq!(a.final_ce.to_bits(), b.final_ce.to_bits());
        assert_eq!(a.audit, b.audit);
        assert_eq!(a.head, b.head);
        for p in 0..shape.programs {
            for s in 0..shape.states {
                for c in 0..shape.classes {
                    let (x, y) = a.head.coefficient(p, s, c).unwrap();
                    let (u, v) = b.head.coefficient(p, s, c).unwrap();
                    assert_eq!((x.to_bits(), y.to_bits()), (u.to_bits(), v.to_bits()));
                }
            }
        }
        assert_eq!(
            a.parameters
                .values
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            b.parameters
                .values
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(b.operations > a.operations);
        let (mut expanded_checkpoint, mut packed_checkpoint) = (vec![], vec![]);
        let expanded_hash =
            crate::persistence::write(&domain, &a.parameters, &mut expanded_checkpoint, &mut work)
                .unwrap();
        let packed_hash =
            crate::persistence::write(&again, &b.parameters, &mut packed_checkpoint, &mut work)
                .unwrap();
        assert_eq!(expanded_hash, packed_hash);
        assert_eq!(expanded_checkpoint, packed_checkpoint);
        if mode == Mode::Mini {
            assert_eq!(b.steps[1].first_record, 127);
            assert_eq!(b.steps[1].end_record_exclusive, 130);
        }
    }
}
#[test]
fn range_checks_prevent_truncation_and_admission_charges_all_unknown_coordinates() {
    let mut work = Work::new(BUDGET).unwrap();
    let shape = Shape {
        programs: 2,
        states: 4096,
        classes: 1024,
    };
    let rows = vec![Record {
        states: vec![4095, 0],
        label: 1023,
    }];
    let packed = Packed::from_records(shape, &rows, &mut work).unwrap();
    assert_eq!(packed.state(0, 0).unwrap(), 4095);
    assert_eq!(packed.label(0).unwrap(), 1023);
    assert!(Packed::from_columns(shape, &[1024], |_, _, _| Ok(0), &mut work).is_err());
    for invalid in [4096, 65536, usize::MAX] {
        assert!(Packed::from_columns(shape, &[0], |_, _, _| Ok(invalid), &mut work).is_err());
    }
    let mut exhausted = Work::new(1).unwrap();
    assert!(Packed::from_records(shape, &rows, &mut exhausted).is_err());
    assert!(
        Source::Packed(&packed)
            .check_shape(Shape { states: 4, ..shape })
            .is_err()
    );
    let large = Shape {
        programs: 299551,
        states: 196,
        classes: 193,
    };
    assert!(large.admit(18528, 1126298, 137825884).is_err());
    assert_eq!(
        large.admit_packed(18528, 1126298, 137825884).unwrap(),
        89281189304
    );
    assert!(
        large
            .admit_packed(18528, large.rows(), large.cells() / 4)
            .is_err()
    );
}
