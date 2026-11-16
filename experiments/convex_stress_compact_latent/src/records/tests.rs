use super::*;
const LIMIT: u64 = 16 * 1024 * 1024;
fn work() -> Work {
    Work::new(2_000_000_000_000).unwrap()
}
#[test]
fn original_words_callback_order_duplicates_and_tail_visits_match() {
    let sh = Shape {
        programs: 2,
        states: 4096,
        classes: 1024,
    };
    let labels: Vec<_> = (0..130).map(|r| r % 2 * 1023).collect();
    let (mut a, mut b) = (work(), work());
    let (mut ca, mut cb) = (Vec::new(), Vec::new());
    let old = legacy::records::Packed::from_columns(
        sh,
        &labels,
        |r, p, _| {
            ca.push((r, p));
            Ok((r % 2 + p) % 2 * 4095)
        },
        &mut a,
    )
    .unwrap();
    let new = Packed::from_columns(
        sh,
        &labels,
        |r, p, _| {
            cb.push((r, p));
            Ok((r % 2 + p) % 2 * 4095)
        },
        LIMIT,
        &mut b,
    )
    .unwrap();
    assert_eq!(ca, cb);
    assert_eq!(a.used, b.used);
    assert_eq!(new.len(), 130);
    for r in 0..130 {
        assert_eq!(old.label(r).unwrap(), new.label(r).unwrap());
        for p in 0..2 {
            assert_eq!(old.state(r, p).unwrap(), new.state(r, p).unwrap());
        }
    }
    for range in [0..127, 127..130, 129..130, 7..7] {
        let (mut x, mut y) = (Vec::new(), Vec::new());
        legacy::records::Source::Packed(&old)
            .visit(range.clone(), &mut a, |r, _| {
                x.push(r.clone());
                Ok(())
            })
            .unwrap();
        Source::Packed(&new)
            .visit(range, &mut b, |r, _| {
                y.push(r.clone());
                Ok(())
            })
            .unwrap();
        assert_eq!(x, y);
        assert_eq!(a.used, b.used);
    }
    assert_eq!(new.allocation_bytes(), packed_bytes(sh, 130).unwrap());
    assert!(new.state(130, 0).is_err());
    assert!(new.state(0, 2).is_err());
    assert!(new.label(130).is_err());
}
#[test]
fn capacity_preflight_precedes_callback_and_keeps_input_and_scratch_separate() {
    let sh = Shape {
        programs: 3,
        states: 4,
        classes: 5,
    };
    let labels = [2, 2, 2];
    let bytes = packed_bytes(sh, labels.len()).unwrap() as u64;
    let mut calls = 0;
    let mut w = work();
    assert!(
        Packed::from_columns(
            sh,
            &labels,
            |_, _, _| {
                calls += 1;
                Ok(0)
            },
            bytes,
            &mut w
        )
        .is_err()
    );
    assert_eq!((calls, w.used), (0, 0));
    let p = Packed::from_columns(sh, &labels, |_, _, _| Ok(3), bytes + 1, &mut w).unwrap();
    assert_eq!(p.allocation_bytes(), bytes as u128);
    assert_eq!(p.scratch_bytes(), size_of::<Record>() as u128 + 3 * 8);
    let rows = vec![
        Record {
            states: vec![3; 3],
            label: 2
        };
        3
    ];
    assert!(Packed::from_records(sh, &rows, bytes + 1, &mut work()).is_err());
    let q = Packed::from_records(sh, &rows, LIMIT, &mut work()).unwrap();
    assert_eq!(p.states, q.states);
    assert_eq!(p.labels, q.labels);
    assert_eq!(Source::Expanded(&rows).scratch_bytes(), 0);
    assert!(
        Source::Packed(&q)
            .visit(0..4, &mut work(), |_, _| Ok(()))
            .is_err()
    );
    assert!(
        Source::Packed(&q)
            .check_shape(Shape { classes: 4, ..sh })
            .is_err()
    );
}
#[test]
fn unsafe_counts_and_values_are_rejected_without_truncating_words() {
    let sh = Shape {
        programs: 1,
        states: 4096,
        classes: 1024,
    };
    assert!(packed_bytes(sh, 0).is_err());
    assert!(packed_bytes(sh, 65537).is_err());
    assert!(
        packed_bytes(
            Shape {
                programs: usize::MAX,
                ..sh
            },
            1
        )
        .is_err()
    );
    assert!(Packed::from_columns(sh, &[1024], |_, _, _| Ok(0), LIMIT, &mut work()).is_err());
    assert!(Packed::from_columns(sh, &[0], |_, _, _| Ok(4096), LIMIT, &mut work()).is_err());
    let bad = [Record {
        states: vec![],
        label: 0,
    }];
    assert!(Packed::from_records(sh, &bad, LIMIT, &mut work()).is_err());
    let mut packed = Packed::from_columns(sh, &[0], |_, _, _| Ok(0), LIMIT, &mut work()).unwrap();
    packed.states.clear();
    assert!(Source::Packed(&packed).check_shape(sh).is_err());
}
