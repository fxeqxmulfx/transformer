use crate::rows::{Builder, Rows};

const LIMIT: u64 = 8 * 1024 * 1024;
fn bits(values: impl IntoIterator<Item = f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}

#[test]
fn mixed_rows_preserve_every_word_and_every_original_row() {
    let nan = f64::from_bits(0x7ff8_0000_0000_0007);
    let values = [
        0.1,
        0.1,
        0.1,
        0.1,
        -0.,
        0.,
        f64::from_bits(1),
        f64::from_bits(0x8000_0000_0000_0001),
        nan,
        nan,
        nan,
        nan,
        0.1,
        0.1,
        0.1,
        0.1,
    ];
    let table = Rows::from_dense(4, &values, LIMIT).unwrap();
    assert_eq!(table.rows(), 4);
    assert_eq!(table.classes(), 4);
    assert_eq!(table.len(), 16);
    assert_eq!(table.dense_rows(), 1);
    assert_eq!(table.stored_values(), 7);
    assert_eq!(bits(table.iter().copied()), bits(values));
    assert_eq!(
        bits(table.row(1).unwrap().copied()),
        bits(values[4..8].iter().copied())
    );
    assert_eq!(
        bits(table.try_clone(LIMIT).unwrap().iter().copied()),
        bits(values)
    );
    assert!(table.try_clone(1).is_err());
    assert!(table.get(4, 0).is_none());
    assert!(table.get(0, 4).is_none());
    assert!(table.row(4).is_none());
    assert_eq!(table.iter().len(), values.len());
}

#[test]
fn nonuniform_fallback_keeps_all_classes_and_shared_rows_can_become_nonuniform() {
    let first = Rows::from_dense(5, &[2.; 15], LIMIT).unwrap();
    let other = Rows::from_dense(5, &(0..15).map(|n| n as f64).collect::<Vec<_>>(), LIMIT).unwrap();
    assert_eq!(other.dense_rows(), 3);
    let mut count = 0;
    let result = first
        .zip_map(&other, LIMIT, |a, b| {
            count += 1;
            a - b + count as f64
        })
        .unwrap();
    assert_eq!(count, 15);
    assert_eq!(bits(result.iter().copied()), bits(vec![3.; 15]));
    assert_eq!(result.dense_rows(), 0);
    let result = first.zip_map(&other, LIMIT, |a, b| a - 0.125 * b).unwrap();
    assert_eq!(result.dense_rows(), 3);
    assert_eq!(
        bits(result.iter().copied()),
        bits((0..15).map(|n| 2. - 0.125 * n as f64))
    );
    assert!(
        first
            .zip_map(
                &Rows::from_dense(3, &[0.; 15], LIMIT).unwrap(),
                LIMIT,
                |a, b| a + b
            )
            .is_err()
    );
}

#[test]
fn reductions_visit_every_logical_entry_in_the_original_order() {
    let original = vec![0.1; 1000];
    let table = Rows::from_dense(1000, &original, LIMIT).unwrap();
    assert_eq!(table.stored_values(), 1);
    let expected: f64 = original.iter().sum();
    let actual: f64 = table.iter().sum();
    assert_eq!(expected.to_bits(), actual.to_bits());
    assert_ne!(actual.to_bits(), (1000. * original[0]).to_bits());
    let mut iter = table.iter();
    for remaining in (0..1000).rev() {
        assert_eq!(iter.next().unwrap().to_bits(), original[0].to_bits());
        assert_eq!(iter.len(), remaining);
    }
    assert!(iter.next().is_none());
    assert!(iter.next().is_none());
}

#[test]
fn complete_shape_and_storage_failures_are_checked_before_growth() {
    assert!(Builder::new(0, 1, LIMIT).is_err());
    assert!(Builder::new(1025, 1, LIMIT).is_err());
    assert!(Builder::new(2, usize::MAX, LIMIT).is_err());
    assert!(Builder::new(2, 1, 0).is_err());
    assert!(Builder::new(2, 1, 96_000_000_001).is_err());
    assert!(Rows::from_dense(3, &[1., 2.], LIMIT).is_err());
    let empty = Builder::new(3, 0, LIMIT).unwrap().finish().unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.iter().len(), 0);
    assert!(Builder::new(3, 2, LIMIT).unwrap().finish().is_err());
    let mut builder = Builder::new(3, 1, LIMIT).unwrap();
    assert!(builder.push(&[1., 2.]).is_err());
    builder.push_repeated(-0.).unwrap();
    assert!(builder.push_repeated(0.).is_err());
    assert_eq!(
        bits(builder.finish().unwrap().iter().copied()),
        bits([-0.; 3])
    );
    // Metadata and one scratch row fit; an explicit row needs additional bytes.
    let floor = (1..1024).find(|&n| Builder::new(3, 1, n).is_ok()).unwrap();
    let mut tight = Builder::new(3, 1, floor).unwrap();
    assert!(tight.push(&[1., 2., 3.]).is_err());
    tight.push_repeated(7.).unwrap();
    let result = tight.finish().unwrap();
    assert!(result.allocation_bytes() <= floor);
    assert_eq!(bits(result.iter().copied()), bits([7.; 3]));
}

#[test]
fn growth_reserves_old_and_new_buffers_and_keeps_prior_rows_after_refusal() {
    let floor = (1..1024).find(|&n| Builder::new(3, 2, n).is_ok()).unwrap();
    let mut builder = Builder::new(3, 2, floor + 48).unwrap();
    builder.push(&[1., 2., 3.]).unwrap();
    // Six explicit values fit after growth, but moving the allocation needs
    // the old three values and the new six simultaneously:72 bytes.
    assert!(builder.push(&[4., 5., 6.]).is_err());
    builder.push_repeated(7.).unwrap();
    let table = builder.finish().unwrap();
    assert_eq!(bits(table.iter().copied()), bits([1., 2., 3., 7., 7., 7.]));
    assert!(table.allocation_bytes() <= floor + 48);
}

#[test]
fn actual_full_unknown_decoder_and_complex_head_survive_row_roundtrip() {
    use legacy::{Parameters, Record, Shape, Work, domain::Domain, head::Head};
    let shape = Shape {
        programs: 2,
        states: 4,
        classes: 5,
    };
    let mut work = Work::new(100_000_000).unwrap();
    let domain = Domain::new(
        shape,
        &[
            Record {
                states: vec![0, 0],
                label: 0,
            },
            Record {
                states: vec![1, 2],
                label: 1,
            },
            Record {
                states: vec![0, 0],
                label: 0,
            },
        ],
        1.,
        8.,
        &mut work,
    )
    .unwrap();
    let mut q = domain.initial(4., &mut work).unwrap();
    let at = domain.offsets[3];
    q.values[at..at + 5].copy_from_slice(&[1., 1., -0., 0., f64::from_bits(1)]);
    domain.check(&q, 1e-8, &mut work).unwrap();
    let stored = Rows::from_dense(shape.classes, &q.values[shape.programs..], LIMIT).unwrap();
    assert_eq!(stored.rows(), domain.unknown.len());
    assert_eq!(stored.dense_rows(), 1);
    let mut restored = q.values[..shape.programs].to_vec();
    restored.extend(stored.iter().copied());
    let rebuilt = Parameters { values: restored };
    assert_eq!(
        bits(q.values.iter().copied()),
        bits(rebuilt.values.iter().copied())
    );
    let first = Head::build(&domain, &q, &mut work).unwrap();
    let second = Head::build(&domain, &rebuilt, &mut work).unwrap();
    let mut nonzero_imaginary = false;
    for p in 0..shape.programs {
        for f in 0..shape.states {
            for c in 0..shape.classes {
                let a = first.coefficient(p, f, c).unwrap();
                let b = second.coefficient(p, f, c).unwrap();
                assert_eq!(
                    (a.0.to_bits(), a.1.to_bits()),
                    (b.0.to_bits(), b.1.to_bits())
                );
                nonzero_imaginary |= a.1 != 0.;
            }
        }
    }
    assert!(nonzero_imaginary);
    for a in 0..4 {
        for b in 0..4 {
            assert_eq!(
                bits(first.logits(&[a, b], &mut work).unwrap()),
                bits(second.logits(&[a, b], &mut work).unwrap())
            );
        }
    }
}
