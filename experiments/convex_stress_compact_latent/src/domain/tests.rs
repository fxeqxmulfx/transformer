use super::*;
use crate::records::Packed;
const LIMIT: u64 = 64 * 1024 * 1024;
fn work() -> Work {
    Work::new(2_000_000_000_000).unwrap()
}
fn fixture() -> Domain {
    from_labels(
        Shape {
            programs: 2,
            states: 8,
            classes: 5,
        },
        vec![
            Some(0),
            None,
            Some(2),
            None,
            Some(1),
            None,
            None,
            None,
            None,
            Some(1),
            None,
            Some(2),
            None,
            None,
            Some(4),
            None,
        ],
        1.,
        8.,
        LIMIT,
        &mut work(),
    )
    .unwrap()
}
#[test]
fn complete_dictionary_and_work_match_the_original_with_spare_label_capacity() {
    let d = fixture();
    let mut labels = Vec::with_capacity(128);
    labels.extend_from_slice(&d.labels);
    let (mut a, mut b) = (work(), work());
    let old = Domain::from_labels(d.shape, labels.clone(), 1., 8., &mut a).unwrap();
    let new = from_labels(d.shape, labels, 1., 8., LIMIT, &mut b).unwrap();
    assert_eq!(old, new);
    assert_eq!(a.used, b.used);
    assert_eq!(new.labels.capacity(), 128);
    assert_eq!(
        allocation_bytes(&new),
        dictionary_bytes(d.shape, 128, d.unknown.len()).unwrap()
    );
    old.validate(&mut a).unwrap();
    validate(&new, &mut b).unwrap();
    assert_eq!(a.used, b.used);
    let bytes = dictionary_bytes(d.shape, d.shape.rows(), d.unknown.len()).unwrap() as u64;
    assert!(from_labels(d.shape, d.labels.clone(), 1., 8., bytes, &mut work()).is_err());
    assert_eq!(
        d,
        from_labels(d.shape, d.labels.clone(), 1., 8., bytes + 1, &mut work()).unwrap()
    );
}
#[test]
fn packed_and_expanded_original_records_produce_identical_dictionaries_and_work() {
    let sh = Shape {
        programs: 2,
        states: 8,
        classes: 5,
    };
    let rows: Vec<_> = (0..130)
        .map(|i| Record {
            states: vec![i % 4, (i % 4 + 1) % 8],
            label: i % 4,
        })
        .collect();
    let old_packed = legacy::records::Packed::from_records(sh, &rows, &mut work()).unwrap();
    let packed = Packed::from_records(sh, &rows, LIMIT, &mut work()).unwrap();
    for (old_source, new_source) in [
        (
            legacy::records::Source::Expanded(&rows),
            Source::Expanded(&rows),
        ),
        (
            legacy::records::Source::Packed(&old_packed),
            Source::Packed(&packed),
        ),
    ] {
        let (mut a, mut b) = (work(), work());
        let old = Domain::from_source(sh, old_source, 1., 8., &mut a).unwrap();
        let new = from_source(sh, new_source, 1., 8., LIMIT, &mut b).unwrap();
        assert_eq!(old, new);
        assert_eq!(a.used, b.used);
    }
    let mut conflicting = rows.clone();
    conflicting[129].label = 4;
    assert!(
        from_source(
            sh,
            Source::Expanded(&conflicting),
            1.,
            8.,
            LIMIT,
            &mut work()
        )
        .is_err()
    );
    assert!(from_source(sh, Source::Expanded(&[]), 1., 8., LIMIT, &mut work()).is_err());
}
#[test]
fn malformed_public_dictionary_fields_fail_canonical_validation_before_indexing() {
    let d = fixture();
    let changes: [fn(&mut Domain); 13] = [
        |d| {
            d.labels.pop();
        },
        |d| {
            d.offsets.pop();
        },
        |d| {
            d.known.pop();
        },
        |d| d.labels[0] = Some(d.shape.classes),
        |d| d.offsets[0] = 0,
        |d| d.offsets[1] = usize::MAX,
        |d| d.unknown[0] = (usize::MAX, 0),
        |d| d.unknown.swap(0, 1),
        |d| {
            d.unknown.pop();
        },
        |d| d.known[0] = usize::MAX,
        |d| d.head_columns_bound += 1,
        |d| d.minimum_mass = f64::NAN,
        |d| d.shape.programs = usize::MAX,
    ];
    for change in changes {
        let mut bad = d.clone();
        change(&mut bad);
        assert!(validate(&bad, &mut work()).is_err());
        assert!(Parameters::initial(&bad, 2., LIMIT, &mut work()).is_err());
    }
    assert!(dictionary_bytes(d.shape, 0, 0).is_err());
    assert!(dictionary_bytes(d.shape, d.shape.rows(), d.shape.rows() + 1).is_err());
}
#[test]
fn large_complete_dictionary_is_admitted_without_a_dense_tensor_allocation() {
    let sh = Shape {
        programs: 131072,
        states: 4,
        classes: 1024,
    };
    assert!(sh.admit(0, sh.rows(), sh.cells() / 4).is_err());
    let d = from_labels(
        sh,
        vec![None; sh.rows()],
        1.,
        8.,
        32 * 1024 * 1024,
        &mut work(),
    )
    .unwrap();
    validate(&d, &mut work()).unwrap();
    assert_eq!(d.unknown.len(), sh.rows());
    assert_eq!(d.parameters(), sh.programs + sh.cells());
    assert_eq!(d.head_columns_bound, sh.cells() / 4);
    assert!(allocation_bytes(&d) < 32 * 1024 * 1024);
    // This checks actual dictionary storage, not a full original-cohort fit.
    assert_eq!(d.unknown.last(), Some(&(sh.programs - 1, 3)));
}
