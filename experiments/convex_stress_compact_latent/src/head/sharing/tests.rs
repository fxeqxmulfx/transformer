use super::*;
const LIMIT: u64 = 32 * 1024 * 1024;
fn work() -> Work {
    Work::new(2_000_000_000_000).unwrap()
}
fn bits(values: Vec<f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}
fn same_queries(head: &mut Head, queries: &[Vec<usize>]) {
    for states in queries {
        let (mut full_work, mut shared_work) = (work(), work());
        let shared = head.logits(states, &mut shared_work).unwrap();
        let plan = head.sharing.take();
        let full = head.logits(states, &mut full_work).unwrap();
        head.sharing = plan;
        for c in 0..head.shape.classes {
            assert_eq!(
                crate::ce(&full, c).to_bits(),
                crate::ce(&shared, c).to_bits()
            );
        }
        assert_eq!(bits(full), bits(shared));
        assert_eq!(full_work.used, shared_work.used);
    }
}

#[test]
fn known_only_active_classes_stay_exceptional_even_with_zero_coefficients() {
    let sh = Shape {
        programs: 1,
        states: 8,
        classes: 7,
    };
    let domain = Domain::from_labels(
        sh,
        vec![Some(3), Some(3), Some(3), Some(3), None, None, None, None],
        1.,
        8.,
        &mut work(),
    )
    .unwrap();
    let mut full = domain.initial(4., &mut work()).unwrap();
    full.values.fill(0.);
    let q = Parameters::from_legacy(&domain, &full, LIMIT, &mut work()).unwrap();
    let mut head = Head::build(&domain, &q, LIMIT, &mut work()).unwrap();
    let plan = head.sharing.as_ref().unwrap();
    assert_eq!(plan.representatives, [0, 3]);
    assert_eq!(plan.duplicates, [1, 2, 4, 5, 6]);
    assert_eq!(head.evaluated_columns(&[0]).unwrap(), 1);
    assert_eq!(head.evaluated_columns(&[4]).unwrap(), 2);
    assert_eq!(head.query(&[4]).unwrap(), 7);
    plan.validate(&head).unwrap();
    same_queries(&mut head, &(0..8).map(|s| vec![s]).collect::<Vec<_>>());
}

#[test]
fn shared_signed_zero_subnormal_and_imaginary_words_preserve_original_operations() {
    let sh = Shape {
        programs: 2,
        states: 8,
        classes: 7,
    };
    let domain = Domain::from_labels(sh, vec![None; 16], 1., 8., &mut work()).unwrap();
    let words = [
        0.,
        -0.,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        0.1,
        -0.1,
        1e12,
        -1e12,
    ];
    for shift in 0..words.len() {
        let mut builder = Builder::new(sh, LIMIT).unwrap();
        for row in (0..16).step_by(4) {
            let mut baseline = [0.; 8];
            for (f, x) in baseline.iter_mut().enumerate() {
                *x = words[(shift + row + f) % words.len()];
            }
            builder.scratch.fill(baseline);
            builder.scratch[3][shift % 8] = words[(shift + 1) % words.len()];
            builder.push(&Support::new(&domain, row)).unwrap();
        }
        let mut head = builder.finish().unwrap();
        assert!(head.sharing.is_some());
        head.sharing.as_ref().unwrap().validate(&head).unwrap();
        for p in 0..2 {
            for f in 0..8 {
                let a = head.coefficient(p, f, 1).unwrap();
                let b = head.coefficient(p, f, 6).unwrap();
                assert_eq!(
                    (a.0.to_bits(), a.1.to_bits()),
                    (b.0.to_bits(), b.1.to_bits())
                );
            }
        }
        let queries = (0..8)
            .flat_map(|a| (0..8).map(move |b| vec![a, b]))
            .collect::<Vec<_>>();
        same_queries(&mut head, &queries);
    }
}

#[test]
fn global_union_and_complete_nonuniform_fallback_are_not_query_local() {
    let sh = Shape {
        programs: 1,
        states: 8,
        classes: 4,
    };
    let domain = Domain::from_labels(sh, vec![None; 8], 1., 8., &mut work()).unwrap();
    let mut builder = Builder::new(sh, LIMIT).unwrap();
    builder.scratch.fill([1.; 8]);
    builder.scratch[1].fill(2.);
    builder.push(&Support::new(&domain, 0)).unwrap();
    builder.scratch.fill([3.; 8]);
    builder.scratch[2].fill(4.);
    builder.scratch[3].fill(5.);
    builder.push(&Support::new(&domain, 4)).unwrap();
    let mut head = builder.finish().unwrap();
    assert!(
        head.sharing.is_none(),
        "each block alone could share but their union cannot"
    );
    for s in 0..8 {
        assert_eq!(head.evaluated_columns(&[s]).unwrap(), 4);
    }
    same_queries(&mut head, &(0..8).map(|s| vec![s]).collect::<Vec<_>>());
}

#[test]
fn plan_metadata_and_simultaneous_construction_require_admitted_capacity() {
    let sh = Shape {
        programs: 1,
        states: 8,
        classes: 193,
    };
    let domain = Domain::from_labels(sh, vec![None; 8], 1., 8., &mut work()).unwrap();
    let q = Parameters::initial(&domain, 4., LIMIT, &mut work()).unwrap();
    let mut head = Head::build(&domain, &q, LIMIT, &mut work()).unwrap();
    let logical_before = bits(head.logits(&[0], &mut work()).unwrap());
    let plan_bytes = head.sharing.as_ref().unwrap().heap_bytes();
    assert_eq!(plan_bytes, sh.classes * std::mem::size_of::<u16>());
    assert!(Sharing::build(&head, 1).is_err());
    assert!(Sharing::build(&head, sh.classes as u64 * 3).is_err());
    assert_eq!(
        logical_before,
        bits(head.logits(&[0], &mut work()).unwrap())
    );
    let mut plan = head.sharing.take().unwrap();
    let without = head.allocation_bytes();
    plan.validate(&head).unwrap();
    plan.duplicates[0] = plan.representative;
    assert!(plan.validate(&head).is_err());
    head.sharing = Sharing::build(&head, LIMIT).unwrap();
    assert_eq!(head.allocation_bytes(), without + plan_bytes as u64);
    head.sharing.as_mut().unwrap().representative = 1024;
    assert!(head.audit(&domain, &q, &mut work()).is_err());
}

#[path = "benchmark.rs"]
mod benchmark;
