use super::*;
const LIMIT: u64 = 32 * 1024 * 1024;
fn work() -> Work {
    Work::new(2_000_000_000_000).unwrap()
}
fn bits(xs: impl IntoIterator<Item = f64>) -> Vec<u64> {
    xs.into_iter().map(f64::to_bits).collect()
}

fn certificate_words(head: &mut Head, states: &[usize]) {
    let bounds = head.intervals(states);
    let second = head.reverse_products(states);
    let integers = exact::numerators(head, states).unwrap();
    let plan = head.sharing.take();
    let reference_bounds = head.intervals(states);
    let reference_second = head.reverse_products(states);
    let reference_integers = exact::numerators(head, states).unwrap();
    head.sharing = plan;
    for (a, b) in bounds.iter().zip(&reference_bounds) {
        assert_eq!(
            (a.lo.to_bits(), a.hi.to_bits()),
            (b.lo.to_bits(), b.hi.to_bits())
        );
    }
    assert_eq!(bits(second), bits(reference_second));
    assert_eq!(integers, reference_integers);
}

fn compare(d: &Domain, old: &legacy::Parameters) -> Head {
    let q = Parameters::from_legacy(d, old, LIMIT, &mut work()).unwrap();
    let (mut wa, mut wb) = (work(), work());
    let full = legacy::head::Head::build(d, old, &mut wa).unwrap();
    let mut new = Head::build(d, &q, LIMIT, &mut wb).unwrap();
    for p in 0..d.shape.programs {
        for f in 0..d.shape.states {
            for c in 0..d.shape.classes {
                let (a, b) = full.coefficient(p, f, c).unwrap();
                let (x, y) = new.coefficient(p, f, c).unwrap();
                assert_eq!((a.to_bits(), b.to_bits()), (x.to_bits(), y.to_bits()));
            }
        }
    }
    for s in 0..d.shape.states {
        let states = (0..d.shape.programs)
            .map(|p| (s + 3 * p) % d.shape.states)
            .collect::<Vec<_>>();
        assert_eq!(
            bits(full.logits(&states, &mut wa).unwrap()),
            bits(new.logits(&states, &mut wb).unwrap())
        );
        certificate_words(&mut new, &states);
    }
    assert_eq!(wa.used, wb.used);
    new
}

#[test]
fn conditional_construction_preserves_every_coefficient_word_and_complete_work() {
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
    for classes in [2, 3, 4, 5, 7, 97, 193] {
        let shape = Shape {
            programs: 2,
            states: 8,
            classes,
        };
        let mut labels = vec![None; 16];
        for row in (0..16).step_by(4) {
            labels[row] = Some(row % classes);
            labels[row + 1] = Some((row + 1) % classes);
        }
        let d = Domain::from_labels(shape, labels, 1., 8., &mut work()).unwrap();
        for shift in 0..words.len() {
            let mut old = d.initial(4., &mut work()).unwrap();
            for (i, &(p, s)) in d.unknown.iter().enumerate() {
                let at = d.offsets[p * 8 + s];
                old.values[at..at + classes].fill(words[(shift + i) % words.len()]);
            }
            let q = Parameters::from_legacy(&d, &old, LIMIT, &mut work()).unwrap();
            let plan = Construction::new(&d, &q, 0, &Support::new(&d, 0));
            assert_eq!(plan.is_some(), classes > 3);
            compare(&d, &old);
        }
    }
}

#[test]
fn one_nonuniform_or_different_zero_word_keeps_the_complete_block_fallback() {
    let shape = Shape {
        programs: 1,
        states: 8,
        classes: 7,
    };
    let d = Domain::from_labels(
        shape,
        vec![Some(0), None, None, None, Some(1), None, None, None],
        1.,
        8.,
        &mut work(),
    )
    .unwrap();
    let mut old = d.initial(4., &mut work()).unwrap();
    for &(p, s) in &d.unknown {
        let at = d.offsets[p * 8 + s];
        old.values[at..at + 7].fill(0.);
    }
    let at = d.offsets[1];
    for exceptional in [-0., f64::from_bits(1), 0.25] {
        old.values[at + 6] = exceptional;
        let q = Parameters::from_legacy(&d, &old, LIMIT, &mut work()).unwrap();
        assert!(Construction::new(&d, &q, 0, &Support::new(&d, 0)).is_none());
        assert!(Construction::new(&d, &q, 4, &Support::new(&d, 4)).is_some());
        compare(&d, &old);
    }
    let all_known = Domain::from_labels(
        shape,
        (0..8).map(|s| Some(s % 7)).collect(),
        1.,
        8.,
        &mut work(),
    )
    .unwrap();
    let q = Parameters::initial(&all_known, 4., LIMIT, &mut work()).unwrap();
    assert!(Construction::new(&all_known, &q, 0, &Support::new(&all_known, 0)).is_none());
}

#[test]
fn independent_interval_reverse_and_integer_passes_expand_all_words_and_ties() {
    let shape = Shape {
        programs: 2,
        states: 8,
        classes: 193,
    };
    for known in [false, true] {
        let mut labels = vec![None; 16];
        if known {
            labels[0] = Some(0);
            labels[8] = Some(0);
        }
        let d = Domain::from_labels(shape, labels, 1., 8., &mut work()).unwrap();
        let old = d.initial(4., &mut work()).unwrap();
        let full = legacy::head::Head::build(&d, &old, &mut work()).unwrap();
        let q = Parameters::from_legacy(&d, &old, LIMIT, &mut work()).unwrap();
        let mut head = Head::build(&d, &q, LIMIT, &mut work()).unwrap();
        assert!(head.sharing.is_some());
        for s in 0..8 {
            let states = [s, s];
            certificate_words(&mut head, &states);
            let (mut wa, mut wb) = (work(), work());
            let a = full
                .predict(&states, &mut PredictionBudget::new(1).unwrap(), &mut wa)
                .unwrap();
            let b = head
                .predict(&states, &mut PredictionBudget::new(1).unwrap(), &mut wb)
                .unwrap();
            super::super::tests::prediction(&a, &b);
            assert_eq!(wa.used, wb.used);
            if !known {
                assert_eq!(b.certificate, Certificate::ExactDyadics);
                assert_eq!(b.maximizers, (0..193).collect::<Vec<_>>());
            }
        }
    }
}
