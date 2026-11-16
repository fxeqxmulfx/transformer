use super::*;
use crate::rows::Rows;

const LIMIT: u64 = 16 * 1024 * 1024;
fn work() -> Work {
    Work::new(2_000_000_000_000).unwrap()
}
fn bits(values: impl IntoIterator<Item = f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}
pub(super) fn prediction(a: &Prediction, b: &Prediction) {
    assert_eq!(
        bits(a.logits.iter().copied()),
        bits(b.logits.iter().copied())
    );
    assert_eq!(a.winner, b.winner);
    assert_eq!(a.maximizers, b.maximizers);
    assert_eq!(a.certificate, b.certificate);
    assert_eq!(
        a.maximum_independent_residual.to_bits(),
        b.maximum_independent_residual.to_bits()
    );
    assert_eq!(
        a.maximum_enclosure_width.to_bits(),
        b.maximum_enclosure_width.to_bits()
    );
    assert!(!a.target_realizability_certified && !b.target_realizability_certified);
    for label in [0, a.winner, a.logits.len() - 1] {
        assert_eq!(
            legacy::ce(&a.logits, label).to_bits(),
            legacy::ce(&b.logits, label).to_bits()
        );
    }
}

/// Compare complete tensors, audits, ordinary CE and both certified query
/// paths. Counts and arithmetic work must still visit the original support.
fn compare(domain: &Domain, q: &legacy::Parameters, queries: &[Vec<usize>]) -> (usize, usize) {
    let compact = Parameters::from_legacy(domain, q, LIMIT, &mut work()).unwrap();
    assert_eq!(
        bits(q.values.iter().copied()),
        bits(compact.values().copied())
    );
    let (mut wa, mut wb) = (work(), work());
    assert_eq!(
        domain.check(q, 1e-8, &mut wa).unwrap().to_bits(),
        compact.check(domain, 1e-8, &mut wb).unwrap().to_bits()
    );
    let full = legacy::head::Head::build(domain, q, &mut wa).unwrap();
    let stored = Head::build(domain, &compact, LIMIT, &mut wb).unwrap();
    assert_eq!(wa.used, wb.used);
    assert!(stored.allocation_bytes() < LIMIT);
    assert!(stored.stored_coefficients() <= full.stored_coefficients());
    for p in 0..domain.shape.programs {
        for f in 0..domain.shape.states {
            for c in 0..domain.shape.classes {
                let a = full.coefficient(p, f, c).unwrap();
                let b = stored.coefficient(p, f, c).unwrap();
                assert_eq!(
                    (a.0.to_bits(), a.1.to_bits()),
                    (b.0.to_bits(), b.1.to_bits()),
                    "coefficient {p}/{f}/{c}"
                );
            }
        }
    }
    let a = full.audit(domain, q, &mut wa).unwrap();
    let b = stored.audit(domain, &compact, &mut wb).unwrap();
    assert_eq!(bits(a.column_l1), bits(b.column_l1));
    assert_eq!(
        a.maximum_inverse_residual.to_bits(),
        b.maximum_inverse_residual.to_bits()
    );
    assert_eq!(a.logical_coefficients, b.logical_coefficients);
    assert_eq!(wa.used, wb.used);
    let (mut ba, mut bb) = (
        PredictionBudget::new(65536).unwrap(),
        PredictionBudget::new(65536).unwrap(),
    );
    let (mut interval, mut exact) = (0, 0);
    for states in queries {
        assert_eq!(
            bits(full.logits(states, &mut wa).unwrap()),
            bits(stored.logits(states, &mut wb).unwrap())
        );
        let a = full.predict(states, &mut ba, &mut wa).unwrap();
        let b = stored.predict(states, &mut bb, &mut wb).unwrap();
        prediction(&a, &b);
        match b.certificate {
            Certificate::OutwardIntervals => interval += 1,
            Certificate::ExactDyadics => exact += 1,
        }
        assert_eq!(ba, bb);
        assert_eq!(wa.used, wb.used);
    }
    (interval, exact)
}

fn mixed() -> Domain {
    let shape = Shape {
        programs: 2,
        states: 8,
        classes: 5,
    };
    let records: Vec<_> = [
        (0, 0, 0),
        (1, 2, 1),
        (0, 0, 0),
        (4, 4, 2),
        (5, 7, 2),
        (6, 6, 3),
        (7, 5, 4),
    ]
    .into_iter()
    .map(|(a, b, label)| legacy::Record {
        states: vec![a, b],
        label,
    })
    .collect();
    Domain::new(shape, &records, 1., 8., &mut work()).unwrap()
}

#[test]
fn complete_mixed_head_matches_every_coefficient_query_certificate_and_ce() {
    let domain = mixed();
    let queries: Vec<_> = (0..8)
        .flat_map(|a| (0..8).map(move |b| vec![a, b]))
        .collect();
    let mut q = domain.initial(4., &mut work()).unwrap();
    let (interval, exact) = compare(&domain, &q, &queries);
    assert!(interval > 0 && exact > 0);
    let at = domain.offsets[3];
    q.values[at..at + 5].copy_from_slice(&[1., 1., -0., 0., f64::from_bits(1)]);
    let at = domain.offsets[11];
    q.values[at..at + 5].copy_from_slice(&[0.5, 0.125, 0.875, 0.25, 0.25]);
    let (interval, exact) = compare(&domain, &q, &queries);
    assert!(interval > 0 && exact > 0);
    let compact = Parameters::from_legacy(&domain, &q, LIMIT, &mut work()).unwrap();
    let head = Head::build(&domain, &compact, LIMIT, &mut work()).unwrap();
    assert!(
        head.columns(&head.blocks[0])
            .any(|(_, v)| v[4..].iter().any(|&x| x != 0.))
    );
    assert!(compact.unknown_rows().dense_rows() > 0);
    assert!(compact.unknown_rows().dense_rows() < compact.unknown_rows().rows());
}

#[test]
fn direct_initialization_and_full_weighted_distance_match_original_arithmetic() {
    let domain = mixed();
    let (mut wa, mut wb) = (work(), work());
    let a = domain.initial(4., &mut wa).unwrap();
    let ca = Parameters::initial(&domain, 4., LIMIT, &mut wb).unwrap();
    assert_eq!(wa.used, wb.used);
    assert_eq!(bits(a.values.iter().copied()), bits(ca.values().copied()));
    assert_eq!(ca.unknown_rows().dense_rows(), 0);
    let mut b = a.clone();
    // Arbitrary nonuniform proposal and changed masses, not only feasible
    // uniform iterates: known-row multiplicities must remain in the metric.
    for (i, x) in b.values.iter_mut().enumerate() {
        *x += (i as f64 - 7.) * 0.123;
    }
    let cb = Parameters::from_legacy(&domain, &b, LIMIT, &mut work()).unwrap();
    assert_eq!(
        domain.distance(&a, &b, &mut wa).unwrap().to_bits(),
        ca.distance(&cb, &domain, &mut wb).unwrap().to_bits()
    );
    assert_eq!(wa.used, wb.used);
    assert!(ca.allocation_bytes() < LIMIT);
}

#[test]
fn actual_shared_head_storage_reduces_bytes_without_reducing_logical_support() {
    let shape = Shape {
        programs: 1,
        states: 128,
        classes: 1024,
    };
    let domain = Domain::from_labels(shape, vec![None; 128], 1., 8., &mut work()).unwrap();
    let full_q = domain.initial(8., &mut work()).unwrap();
    let q = Parameters::initial(&domain, 8., LIMIT, &mut work()).unwrap();
    let full = legacy::head::Head::build(&domain, &full_q, &mut work()).unwrap();
    let h = Head::build(&domain, &q, LIMIT, &mut work()).unwrap();
    assert_eq!(h.stored_coefficients(), 32 * 8);
    assert_eq!(full.stored_coefficients(), 32 * 8 * 1024);
    assert!(h.allocation_bytes() < 8192);
    assert!(q.allocation_bytes() < 8192);
    assert_eq!(q.values().count(), full_q.values.len());
    for s in 0..128 {
        assert_eq!(h.query(&[s]).unwrap(), 1024);
        assert_eq!(h.columns(&h.blocks[s / 4]).count(), 1024);
        assert_eq!(
            bits(h.logits(&[s], &mut work()).unwrap()),
            bits(full.logits(&[s], &mut work()).unwrap())
        );
    }
    let audit = h.audit(&domain, &q, &mut work()).unwrap();
    assert_eq!(audit.logical_coefficients, 128 * 1024 * 2);
    assert!(
        audit
            .column_l1
            .iter()
            .all(|&n| n == audit.column_l1[0] && n > 0.)
    );
}

#[test]
fn raw_class_boundaries_zero_mass_and_full_nonuniform_fallback_survive() {
    for classes in [2, 3, 4, 5, 7, 97, 193, 1024] {
        let shape = Shape {
            programs: 2,
            states: 8,
            classes,
        };
        let mut labels = vec![None; 16];
        for p in 0..2 {
            for s in 0..4 {
                labels[p * 8 + s] = Some(s % classes);
            }
        }
        let domain = Domain::from_labels(shape, labels, 1., 8., &mut work()).unwrap();
        let mut q = domain.initial(4., &mut work()).unwrap();
        q.values[0] = 4.;
        q.values[1] = -0.;
        for &(p, s) in &domain.unknown {
            let at = domain.offsets[p * 8 + s];
            for c in 0..classes {
                q.values[at + c] = if p == 1 {
                    -0.
                } else {
                    8. * (c + 1) as f64 / (classes * (classes + 1)) as f64
                };
            }
        }
        let queries: Vec<_> = (0..8).map(|s| vec![s, 7 - s]).collect();
        compare(&domain, &q, &queries);
    }
}

#[test]
fn opposite_unknown_answers_and_genuine_exact_ties_are_not_collapsed() {
    let shape = Shape {
        programs: 1,
        states: 4,
        classes: 2,
    };
    let domain = Domain::from_labels(
        shape,
        vec![Some(0), Some(1), None, None],
        1.,
        8.,
        &mut work(),
    )
    .unwrap();
    let mut q = domain.initial(4., &mut work()).unwrap();
    let queries: Vec<_> = (0..4).map(|s| vec![s]).collect();
    let (_, exact) = compare(&domain, &q, &queries);
    assert_eq!(exact, 2);
    for winner in 0..2 {
        let at = domain.offsets[2];
        q.values[at..at + 2].copy_from_slice(if winner == 0 { &[4., 0.] } else { &[0., 4.] });
        compare(&domain, &q, &queries);
        let stored = Parameters::from_legacy(&domain, &q, LIMIT, &mut work()).unwrap();
        let head = Head::build(&domain, &stored, LIMIT, &mut work()).unwrap();
        assert_eq!(
            head.predict(&[2], &mut PredictionBudget::new(1).unwrap(), &mut work())
                .unwrap()
                .winner,
            winner
        );
    }
    let q = Parameters::initial(&domain, 4., LIMIT, &mut work()).unwrap();
    let head = Head::build(&domain, &q, LIMIT, &mut work()).unwrap();
    assert!(
        head.predict(&[2], &mut PredictionBudget::new(0).unwrap(), &mut work())
            .is_err()
    );
    assert_eq!(
        head.predict(&[2], &mut PredictionBudget::new(1).unwrap(), &mut work())
            .unwrap()
            .maximizers,
        vec![0, 1]
    );
}

#[test]
fn arbitrary_proposals_keep_signed_zero_subnormal_and_negative_complex_products() {
    let shape = Shape {
        programs: 1,
        states: 4,
        classes: 7,
    };
    let domain = Domain::from_labels(shape, vec![None; 4], 1., 8., &mut work()).unwrap();
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
        let mut q = domain.initial(4., &mut work()).unwrap();
        for (i, x) in q.values.iter_mut().enumerate() {
            *x = words[(i + shift) % words.len()];
        }
        let compact = Parameters::from_legacy(&domain, &q, LIMIT, &mut work()).unwrap();
        let old = legacy::head::Head::build(&domain, &q, &mut work()).unwrap();
        let new = Head::build(&domain, &compact, LIMIT, &mut work()).unwrap();
        for f in 0..4 {
            for c in 0..7 {
                let a = old.coefficient(0, f, c).unwrap();
                let b = new.coefficient(0, f, c).unwrap();
                assert_eq!(
                    (a.0.to_bits(), a.1.to_bits()),
                    (b.0.to_bits(), b.1.to_bits())
                );
            }
        }
        for s in 0..4 {
            assert_eq!(
                bits(old.logits(&[s], &mut work()).unwrap()),
                bits(new.logits(&[s], &mut work()).unwrap())
            );
        }
        assert!(compact.check(&domain, 1e-7, &mut work()).is_err());
    }
}

#[test]
fn malformed_parameters_queries_support_and_head_values_are_rejected() {
    let domain = mixed();
    let mut q = domain.initial(4., &mut work()).unwrap();
    let compact = Parameters::from_legacy(&domain, &q, LIMIT, &mut work()).unwrap();
    assert!(Parameters::initial(&domain, 4., 1, &mut work()).is_err());
    assert!(Parameters::initial(&domain, f64::NAN, LIMIT, &mut work()).is_err());
    assert!(Head::build(&domain, &compact, 1, &mut work()).is_err());
    for bad in [f64::NAN, f64::INFINITY, 1.1e12] {
        q.values[0] = bad;
        assert!(Parameters::from_legacy(&domain, &q, LIMIT, &mut work()).is_err());
    }
    q.values.pop();
    assert!(Parameters::from_legacy(&domain, &q, LIMIT, &mut work()).is_err());
    let head = || Head::build(&domain, &compact, LIMIT, &mut work()).unwrap();
    assert!(head().logits(&[0], &mut work()).is_err());
    assert!(head().logits(&[0, 8], &mut work()).is_err());
    for (p, f, c) in [(2, 0, 0), (0, 8, 0), (0, 0, 5)] {
        assert!(head().coefficient(p, f, c).is_err());
    }
    let mut h = head();
    h.blocks[0].exception_start = usize::MAX;
    assert!(h.audit(&domain, &compact, &mut work()).is_err());
    let mut h = head();
    h.blocks[0].active_len -= 1;
    assert!(h.audit(&domain, &compact, &mut work()).is_err());
    let mut h = head();
    h.active[0] = 1024;
    assert!(h.audit(&domain, &compact, &mut work()).is_err());
    let mut h = head();
    h.exceptions[0].class = 1024;
    assert!(h.audit(&domain, &compact, &mut work()).is_err());
    let mut h = head();
    h.blocks[0].baseline[0] = f64::NAN;
    assert!(h.audit(&domain, &compact, &mut work()).is_err());
}

#[test]
fn head_growth_reserves_old_plus_new_and_refusal_keeps_existing_blocks() {
    let shape = Shape {
        programs: 1,
        states: 8,
        classes: 5,
    };
    let domain = Domain::from_labels(shape, vec![None; 8], 1., 8., &mut work()).unwrap();
    let floor = (1..4096).find(|&n| Builder::new(shape, n).is_ok()).unwrap();
    let exception = std::mem::size_of::<Exception>() as u64;
    let mut builder = Builder::new(shape, floor + 8 * exception).unwrap();
    let support = Support::new(&domain, 0);
    for (c, v) in builder.scratch.iter_mut().enumerate() {
        v.fill(c as f64);
    }
    builder.push(&support).unwrap();
    // Eight exceptions fit after growth; twelve are live during a move.
    assert!(builder.push(&support).is_err());
    builder.scratch.fill([-0.; 8]);
    builder.push(&support).unwrap();
    let head = builder.finish().unwrap();
    for c in 0..5 {
        assert_eq!(head.coefficient(0, 0, c).unwrap().0, c as f64);
        assert_eq!(
            head.coefficient(0, 4, c).unwrap().0.to_bits(),
            (-0.0f64).to_bits()
        );
    }
    assert!(head.allocation_bytes() <= floor + 8 * exception);
    assert!(Builder::new(shape, LIMIT).unwrap().finish().is_err());
    // A row codec can preserve signed zeros; the actual head tests above
    // additionally compare their numerical products against the old model.
    assert_eq!(
        Rows::from_dense(2, &[-0., 0.], LIMIT).unwrap().dense_rows(),
        1
    );
}
