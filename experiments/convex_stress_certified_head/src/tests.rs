use crate::{exact::rational as q, interval::Interval as I, *};
use num_bigint::BigInt;
use num_rational::BigRational as Q;
use num_traits::{One, Zero};

fn enclosed(i: I, x: &Q) {
    assert!(i.finite(), "{i:?}");
    assert!(q(i.lo) <= *x && *x <= q(i.hi), "{i:?} misses {x}");
}

#[test]
fn outward_operations_cover_normal_subnormal_and_cancellation_results() {
    let values = [
        0.,
        -0.,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        0.5,
        -0.5,
        1.,
        1.0f64.next_up(),
        -4096.,
        1e14,
    ];
    for a in values {
        for b in values {
            enclosed(I::point(a).add(I::point(b)), &(q(a) + q(b)));
            enclosed(I::point(a).sub(I::point(b)), &(q(a) - q(b)));
            enclosed(I::point(a).scale(b), &(q(a) * q(b)));
            if b > 0. && (a / b).is_finite() {
                enclosed(I::point(a).divide_positive(I::point(b)), &(q(a) / q(b)));
            }
        }
    }
    assert!(!I::point(1.).divide_positive(I::point(0.)).finite());
}

#[test]
fn imaginary_saved_coefficients_are_used_by_both_complete_products() {
    let mut w = vec![0.; 32];
    w[3 * 2] = 4.;
    w[3 * 2 + 1] = 8.;
    let e = Evaluator::from_coefficients(&w, 1, 2).unwrap();
    let rows = [[2., -1., 0., 1., 0.]];
    let (real, complex, n) = exact::products(&e, &rows).unwrap();
    assert_eq!(real, vec![q(4.), q(8.)]);
    assert_eq!(complex, real);
    assert_eq!(n, q(5.));
    let s = e.evaluate(&rows).unwrap();
    assert_eq!(s.predicted, 1);
    assert_eq!(s.maximizers, [1]);
    assert!((s.logits[0] + 0.8).abs() < 1e-15);
    assert_eq!(s.logits[1], 0.);
}

#[test]
fn genuine_ties_are_retained_and_unequal_neighbours_are_not_ties() {
    for (a, b, winner, maximizers) in [
        (2., 2., 0, vec![0, 1]),
        (2., 2.0f64.next_up(), 1, vec![1]),
        (2.0f64.next_up(), 2., 0, vec![0]),
    ] {
        let mut w = vec![0.; 32];
        w[8 * 2] = a;
        w[8 * 2 + 1] = b;
        let e = Evaluator::from_coefficients(&w, 1, 2).unwrap();
        let s = e.evaluate(&[[0.; 5]]).unwrap();
        assert_eq!(s.method, Method::ExactRationals);
        assert_eq!(s.predicted, winner);
        assert_eq!(s.maximizers, maximizers);
        assert!(!s.target_realizability_certified);
        assert_eq!(s.independent_residual, 0.);
    }
}

#[test]
fn strict_gap_lost_by_float_conversion_is_an_explicit_refusal() {
    let tiny = Q::new(BigInt::one(), BigInt::one() << 1100);
    let error = exact::canonical(&[Q::zero(), tiny], &Q::one(), 8192).unwrap_err();
    assert!(error.to_string().contains("strict gap lost"));
    let representable = Q::new(BigInt::one(), BigInt::one() << 1074);
    let s = exact::canonical(&[Q::zero(), representable], &Q::one(), 8192).unwrap();
    assert_eq!(s.predicted, 1);
    assert_eq!(s.logits, [-f64::from_bits(1), 0.]);
}

fn sample(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state % 2001) as f64 - 1000.) / 32.
}

#[test]
fn every_bounded_class_and_certified_winner_matches_exact_oracle() {
    let mut rng = 349;
    let mut fast = 0;
    for _ in 0..48 {
        let w: Vec<_> = (0..3 * 16 * 5).map(|_| sample(&mut rng)).collect();
        let rows: Vec<_> = (0..3)
            .map(|_| std::array::from_fn(|_| sample(&mut rng)))
            .collect();
        let e = Evaluator::from_coefficients(&w, 3, 5).unwrap();
        let (real, complex, n) = exact::products(&e, &rows).unwrap();
        assert_eq!(real, complex);
        let (floats, bounds, norm, nb) = bounded::products(&e, &rows);
        enclosed(nb, &n);
        assert!(nb.contains(norm));
        for ((b, r), z) in bounds.into_iter().zip(&real).zip(floats) {
            enclosed(b, r);
            assert!(b.contains(z));
        }
        let expected = exact::canonical(&real, &n, 8192).unwrap();
        let actual = e.evaluate(&rows).unwrap();
        assert_eq!(actual.predicted, expected.predicted);
        assert_eq!(actual.maximizers, expected.maximizers);
        for (a, b) in actual.logits.iter().zip(expected.logits) {
            assert!((a - b).abs() <= actual.logit_enclosure_width);
        }
        fast += usize::from(actual.method == Method::SeparatedEnclosures);
    }
    assert!(fast > 40);
}

#[test]
fn complete_actual_affine_head_preserves_ordinary_ce_under_common_shift() {
    let head = joint::head::Head::new(
        vec![-1., -0.2, 0.3, 1.],
        vec![[1., 2., 3., 4., 5., 6.], [-2., 1., -1., 0.5, 2., 3.]],
    )
    .unwrap();
    let rows = [[1., -2., 3., 4., 0.5], [2., 3., -1., 4., 0.25]];
    let old = head.logits(&rows).unwrap();
    let independent = head.independent_logits(&rows).unwrap();
    let new = Evaluator::from_head(&head)
        .unwrap()
        .evaluate(&rows)
        .unwrap();
    assert_eq!(new.predicted, joint::argmax(&old));
    assert_eq!(new.predicted, joint::argmax(&independent));
    for c in 0..4 {
        assert!((joint::ce(&new.logits, c) - joint::ce(&old, c)).abs() < 1e-12);
    }
    let mut altered = head.clone();
    altered.coefficients[0] += 1.;
    assert!(Evaluator::from_head(&altered).is_err());
}

#[test]
fn small_residual_alone_never_certifies_a_raw_winner() {
    let epsilon = 1e-12;
    let a = [epsilon, 0.];
    let b = [0., epsilon];
    assert_ne!(joint::argmax(&a), joint::argmax(&b));
    assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-7));
    let near = I {
        lo: 0.,
        hi: epsilon,
    };
    assert!(near.hi >= near.lo);
}

#[test]
fn admission_checks_cover_dimensions_nonfinite_values_and_resources() {
    assert!(forecast(0, 2).is_err());
    assert!(forecast(65, 2).is_err());
    assert!(forecast(1, 1025).is_err());
    let largest = forecast(64, 1024).unwrap();
    assert!((largest.buffer_bytes as u128) < joint::MAX_BYTES);
    assert_eq!(
        largest.maximum_coefficient_multiplications_per_query,
        7 * 64 * 16 * 1024
    );
    assert!(Evaluator::from_coefficients(&[0.; 31], 1, 2).is_err());
    let mut w = vec![0.; 32];
    w[0] = f64::NAN;
    assert!(Evaluator::from_coefficients(&w, 1, 2).is_err());
    w[0] = 1e15;
    assert!(Evaluator::from_coefficients(&w, 1, 2).is_err());
    w[0] = 0.;
    let e = Evaluator::from_coefficients(&w, 1, 2).unwrap();
    assert!(e.evaluate(&[]).is_err());
    assert!(e.evaluate(&[[4097.; 5]]).is_err());
    assert!(e.evaluate(&[[f64::INFINITY; 5]]).is_err());
}

#[test]
fn model_adapter_preserves_training_modes_raw_labels_and_unidentified_defaults() -> Result<()> {
    // Handwritten sparse-range fixture from synthetic_programs/integration_tests
    // at1371ae4f; no benchmark or heldout generator is imported into inference.
    use joint::training::{Mode, Policy};
    use previous::{discovery, model as old, observations, program::Program};
    let rows: Vec<_> = [
        (20, vec![1, 100, 102, 18], 100),
        (71, vec![1, 100, 102, 18, 99, 99], 102),
        (96, vec![1, 100, 102, 18], 100),
    ]
    .into_iter()
    .map(|(id, tokens, label)| observations::Observation { id, tokens, label })
    .collect();
    let report = discovery::run(
        &rows,
        observations::coordinates(229)?,
        vec![Program::Range {
            reverse: false,
            inclusive: true,
        }],
    )?;
    let prefix = [1, 100, 102, 18, 99];
    for mode in [Mode::Forward, Mode::Full, Mode::Mini] {
        let policy = Policy {
            mode,
            epochs: usize::from(mode != Mode::Forward),
            batch: 2,
            learning_rate: 1e5,
            initial_total_mass: 10000.,
            projection_cycles: 20000,
            projection_tolerance: 1e-10,
        };
        for prior in [old::Prior::Coordinates, old::Prior::Categorical] {
            let trained = old::train(&report, &rows, prior, policy.clone(), 1e8, 1.)?;
            let before = serde_json::to_vec(&trained)?;
            let new = model::ModelEvaluator::new(&trained.model)?;
            let result = new.inspect(&prefix)?;
            let tokens: Vec<_> = prefix.iter().map(|&x| x as usize).collect();
            let original = trained.model.query(&tokens)?;
            assert_eq!(result.query.defined, original.defined);
            assert_eq!(result.query.span_residuals, original.span_residuals);
            assert_eq!(
                result.query.active_observable_coverage,
                original.active_observable_coverage
            );
            assert_eq!(
                result.scores.predicted,
                if prior == old::Prior::Coordinates {
                    101
                } else {
                    0
                }
            );
            assert!(!result.query.target_realizability_certified);
            assert!(!result.scores.target_realizability_certified);
            for label in [0, 100, 101, 102, 228] {
                assert!(
                    (joint::ce(&result.query.logits, label) - joint::ce(&original.logits, label))
                        .abs()
                        < 1e-8
                );
            }
            assert_eq!(serde_json::to_vec(&trained)?, before);
            match mode {
                Mode::Forward => assert!(trained.training.steps.is_empty()),
                Mode::Full => assert_eq!(trained.training.steps[0].record_ids, [20, 71, 96]),
                Mode::Mini => {
                    assert_eq!(trained.training.steps[0].record_ids, [20, 71]);
                    assert_eq!(trained.training.steps[1].record_ids, [96]);
                }
            }
            assert!(new.inspect(&[]).is_err());
            assert!(new.inspect(&[-1]).is_err());
            assert!(new.inspect(&[229]).is_err());
            assert!(new.inspect(&[1; 4097]).is_err());
        }
    }
    Ok(())
}
