use crate::counts::{Context, Counts, Observation};
use baseline::channel::cross_entropy;
use bilinear::train::Record;

fn context(values: [i64; 2], selected: usize, position: usize) -> Context {
    let mut prefix = vec![1, 36, values[0], 37, values[1]];
    prefix.resize(position, 99);
    prefix.push([36, 37][selected]);
    Context::new(&Record {
        prefix: &prefix,
        target: values[selected] as usize,
    })
    .unwrap()
}

#[test]
fn original_partition_is_not_candidate_restricted_accuracy() {
    let c = context([68, 69], 0, 5);
    let mut counts = Counts::default();
    for (prediction, category) in [(68, 0), (69, 1), (3, 2)] {
        let mut logits = vec![0.; 100];
        logits[prediction] = 4.;
        let o = Observation::new(&c, &logits).unwrap();
        assert_eq!(o.category, category);
        assert_eq!(o.choice_correct, category != 1);
        assert!((o.full_ce - cross_entropy(&logits, 68)).abs() < 1e-14);
        counts.observe(&o);
    }
    let r = counts.report();
    assert_eq!(counts.categories, [1, 1, 1]);
    assert_eq!(r["original_accuracy"], 1. / 3.);
    assert_eq!(r["two_value_choice_accuracy"], 2. / 3.);
    assert_eq!(r["stored_value_prediction_rate"], 2. / 3.);
}

#[test]
fn duplicate_values_keep_original_metrics_but_not_two_choice_denominators() {
    let mut counts = Counts::default();
    for selected in 0..2 {
        let c = context([72, 72], selected, 11);
        for prediction in [72, 99] {
            let mut logits = vec![0.; 100];
            logits[prediction] = 3.;
            let o = Observation::new(&c, &logits).unwrap();
            assert!(!o.distinct);
            assert!(o.choice_correct);
            assert_eq!(o.pair_ce, 0.);
            assert_eq!(o.full_ce, o.mass_ce);
            counts.observe(&o);
        }
    }
    let r = counts.report();
    assert_eq!(counts.categories, [2, 0, 2]);
    assert_eq!(r["duplicate_value_records"], 4);
    assert_eq!(r["original_accuracy"], 0.5);
    assert!(r["two_value_choice_accuracy"].is_null());
    assert!(r["mean_correct_minus_other_logit"].is_null());
    assert!(Counts::default().report()["original_accuracy"].is_null());
}

#[test]
fn choice_uses_actual_class_tie_order_and_total_float_order() {
    for values in [[69, 68], [68, 69]] {
        for selected in 0..2 {
            let c = context(values, selected, 7);
            let mut logits = vec![-2.; 100];
            logits[68] = 0.;
            logits[69] = 0.;
            let o = Observation::new(&c, &logits).unwrap();
            assert_eq!(o.choice_correct, c.target == 68);
            assert_eq!(o.category, usize::from(c.target != 68));
            logits[68] = -0.;
            let o = Observation::new(&c, &logits).unwrap();
            assert_eq!(o.choice_correct, c.target == 69);
            assert_eq!(o.category, usize::from(c.target != 69));
        }
    }
}

#[test]
fn complete_ce_decomposes_with_all_classes_and_is_shift_invariant() {
    let c = context([90, 71], 1, 9);
    let logits: Vec<_> = (0..100)
        .map(|i| ((i * 17 % 31) as f64 - 15.) / 3.)
        .collect();
    let a = Observation::new(&c, &logits).unwrap();
    let shifted: Vec<_> = logits.iter().map(|v| v + 73.).collect();
    let b = Observation::new(&c, &shifted).unwrap();
    assert!(a.mass_ce > 0.);
    assert!(a.pair_ce > 0.);
    assert!((a.full_ce - a.mass_ce - a.pair_ce).abs() < 2e-14);
    assert!((a.mass - (-a.mass_ce).exp()).abs() < 1e-14);
    for (x, y) in [
        (a.full_ce, b.full_ce),
        (a.mass_ce, b.mass_ce),
        (a.pair_ce, b.pair_ce),
        (a.mass, b.mass),
    ] {
        assert!((x - y).abs() < 2e-14);
    }
    assert!(Observation::new(&c, &logits[..99]).is_err());
    let mut bad = logits;
    bad[0] = f64::NAN;
    assert!(Observation::new(&c, &bad).is_err());
}

#[test]
fn all_original_prefix_positions_and_duplicate_values_are_retained() {
    for position in 5..12 {
        for selected in 0..2 {
            for values in [[68, 99], [68, 68]] {
                let c = context(values, selected, position);
                assert_eq!(c.query_position, position);
                assert_eq!(c.selected, selected);
                assert_eq!(c.target, values[selected] as usize);
            }
        }
    }
}

#[test]
fn malformed_roles_are_refused_instead_of_relabelled() {
    let good = [1, 36, 68, 37, 99, 70, 36];
    for (position, replacement) in [
        (0, 2),
        (1, 35),
        (3, 36),
        (2, 67),
        (4, 100),
        (5, 37),
        (6, 38),
    ] {
        let mut prefix = good;
        prefix[position] = replacement;
        assert!(
            Context::new(&Record {
                prefix: &prefix,
                target: 68
            })
            .is_err()
        );
    }
    assert!(
        Context::new(&Record {
            prefix: &good,
            target: 99
        })
        .is_err()
    );
    for prefix in [
        &[][..],
        &good[..5],
        &[1, 36, 68, 37, 99, 70, 70, 70, 70, 70, 70, 70, 36][..],
    ] {
        assert!(Context::new(&Record { prefix, target: 68 }).is_err());
    }
}
