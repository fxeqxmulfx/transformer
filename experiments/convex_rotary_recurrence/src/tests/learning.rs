use crate::{
    C,
    atom::{Atom, Family},
    coordinates,
    model::Model,
    proposal::{Search, readout_direction},
    train::{self, Config},
};
use baseline::{data::Example, random::Random};
use bilinear::{
    experiment as original,
    model::cross_entropy,
    train::{Mode, Record},
};
#[test]
fn complete_readout_and_finite_ce_competitors_remain_present() {
    let a = [C::new(0.3, 0.4), C::new(-0.6, 0.8)];
    let u = readout_direction(a).unwrap();
    assert!(((u[0] * a[0] + u[1] * a[1]).re + 1.).abs() < 1e-15);
    assert_eq!(u[0], C::default());
    assert_eq!(u[1], -a[1].conj());
    for candidate in [
        [C::new(1., 0.), C::default()],
        [C::default(), C::new(0., 1.)],
        [C::new(-0.3, 0.4), C::new(0.5, 0.)],
    ] {
        assert!(
            (u[0] * a[0] + u[1] * a[1]).re
                <= (candidate[0] * a[0] + candidate[1] * a[1]).re + 1e-15
        );
    }
    let logits = [0.1, -0.7, 1.8, -2.1, 0.3];
    let analytic = train::scalars(&logits, 3, Mode::Mini, 1e-4).unwrap();
    let forward = train::scalars(&logits, 3, Mode::ForwardMini, 1e-4).unwrap();
    for (a, f) in analytic.iter().zip(&forward) {
        assert!((a - f).abs() < 3e-9);
        assert_ne!(*f, 0.);
    }
    assert_eq!(forward.len(), logits.len());
    assert!(forward[3] < 0.);
    assert!(train::scalars(&logits, 5, Mode::Mini, 1e-4).is_err());
}
#[test]
fn exact_absence_skip_equals_dense_search_and_keeps_all_coordinates() {
    let records = [
        Record {
            prefix: &[0, 2, 0],
            target: 1,
        },
        Record {
            prefix: &[2, 2],
            target: 0,
        },
    ];
    let base = vec![vec![0.; 4]; 2];
    let signals: Vec<_> = records
        .iter()
        .map(|r| train::scalars(&base[0], r.target, Mode::ForwardMini, 1e-4).unwrap())
        .collect();
    for family in [Family::Plain, Family::Rotary] {
        let mut sparse = Search {
            base: &base,
            signals: &signals,
            records: &records,
            budget: 16.,
            loss_queries: 0,
            coordinate_trials: [0; 3],
        };
        let mut dense = Search {
            base: &base,
            signals: &signals,
            records: &records,
            budget: 16.,
            loss_queries: 0,
            coordinate_trials: [0; 3],
        };
        let mut start = Atom::identity_start(4, family);
        start.transition[3][0][1] = C::new(1e-100, -1e-100);
        let (a, la) = sparse.improve(start.clone(), 3, true).unwrap();
        let (b, lb) = dense.improve(start.clone(), 3, false).unwrap();
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::to_value(&b).unwrap()
        );
        assert_eq!(la, lb);
        assert_eq!(a.transition[3], start.transition[3]);
        assert!(sparse.loss_queries < dense.loss_queries);
        assert_eq!(
            sparse.coordinate_trials,
            [24, 96, if family == Family::Rotary { 12 } else { 0 }]
        );
        assert_eq!(
            dense.coordinate_trials,
            [24, 192, if family == Family::Rotary { 12 } else { 0 }]
        );
    }
}
#[test]
fn content_initialization_is_matched_and_matrices_are_not_frozen_monomial() {
    let mut r = Random::new(33);
    let mut s = Random::new(33);
    let a = coordinates::random_start(7, Family::Plain, &mut r);
    let b = coordinates::random_start(7, Family::Rotary, &mut s);
    assert_eq!(a.initial, b.initial);
    assert_eq!(a.transition, b.transition);
    assert!(b.frequency.is_some());
    assert_eq!(r.below(1000), s.below(1000));
    let id = Atom::identity_start(2, Family::Rotary);
    let changed = coordinates::Coordinate::Matrix(0, 0, 1)
        .apply(&id, C::new(0.3, 0.4))
        .unwrap();
    assert_ne!(changed.transition[0][0][0], C::default());
    assert_ne!(changed.transition[0][0][1], C::default());
    changed.validate(2, Family::Rotary).unwrap();
    let changed = coordinates::Coordinate::Frequency(1)
        .apply(&changed, C::new(0.7, 0.))
        .unwrap();
    assert_eq!(changed.frequency.unwrap()[1], 0.7);
}
#[test]
fn both_modes_learn_swapped_order_without_witness_weights_and_replay() {
    let records = [
        Record {
            prefix: &[2, 4],
            target: 3,
        },
        Record {
            prefix: &[4, 2],
            target: 5,
        },
    ];
    for family in [Family::Plain, Family::Rotary] {
        for mode in [Mode::Mini, Mode::ForwardMini] {
            let mut m = Model::zero(7, 16., family).unwrap();
            let mut rng = Random::new(31);
            let config = Config {
                mode,
                query_step: 1e-4,
                sweeps: 6,
                line_iterations: 64,
            };
            for _ in 0..6 {
                let receipt = train::step(&mut m, &records, config, &mut rng).unwrap();
                let actual: f64 = records
                    .iter()
                    .map(|r| cross_entropy(&m.logits(r.prefix).unwrap(), r.target).unwrap())
                    .sum::<f64>()
                    / 2.;
                assert!((actual - receipt.after_ce).abs() < 1e-12);
                assert!(receipt.after_ce <= receipt.centre_ce + 1e-12);
                assert_eq!(
                    receipt.bias_loss_queries,
                    if mode == Mode::ForwardMini { 28 } else { 0 }
                );
                assert_eq!(receipt.coordinate_trials[2] > 0, family == Family::Rotary);
            }
            for r in &records {
                assert_eq!(
                    m.next_object(r.prefix).unwrap(),
                    r.target,
                    "family={family:?} mode={mode:?}"
                );
            }
            let loss: f64 = records
                .iter()
                .map(|r| cross_entropy(&m.logits(r.prefix).unwrap(), r.target).unwrap())
                .sum::<f64>()
                / 2.;
            assert!(loss < 0.5, "family={family:?} mode={mode:?} loss={loss}");
            let copied: Model = serde_json::from_slice(&serde_json::to_vec(&m).unwrap()).unwrap();
            for r in &records {
                assert_eq!(
                    m.logits(r.prefix).unwrap(),
                    copied.logits(r.prefix).unwrap()
                );
            }
        }
    }
}
#[test]
fn duplicate_original_records_and_short_tails_are_retained() {
    let e = Example {
        task: "opaque".into(),
        tokens: vec![0, 2, 1],
        targets: vec![-100, 1, 2],
        changes: vec![false, true, true],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    };
    let examples = vec![e];
    let duplicate = original::records(&examples, &[0, 0]).unwrap();
    let tail = original::records(&examples, &[0]).unwrap();
    assert_eq!(duplicate.len(), 4);
    assert_eq!(tail.len(), 2);
    for (a, b) in duplicate[..2].iter().zip(&duplicate[2..]) {
        assert_eq!(a.prefix, b.prefix);
        assert_eq!(a.target, b.target);
    }
    assert_eq!(tail[0].prefix, &[0, 2]);
    assert_eq!(tail[1].prefix, &[0, 2, 1]);
    assert_eq!(tail[0].target, 1);
    assert_eq!(tail[1].target, 2);
}
