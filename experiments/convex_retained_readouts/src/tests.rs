use super::*;
use bilinear::{
    model::cross_entropy,
    train::{Mode, Record},
};
use joint::capacity;
use rotary::{
    atom::{Atom, Family},
    model::Model,
};

// Actual nonzero complex CM104/105 model, not a synthetic coefficient array.
fn witness() -> Model {
    let mut m = Model::zero(2, 16., Family::Rotary).unwrap();
    let mut constant = Atom::identity_start(2, Family::Rotary);
    constant.initial = [C::new(1., 0.), C::default()];
    constant.frequency = Some([0., 0.]);
    constant.output.fill([C::default(), C::new(0., 1.)]);
    let mut ordered = constant.clone();
    ordered.frequency = Some([std::f64::consts::FRAC_PI_2, 0.]);
    ordered.transition[0] = [
        [C::default(), C::new(1., 0.)],
        [C::new(1., 0.), C::default()],
    ];
    ordered.output[0] = [C::default(), C::new(1., 0.)];
    m.mix(constant, 1.).unwrap();
    m.mix(ordered, 0.).unwrap();
    m
}
fn data() -> [Record<'static>; 5] {
    [
        Record {
            prefix: &[1, 0],
            target: 1,
        },
        Record {
            prefix: &[0, 1],
            target: 0,
        },
        Record {
            prefix: &[1, 0],
            target: 1,
        },
        Record {
            prefix: &[],
            target: 0,
        },
        Record {
            prefix: &[1, 0, 1],
            target: 0,
        },
    ]
}
fn loss(m: &Model, records: &[Record<'_>]) -> f64 {
    records
        .iter()
        .map(|r| cross_entropy(&m.logits(r.prefix).unwrap(), r.target).unwrap())
        .sum::<f64>()
        / records.len() as f64
}
fn difference(a: &capacity::Table, b: &capacity::Table) -> f64 {
    a.iter()
        .flatten()
        .flatten()
        .zip(b.iter().flatten().flatten())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0f64, f64::max)
}
#[test]
fn indexed_cache_matches_stored_signals_under_partial_coverage_staleness_duplicates_and_tail() {
    for mode in [Mode::Mini, Mode::ForwardMini] {
        let mut m = witness();
        m.shrink_for_queries(1e-4).unwrap();
        let records = data();
        let mut c = cache::Cache::new(&m, &records).unwrap();
        assert_eq!(c.mean, capacity::zero(&m));
        c.observe(&m, &[0, 1], &records[..2], mode, 1e-4).unwrap();
        assert_eq!(c.coverage(), 2);
        assert_eq!(c.signals[2], vec![0.; 2]);
        assert!(difference(&c.mean, &c.recompute(&m, &records).unwrap()) < 1e-15);
        // Change readouts, making the stored first two signals stale.
        let (target, _) = joint::diagnostic::direction(&m, &c.mean).unwrap();
        m = capacity::mix(&m, &target, 0.1).unwrap();
        m.shrink_for_queries(1e-4).unwrap();
        c.observe(&m, &[2, 3], &records[2..4], mode, 1e-4).unwrap();
        c.observe(&m, &[4], &records[4..], mode, 1e-4).unwrap();
        assert_eq!(c.coverage(), 5);
        assert_eq!(c.refreshed, 5);
        assert!(difference(&c.mean, &c.recompute(&m, &records).unwrap()) < 1e-15);
        assert_ne!(c.signals[0], c.signals[2]); // Identical records at different refresh times.
        c.observe(&m, &[0, 1], &records[..2], mode, 1e-4).unwrap();
        assert_eq!(c.refreshed, 7);
        assert_eq!(c.signals[0], c.signals[2]);
        assert!(difference(&c.mean, &c.recompute(&m, &records).unwrap()) < 1e-15);
        let restored: cache::Cache =
            serde_json::from_slice(&serde_json::to_vec(&c).unwrap()).unwrap();
        assert_eq!(restored.mean, c.mean);
        let before = serde_json::to_vec(&c).unwrap();
        assert!(c.observe(&m, &[1, 0], &records[..2], mode, 1e-4).is_err());
        assert!(
            c.observe(
                &m,
                &[0, 0],
                &[
                    Record {
                        prefix: records[0].prefix,
                        target: records[0].target
                    },
                    Record {
                        prefix: records[0].prefix,
                        target: records[0].target
                    },
                ],
                mode,
                1e-4
            )
            .is_err()
        );
        assert_eq!(serde_json::to_vec(&c).unwrap(), before);
        m.atoms[2].frequency.as_mut().unwrap()[0] += 0.1;
        assert!(c.observe(&m, &[4], &records[4..], mode, 1e-4).is_err());
    }
}
#[test]
fn original_mask_occurrences_are_not_deduplicated_or_renumbered_by_shuffle() {
    let e = baseline::data::Example {
        task: "fixture".into(),
        tokens: vec![1, 0, 1],
        targets: vec![-100, 1, 0],
        changes: vec![false, true, true],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    };
    let mut tail = e.clone();
    tail.tokens = vec![0];
    tail.targets = vec![1];
    let examples = vec![e.clone(), e, tail];
    let index = index::Index::new(&examples).unwrap();
    let (ids, records) = index.batch(&examples, &[1, 0, 2]).unwrap();
    assert_eq!(ids, vec![2, 3, 0, 1, 4]);
    assert_eq!(records.len(), 5);
    assert_eq!(records[0].prefix, &[1, 0]);
    assert_eq!(records[1].target, 0);
    assert_eq!(records[4].prefix, &[0]);
    assert_eq!(index::identity(&records[0]), index::identity(&records[2]));
}
#[test]
fn actual_cm105_finite_minibatch_improvement_worsens_training_mean() {
    let m = witness();
    let records = data();
    let mut target = m.clone();
    target.atoms[1].output[0] = [C::new(-1., 0.), C::default()];
    target.atoms[1].output[1] = [C::new(1., 0.), C::default()];
    target.validate().unwrap();
    let halfway = capacity::mix(&m, &target, 0.5).unwrap();
    let logits = halfway.logits(records[0].prefix).unwrap();
    assert_eq!(logits, vec![-8., 8.]);
    assert!(loss(&halfway, &records[..1]) < loss(&m, &records[..1]));
    assert!(loss(&halfway, &records[..2]) > loss(&m, &records[..2]));
    assert_eq!(halfway.weights[1] * halfway.atoms[1].output[1][1].im, 8.);
    assert_eq!(
        readouts::source::content_hash(&halfway).unwrap(),
        readouts::source::content_hash(&m).unwrap()
    );
}
#[test]
fn cm105_nonzero_error_correction_uses_actual_complete_ce_coefficient() {
    let mut m = witness();
    m.weights = vec![0., 0., 16.];
    m.atoms[2].output[0] = [C::new(0., -1.), C::default()];
    m.atoms[2].output[1] = [C::new(0., 1.), C::default()];
    let mut a = joint::coefficient::batch(&m, &data()[..1], Mode::Mini, 1e-4)
        .unwrap()
        .sum;
    joint::coefficient::mean(&mut a, 1).unwrap();
    let mut h = a.clone();
    for z in h.iter_mut().flatten().flatten() {
        *z *= 0.75;
    }
    let b = diagnostic::error_bound(&m, &a, &h).unwrap();
    assert!((b.actual_gap - 16.).abs() < 1e-12);
    assert!((b.estimated_gap - 12.).abs() < 1e-12);
    assert!((b.coefficient_error_peak - 0.25).abs() < 1e-12);
    assert!((b.correction - 8.).abs() < 1e-12);
    assert!(b.strict_negative_condition);
    assert!(b.estimated_direction_actual_change < 0.);
}
#[test]
fn both_modes_use_full_feasible_queries_and_reconstructed_logits_without_extra_signal_queries() {
    for mode in [Mode::Mini, Mode::ForwardMini] {
        let mut m = witness();
        let records = data();
        let mut c = cache::Cache::new(&m, &records).unwrap();
        let r = update::step(
            &mut m,
            &mut c,
            &[0, 1],
            &records[..2],
            joint::update::Config {
                mode,
                query_step: 1e-4,
                line_iterations: 12,
                step_ceiling: 1. / 16.,
            },
        )
        .unwrap();
        assert_eq!(r.bias_loss_queries, if mode == Mode::Mini { 0 } else { 8 });
        assert_eq!(r.total_ce_evaluations, r.bias_loss_queries + 32 + 6);
        assert!(r.after_ce < r.centre_ce);
        assert!(r.maximum_actual_logit_error < 1e-12);
        assert_eq!(r.real_coordinates, 24);
        assert_eq!(c.coverage(), 2);
        assert_eq!(m.atoms.len(), 3);
        assert!(m.weights[2] > 0.);
    }
}
