use super::*;
use bilinear::train::{Mode, Record};
use rotary::{
    atom::{Atom, Family},
    model::Model,
};

fn witness() -> Model {
    let mut m = Model::zero(2, 16., Family::Rotary).unwrap();
    let mut a = Atom::identity_start(2, Family::Rotary);
    a.initial = [C::new(1., 0.), C::default()];
    a.frequency = Some([0., 0.]);
    a.output.fill([C::default(), C::new(0., 1.)]);
    let mut b = a.clone();
    b.frequency = Some([std::f64::consts::FRAC_PI_2, 0.]);
    b.transition[0] = [
        [C::default(), C::new(1., 0.)],
        [C::new(1., 0.), C::default()],
    ];
    b.output[0] = [C::default(), C::new(1., 0.)];
    m.mix(a, 1.).unwrap();
    m.mix(b, 0.).unwrap();
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
fn cache(model: &Model, records: &[Record<'_>], mode: Mode) -> cache::Cache {
    let mut p = retained::cache::Cache::new(model, records).unwrap();
    p.observe(
        model,
        &(0..records.len()).collect::<Vec<_>>(),
        records,
        mode,
        1e-4,
    )
    .unwrap();
    cache::Cache::from_parent(model, records, p).unwrap()
}
#[test]
fn changed_states_with_unchanged_signals_require_the_complete_product_refresh() {
    for mode in [Mode::Mini, Mode::ForwardMini] {
        let mut base = witness();
        base.shrink_for_queries(1e-4).unwrap();
        let records = data();
        let mut c = cache(&base, &records, mode);
        let old_mean = c.mean.clone();
        let old_signal = c.signals.clone();
        let mut changed = base.clone();
        changed.atoms[2].frequency = Some([0., 0.]);
        changed.validate().unwrap();
        c.observe(&changed, &[0, 1], &records[..2], mode, 1e-4, 1)
            .unwrap();
        assert_eq!(c.signals, old_signal);
        assert_ne!(c.mean, old_mean);
        assert_eq!(c.identities[0], c.identities[2]);
        assert_eq!(c.origins, vec![1, 1, 0, 0, 0]);
        assert!(c.recomputation_error(&changed).unwrap() < 1e-15);
        let history = vec![history::Entry {
            version: 1,
            component: Some(2),
            atom: Some(changed.atoms[2].clone()),
            content_sha256: readouts::source::content_hash(&changed).unwrap(),
        }];
        assert_eq!(history::verify(&base, &records, &c, &history).unwrap(), 5);
        let report = diagnostic::analyze(&changed, &c, &records, 2, |_| Ok(())).unwrap();
        assert!(report["cm106"]["signal_error_mean"].as_f64().unwrap() < 1e-8);
        assert!(report["cm106"]["state_error_mean"].as_f64().unwrap() > 0.1);
        // The two stale nonempty records have opposite labels and identical
        // states, so their coefficient errors cancel even though both drifted.
        assert!(report["cm105"]["coefficient_error_peak"].as_f64().unwrap() < 1e-10);
        assert_eq!(report["records"], 5);
        c.origins[0] = 0;
        assert!(
            history::verify(&base, &records, &c, &history)
                .unwrap_err()
                .to_string()
                .contains("recorded content origin")
        );
        c.origins[0] = 1;
        c.observe(&changed, &[2, 3], &records[2..4], mode, 1e-4, 1)
            .unwrap();
        // Refreshing just one member removes that cancellation. The remaining
        // stale product has peak error |1+i| / 5, with unchanged CE signals.
        let partial = diagnostic::analyze(&changed, &c, &records, 2, |_| Ok(())).unwrap();
        assert!(
            (partial["cm105"]["coefficient_error_peak"].as_f64().unwrap() - 2f64.sqrt() / 5.).abs()
                < 1e-8
        );
        c.observe(&changed, &[4], &records[4..], mode, 1e-4, 1)
            .unwrap();
        assert_eq!(c.refreshes, 5);
        assert_eq!(c.origins, vec![1; 5]);
        assert_eq!(history::verify(&base, &records, &c, &history).unwrap(), 5);
        let mut independent = joint::coefficient::batch(&changed, &records, mode, 1e-4)
            .unwrap()
            .sum;
        joint::coefficient::mean(&mut independent, 5).unwrap();
        for (a, b) in independent
            .iter()
            .flatten()
            .flatten()
            .zip(c.mean.iter().flatten().flatten())
        {
            assert!((*a - *b).norm() < 1e-15);
        }
    }
}
#[test]
fn original_indices_versions_and_initial_cache_are_checked() {
    let mut m = witness();
    m.shrink_for_queries(1e-4).unwrap();
    let records = data();
    let mut c = cache(&m, &records, Mode::Mini);
    let before = serde_json::to_vec(&c).unwrap();
    assert!(
        c.observe(&m, &[1, 0], &records[..2], Mode::Mini, 1e-4, 1)
            .is_err()
    );
    assert!(
        c.observe(&m, &[0, 0], &records[..2], Mode::Mini, 1e-4, 1)
            .is_err()
    );
    assert_eq!(before, serde_json::to_vec(&c).unwrap());
    c.origins[0] = 2;
    assert!(
        c.observe(&m, &[4], &records[4..], Mode::Mini, 1e-4, 1)
            .is_err()
    );
    let mut parent = retained::cache::Cache::new(&m, &records).unwrap();
    parent.mean[1][0][0] = C::new(0.01, 0.);
    assert!(cache::Cache::from_parent(&m, &records, parent).is_err());
}
#[test]
fn full_content_sweep_preserves_feasibility_and_checks_actual_ce() {
    let mut m = witness();
    m.weights = vec![0., 0., 16.];
    m.atoms[2].frequency = Some([0., 0.]);
    let receipt = content::step(
        &mut m,
        &data()[..2],
        2,
        content::Config {
            radius: 0.05,
            frequency_radius: std::f64::consts::FRAC_PI_2,
        },
    )
    .unwrap();
    assert_eq!(receipt.coordinate_trials, [8, 32, 4]);
    assert_eq!(receipt.total_ce_evaluations, (8 + 32 + 4 + 2) * 2);
    assert!(receipt.content_changed);
    assert!(receipt.after_ce < receipt.before_ce);
    assert!(receipt.maximum_actual_logit_error < 1e-12);
    m.validate().unwrap();
    assert_eq!(m.atoms.len(), 3);
    assert_eq!(m.weights, vec![0., 0., 16.]);
    let row = m.atoms[2].transition[0];
    let single = [Record {
        prefix: &[1, 1],
        target: 1,
    }];
    let r = content::step(
        &mut m,
        &single,
        2,
        content::Config {
            radius: 0.01,
            frequency_radius: 0.01,
        },
    )
    .unwrap();
    assert_eq!(r.coordinate_trials, [8, 16, 4]);
    assert_eq!(m.atoms[2].transition[0], row);
}
#[test]
fn all_complex_initial_matrix_and_frequency_trials_remain_available_and_bounded() {
    let m = witness();
    let a = &m.atoms[2];
    for k in 0..2 {
        let b = content::Coordinate::Initial(k)
            .apply(a, C::new(0.3, -0.2))
            .unwrap();
        b.validate(2, Family::Rotary).unwrap();
        let b = content::Coordinate::Frequency(k)
            .apply(a, C::new(0.1, 0.))
            .unwrap();
        b.validate(2, Family::Rotary).unwrap();
        assert_ne!(b.frequency, a.frequency);
        for o in 0..2 {
            for j in 0..2 {
                let b = content::Coordinate::Matrix(o, k, j)
                    .apply(a, C::new(0.3, -0.2))
                    .unwrap();
                b.validate(2, Family::Rotary).unwrap();
                assert_ne!(b.transition[o][k], a.transition[o][k]);
            }
        }
    }
    assert!(
        content::Coordinate::Frequency(0)
            .apply(a, C::new(0., 0.1))
            .is_err()
    );
}
#[test]
fn both_minibatch_modes_keep_complete_actual_query_accounting() {
    for mode in [Mode::Mini, Mode::ForwardMini] {
        let mut m = witness();
        let records = data();
        let parent = retained::cache::Cache::new(&m, &records).unwrap();
        let mut c = cache::Cache::from_parent(&m, &records, parent).unwrap();
        let r = update::step(
            &mut m,
            &mut c,
            &[0, 1],
            &records[..2],
            joint::update::Config {
                mode,
                query_step: 1e-4,
                line_iterations: 12,
                step_ceiling: 1. / 1024.,
            },
            0,
        )
        .unwrap();
        assert_eq!(r.bias_loss_queries, if mode == Mode::Mini { 0 } else { 8 });
        assert_eq!(r.total_ce_evaluations, r.bias_loss_queries + 32 + 6);
        assert!(r.after_ce < r.centre_ce);
        assert!(c.recomputation_error(&m).unwrap() < 1e-15);
        assert_eq!(r.real_coordinates, 24);
        assert_eq!(c.coverage(), 2);
    }
}
