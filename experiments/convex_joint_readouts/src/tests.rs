use super::*;
use bilinear::train::{Mode, Record};
use rotary::{
    atom::{Atom, Family},
    model::Model,
};

fn witness() -> Model {
    let mut model = Model::zero(2, 16., Family::Rotary).unwrap();
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
    model.mix(constant, 1.).unwrap();
    model.mix(ordered, 0.).unwrap();
    model
}
fn data() -> [Record<'static>; 2] {
    [
        Record {
            prefix: &[1, 0],
            target: 1,
        },
        Record {
            prefix: &[0, 1],
            target: 0,
        },
    ]
}
fn loss(model: &Model, records: &[Record<'_>]) -> f64 {
    let mut scan = optimality::probes::ScoreScan::default();
    scan.observe(model, records).unwrap();
    scan.finish().unwrap().ce
}
fn analyze(model: &Model, records: &[Record<'_>]) -> diagnostic::Analysis {
    let mut scan = diagnostic::Scan::new(model).unwrap();
    for batch in records.chunks(2) {
        scan.observe(batch).unwrap();
    }
    scan.finish().unwrap()
}

#[test]
fn actual_cm104_counterexample_recovers_capacity_from_zero_weight() {
    let model = witness();
    let a = analyze(&model, &data());
    assert_eq!(a.fixed.approximate_gap, 0.);
    assert_eq!(a.support.selected, 2);
    assert!((a.support.approximate_gap - 8. * 2f64.sqrt()).abs() < 1e-12);
    assert_eq!(model.weights[2], 0.);
    assert_eq!(a.best.weights[2], 16.);
    assert!(loss(&a.best, &data()) < 1e-8);
    assert_eq!(model.atoms.len(), a.best.atoms.len());
    assert_eq!(
        readouts::source::content_hash(&model).unwrap(),
        readouts::source::content_hash(&a.best).unwrap()
    );
    for eta in [0., 0.001, 0.1, 0.5, 1.] {
        let mixed = capacity::mix(&model, &a.best, eta).unwrap();
        for r in data() {
            let x = model.logits(r.prefix).unwrap();
            let y = a.best.logits(r.prefix).unwrap();
            let z = mixed.logits(r.prefix).unwrap();
            for c in 0..2 {
                assert!((z[c] - ((1. - eta) * x[c] + eta * y[c])).abs() < 1e-13);
            }
        }
        assert!(loss(&mixed, &data()) + 1e-12 >= a.fixed.mean_ce - a.support.approximate_gap);
    }
}

#[test]
fn both_modes_use_complete_feasible_queries_and_actual_reconstructed_logits() {
    for mode in [Mode::Mini, Mode::ForwardMini] {
        let mut model = witness();
        let before = readouts::source::content_hash(&model).unwrap();
        let receipt = update::step(
            &mut model,
            &data(),
            update::Config {
                mode,
                query_step: 1e-4,
                line_iterations: 12,
                step_ceiling: 1.,
            },
        )
        .unwrap();
        assert_eq!(receipt.selected_component, 2);
        assert_eq!(receipt.selected_weight_before, 0.);
        assert!(receipt.selected_weight_after > 0.);
        assert!(receipt.after_ce < receipt.centre_ce);
        assert!(receipt.maximum_actual_logit_error < 1e-12);
        assert_eq!(
            receipt.bias_loss_queries,
            if mode == Mode::Mini { 0 } else { 8 }
        );
        assert_eq!(
            receipt.total_ce_evaluations,
            receipt.bias_loss_queries + 32 + 6
        );
        assert_eq!(before, readouts::source::content_hash(&model).unwrap());
        assert_eq!(model.atoms.len(), 3);
    }
}

#[test]
fn every_weighted_complex_coordinate_matches_actual_ce_with_duplicates_and_short_tail() {
    let mut model = witness();
    model.weights = vec![14., 1., 1.];
    for a in model.atoms.iter_mut().skip(1) {
        a.initial = [C::new(0.3, 0.4), C::new(-0.2, 0.5)];
        a.output.fill([C::new(0.1, -0.2), C::new(0.2, 0.1)]);
    }
    let records = [
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
    ];
    let a = analyze(&model, &records);
    let rows = capacity::weighted(&model);
    let h = 1e-5;
    for j in 1..model.atoms.len() {
        for c in 0..model.objects {
            for k in 0..2 {
                for imaginary in [false, true] {
                    let d = if imaginary {
                        C::new(0., h)
                    } else {
                        C::new(h, 0.)
                    };
                    let mut plus = rows.clone();
                    let mut minus = rows.clone();
                    plus[j][c][k] += d;
                    minus[j][c][k] -= d;
                    let p = capacity::reconstruct(&model, &plus).unwrap();
                    let m = capacity::reconstruct(&model, &minus).unwrap();
                    let numerical = (loss(&p, &records) - loss(&m, &records)) / (2. * h);
                    let expected = if imaginary {
                        -a.coefficient[j][c][k].im
                    } else {
                        a.coefficient[j][c][k].re
                    };
                    assert!(
                        (numerical - expected).abs() < 3e-8,
                        "{j}/{c}/{k}/{imaginary}"
                    );
                }
            }
        }
    }
    assert_eq!(a.coefficient[0], vec![[C::default(); 2]; 2]);
    let mut full = diagnostic::Scan::new(&model).unwrap();
    full.observe(&records).unwrap();
    for (a, b) in a.coefficient.iter().flatten().flatten().zip(
        full.finish()
            .unwrap()
            .coefficient
            .iter()
            .flatten()
            .flatten(),
    ) {
        assert!((*a - *b).norm() < 1e-15);
    }
    assert_eq!(a.fixed.records, 5);
    assert_ne!(a.coefficient, analyze(&model, &records[..4]).coefficient);
    assert_ne!(
        a.coefficient,
        analyze(
            &model,
            &[0, 1, 3, 4].map(|i| Record {
                prefix: records[i].prefix,
                target: records[i].target
            })
        )
        .coefficient
    );
}

#[test]
fn reconstruction_preserves_zero_blocks_and_rejects_incomplete_or_infeasible_inputs() {
    let model = witness();
    let rows = capacity::weighted(&model);
    let restored = capacity::reconstruct(&model, &rows).unwrap();
    assert_eq!(restored.weights, model.weights);
    assert_eq!(restored.atoms[2].output, vec![[C::default(); 2]; 2]);
    assert_eq!(
        readouts::source::content_hash(&restored).unwrap(),
        readouts::source::content_hash(&model).unwrap()
    );
    let mut bad = rows.clone();
    bad.pop();
    assert!(capacity::reconstruct(&model, &bad).is_err());
    let mut bad = rows.clone();
    bad[1][0][0] = C::new(32., 0.);
    assert!(capacity::reconstruct(&model, &bad).is_err());
    let mut bad = rows;
    bad[0][0][0] = C::new(0.1, 0.);
    assert!(capacity::reconstruct(&model, &bad).is_err());
    assert!(capacity::mix(&model, &restored, f64::NAN).is_err());
    assert!(diagnostic::Scan::new(&model).unwrap().finish().is_err());
    assert!(coefficient::batch(&model, &data(), Mode::ForwardMini, 1e-4).is_err());
}
