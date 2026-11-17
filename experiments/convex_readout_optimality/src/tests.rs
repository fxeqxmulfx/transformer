use super::*;
use rotary::atom::{Atom, Family};

fn complex_model() -> Model {
    let mut model = Model::zero(4, 3., Family::Rotary).unwrap();
    for j in 1..=2 {
        let mut a = Atom::identity_start(4, Family::Rotary);
        a.frequency = Some([0.12 * j as f64, -0.07 * j as f64]);
        a.initial = [C::new(0.3, 0.4), C::new(-0.2, 0.5)];
        for (c, m) in a.transition.iter_mut().enumerate() {
            let t = (c + j) as f64;
            *m = [
                [C::from_polar(0.4, t), C::from_polar(0.2, -t)],
                [C::from_polar(0.3, t + 0.2), C::from_polar(0.4, t - 0.7)],
            ];
        }
        for (c, u) in a.output.iter_mut().enumerate() {
            *u = [
                C::from_polar(0.1, c as f64),
                C::from_polar(0.15, 0.4 * j as f64),
            ];
        }
        model.mix(a, 0.4).unwrap();
    }
    model
}
fn records() -> [Record<'static>; 5] {
    [
        Record {
            prefix: &[0, 1],
            target: 2,
        },
        Record {
            prefix: &[1, 0],
            target: 1,
        },
        Record {
            prefix: &[0, 1],
            target: 2,
        },
        Record {
            prefix: &[3, 2, 1],
            target: 0,
        },
        Record {
            prefix: &[],
            target: 3,
        },
    ]
}
fn loss(m: &Model, records: &[Record<'_>]) -> f64 {
    let mut scan = probes::ScoreScan::default();
    scan.observe(m, records).unwrap();
    scan.finish().unwrap().ce
}
fn analyze(m: &Model, records: &[Record<'_>]) -> Analysis {
    let mut scan = Scan::new(m).unwrap();
    for batch in records.chunks(2) {
        scan.observe(batch).unwrap();
    }
    scan.finish().unwrap()
}

#[test]
fn every_complex_coordinate_matches_actual_ce_finite_differences_with_duplicates_and_tail() {
    let model = complex_model();
    let before = serde_json::to_vec(&model).unwrap();
    let data = records();
    let a = analyze(&model, &data);
    assert_eq!(a.summary.records, 5);
    assert_eq!(a.summary.real_coordinates, 48);
    assert_eq!(a.summary.ce_evaluations, 10);
    assert_eq!(a.summary.mean_ce, loss(&model, &data));
    assert!(
        a.coefficient
            .iter()
            .flatten()
            .flatten()
            .any(|z| z.im.abs() > 1e-4)
    );
    let h = 1e-5;
    for j in 0..model.atoms.len() {
        for c in 0..model.objects {
            for k in 0..2 {
                for imaginary in [false, true] {
                    let d = if imaginary {
                        C::new(0., h)
                    } else {
                        C::new(h, 0.)
                    };
                    let mut plus = model.clone();
                    let mut minus = model.clone();
                    plus.atoms[j].output[c][k] += d;
                    minus.atoms[j].output[c][k] -= d;
                    let finite = (loss(&plus, &data) - loss(&minus, &data)) / (2. * h);
                    let exact = if imaginary {
                        -a.coefficient[j][c][k].im
                    } else {
                        a.coefficient[j][c][k].re
                    };
                    assert!(
                        (finite - exact).abs() < 3e-8,
                        "{j}/{c}/{k}/{imaginary}: {finite} vs {exact}"
                    );
                }
            }
        }
    }
    let mut all = Scan::new(&model).unwrap();
    all.observe(&data).unwrap();
    assert_eq!(all.finish().unwrap().coefficient, a.coefficient);
    let without_duplicate = [0, 1, 3, 4].map(|i| Record {
        prefix: data[i].prefix,
        target: data[i].target,
    });
    assert_ne!(
        analyze(&model, &without_duplicate).coefficient,
        a.coefficient
    );
    assert_eq!(before, serde_json::to_vec(&model).unwrap());
}

#[test]
fn support_direction_and_bound_hold_against_actual_feasible_ce_competitors() {
    let model = complex_model();
    let data = records();
    let a = analyze(&model, &data);
    assert!(a.summary.approximate_gap > 1e-3);
    assert!(a.summary.support_identity_error < 1e-14);
    assert!(a.summary.change_identity_error < 1e-14);
    for j in 0..9 {
        let mut v = model.clone();
        for atom in v.atoms.iter_mut().skip(1) {
            for (c, u) in atom.output.iter_mut().enumerate() {
                *u = [
                    C::from_polar(0.4, j as f64 + c as f64),
                    C::from_polar(0.3, 0.7 * j as f64),
                ];
            }
        }
        v.validate().unwrap();
        assert!(loss(&v, &data) + 1e-12 >= a.summary.approximate_ce_lower_bound);
    }
    let eta = 1e-6;
    let v = probes::mix(&model, &a.best, eta).unwrap();
    let finite = (loss(&v, &data) - loss(&model, &data)) / eta;
    assert!((finite - a.summary.predicted_change).abs() < 1e-4);
    assert!(loss(&v, &data) < loss(&model, &data));
    assert_eq!(
        readouts::source::content_hash(&v).unwrap(),
        readouts::source::content_hash(&model).unwrap()
    );
    assert_eq!(v.weights, model.weights);
}

#[test]
fn full_complex_support_includes_zeros_ties_phases_and_all_feasible_competitors() {
    assert_eq!(best_row([C::default(); 2]).unwrap(), [C::default(); 2]);
    let tied = best_row([C::new(0., 1.), C::new(-1., 0.)]).unwrap();
    assert_eq!(tied, [C::new(0., 1.), C::default()]);
    for j in 0..256 {
        let t = j as f64;
        let a = [
            C::from_polar((j % 17) as f64 / 7., 0.17 * t),
            C::from_polar((j % 23) as f64 / 11., -0.31 * t),
        ];
        let v = best_row(a).unwrap();
        let minimum = row(v, a).re;
        assert!((minimum + a[0].norm().max(a[1].norm())).abs() < 2e-15);
        assert!(v[0].norm() + v[1].norm() <= 1. + 1e-15);
        for b in 0..=4 {
            for c in 0..=4 - b {
                let u = [
                    C::from_polar(b as f64 / 4., 0.3 * t),
                    C::from_polar(c as f64 / 4., -0.5 * t),
                ];
                assert!(minimum <= row(u, a).re + 2e-15);
            }
        }
    }
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(best_row([C::new(x, 0.), C::default()]).is_err());
    }
}

#[test]
fn actual_nonzero_cm103_witnesses_have_zero_and_positive_gaps_without_model_mutation() {
    let mut a = Atom::identity_start(2, Family::Rotary);
    a.initial = [C::new(1., 0.), C::default()];
    a.frequency = Some([std::f64::consts::FRAC_PI_2, 0.]);
    a.transition[0] = [
        [C::default(), C::new(1., 0.)],
        [C::new(1., 0.), C::default()],
    ];
    a.output = vec![
        [C::new(0., -1.), C::default()],
        [C::new(0., 1.), C::default()],
    ];
    let mut model = Model::zero(2, 16., Family::Rotary).unwrap();
    model.mix(a, 1.).unwrap();
    let data = [
        Record {
            prefix: &[1, 0],
            target: 1,
        },
        Record {
            prefix: &[1, 0],
            target: 0,
        },
    ];
    let balanced = analyze(&model, &data);
    assert_eq!(balanced.summary.approximate_gap, 0.);
    assert!((balanced.summary.mean_ce - 2f64.ln()).abs() < 1e-15);
    let single = analyze(&model, &data[..1]);
    assert_eq!(single.summary.approximate_gap, 16.);
    assert_eq!(single.summary.predicted_change, -16.);
    assert!(loss(&single.best, &data[..1]) < 1e-12);
    assert!(Scan::new(&model).unwrap().finish().is_err());
    assert!(Scan::new(&model).unwrap().observe(&[]).is_err());
    assert!(
        Scan::new(&model)
            .unwrap()
            .observe(&[Record {
                prefix: &[0],
                target: 2
            }])
            .is_err()
    );
    assert!(
        Scan::new(&model)
            .unwrap()
            .observe(&[Record {
                prefix: &[2],
                target: 0
            }])
            .is_err()
    );
    assert!(probes::mix(&model, &single.best, f64::NAN).is_err());
}
