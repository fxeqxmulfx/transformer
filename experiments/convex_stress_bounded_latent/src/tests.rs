use crate::{
    domain::Domain,
    head::{Certificate, Head, PredictionBudget},
    training::{self, Config, Mode},
    *,
};
mod head;
mod projection;

fn meter() -> Work {
    Work::new(MAX_WORK).unwrap()
}
fn fixture() -> (Domain, Vec<Record>) {
    let records = vec![
        Record {
            states: vec![0, 1],
            label: 0,
        },
        Record {
            states: vec![1, 0],
            label: 1,
        },
        Record {
            states: vec![2, 2],
            label: 0,
        },
    ];
    let shape = Shape {
        programs: 2,
        states: 4,
        classes: 2,
    };
    (Domain::new(shape, &records, 1., 32.).unwrap(), records)
}
fn policy(mode: Mode) -> Config {
    Config {
        mode,
        epochs: if mode == Mode::Forward { 0 } else { 2 },
        batch_size: 2,
        learning_rate: 100.,
        initial_mass: 4.,
    }
}
fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a} != {b}");
}

#[test]
fn every_coordinate_gradient_matches_actual_complex_ce() {
    let (domain, records) = fixture();
    let sh = domain.shape;
    // The derivative is tested away from all observed-row equalities and
    // simplex constraints, including negative and nonuniform decoder entries.
    let mut q = Parameters {
        values: (0..sh.parameters())
            .map(|i| ((i * 17) % 23) as f64 / 16. - 0.6)
            .collect(),
    };
    let mut work = meter();
    let h = Head::build(sh, &q, &mut work).unwrap();
    let (value, g) = training::gradient(&h, &records, &mut work).unwrap();
    close(
        value,
        training::loss(&h, &records, &mut work).unwrap(),
        1e-14,
    );
    for i in 0..sh.parameters() {
        let old = q.values[i];
        q.values[i] = old + 1e-5;
        let plus = training::loss(
            &Head::build(sh, &q, &mut work).unwrap(),
            &records,
            &mut work,
        )
        .unwrap();
        q.values[i] = old - 1e-5;
        let minus = training::loss(
            &Head::build(sh, &q, &mut work).unwrap(),
            &records,
            &mut work,
        )
        .unwrap();
        q.values[i] = old;
        close((plus - minus) / 2e-5, g.values[i], 1e-9);
    }
}

#[test]
fn actual_modes_preserve_records_tail_and_reduce_ce() {
    let (_, base) = fixture();
    let records: Vec<_> = (0..7).map(|i| base[i % 3].clone()).collect();
    let domain = Domain::new(
        Shape {
            programs: 2,
            states: 4,
            classes: 2,
        },
        &records,
        1.,
        32.,
    )
    .unwrap();
    let mut work = meter();
    let forward = training::train(&domain, &records, policy(Mode::Forward), &mut work).unwrap();
    assert!(forward.steps.is_empty());
    assert_eq!(forward.parameters, domain.initial(4.).unwrap());
    for mode in [Mode::Full, Mode::Mini] {
        let fit = training::train(&domain, &records, policy(mode), &mut work).unwrap();
        assert!(fit.final_ce < fit.initial_ce);
        assert!(fit.steps.iter().any(|s| s.parameter_change_l2 > 0.));
        domain.check(&fit.parameters, 1e-8).unwrap();
        let mut counts = vec![0; records.len()];
        for s in &fit.steps {
            for count in &mut counts[s.first_record..s.end_record_exclusive] {
                *count += 1;
            }
            assert!(s.after_ce <= s.before_ce + 1e-12);
        }
        assert!(counts.iter().all(|&n| n == 2));
        if mode == Mode::Mini {
            assert_eq!(
                fit.steps.last().unwrap().end_record_exclusive
                    - fit.steps.last().unwrap().first_record,
                1
            );
        }
        let rho = fit.parameters.values[..2].iter().sum::<f64>();
        close(fit.final_ce, (-rho).exp().ln_1p(), 1e-12);
        assert!(!fit.target_realizability_certified && !fit.exact_real_projection_certified);
    }
    // Repeated records have their original weights, not a deduplicated mean.
    let q = Parameters {
        values: (0..domain.shape.parameters())
            .map(|i| (i % 11) as f64 / 7.)
            .collect(),
    };
    let h = Head::build(domain.shape, &q, &mut work).unwrap();
    let duplicate = vec![base[0].clone(), base[0].clone(), base[1].clone()];
    let value = training::loss(&h, &duplicate, &mut work).unwrap();
    let a = training::loss(&h, &base[..1], &mut work).unwrap();
    let b = training::loss(&h, &base[1..2], &mut work).unwrap();
    close(value, (2. * a + b) / 3., 1e-14);
    assert!((value - (a + b) / 2.).abs() > 1e-4);
}

#[test]
fn unseen_state_ambiguity_survives_identical_observed_ce() {
    let (domain, records) = fixture();
    let sh = domain.shape;
    let mut a = domain.initial(4.).unwrap();
    let mut b = a.clone();
    for p in 0..sh.programs {
        a.values[sh.at(p, 3, 0)] = 2.;
        a.values[sh.at(p, 3, 1)] = 0.;
        b.values[sh.at(p, 3, 0)] = 0.;
        b.values[sh.at(p, 3, 1)] = 2.;
    }
    domain.check(&a, 0.).unwrap();
    domain.check(&b, 0.).unwrap();
    let mut work = meter();
    let ha = Head::build(sh, &a, &mut work).unwrap();
    let hb = Head::build(sh, &b, &mut work).unwrap();
    close(
        training::loss(&ha, &records, &mut work).unwrap(),
        training::loss(&hb, &records, &mut work).unwrap(),
        0.,
    );
    let mut budget = PredictionBudget::new(10).unwrap();
    assert_eq!(
        ha.predict(&[3, 3], &mut budget, &mut work).unwrap().winner,
        0
    );
    assert_eq!(
        hb.predict(&[3, 3], &mut budget, &mut work).unwrap().winner,
        1
    );
}

#[test]
fn refuses_conflicts_extra_constraints_and_resources() {
    let (domain, records) = fixture();
    let mut bad = records.clone();
    bad.push(Record {
        states: records[0].states.clone(),
        label: 1,
    });
    assert!(Domain::new(domain.shape, &bad, 1., 8.).is_err());
    let mut extra = domain.clone();
    extra.labels[3] = Some(0);
    assert!(training::train(&extra, &records, policy(Mode::Forward), &mut meter()).is_err());
    assert!(
        Shape {
            programs: 65536,
            states: 4096,
            classes: 1024
        }
        .admit(0)
        .is_err()
    );
    assert!(
        Shape {
            programs: usize::MAX,
            states: 4,
            classes: 2
        }
        .admit(0)
        .is_err()
    );
    assert!(
        Shape {
            programs: 1,
            states: 5,
            classes: 2
        }
        .admit(0)
        .is_err()
    );
    assert!(
        Head::build(
            domain.shape,
            &domain.initial(4.).unwrap(),
            &mut Work::new(1).unwrap()
        )
        .is_err()
    );
}
