use super::*;
use crate::projection::project;

#[test]
fn known_analytic_simplex_and_mass_solutions() {
    let sh = Shape {
        programs: 1,
        states: 4,
        classes: 2,
    };
    let records = [Record {
        states: vec![0],
        label: 0,
    }];
    let fixed = Domain::new(sh, &records, 1., 1.).unwrap();
    let q = Parameters {
        values: vec![9., 7., 2., 2., -1., -2., -3., 0.2, 0.4],
    };
    let out = project(&fixed, &q, &mut meter()).unwrap();
    for (a, b) in out
        .parameters
        .values
        .iter()
        .zip([1., 1., 0., 1., 0., 1., 0., 0.4, 0.6])
    {
        close(*a, b, 1e-8);
    }
    assert!(!out.exact_real_certificate);
    assert_eq!(out.uniform_unknown_programs, 0);
    let free = Domain::new(sh, &records, 0.1, 8.).unwrap();
    let q = Parameters {
        values: vec![2., 3., 9., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    };
    let out = project(&free, &q, &mut meter()).unwrap();
    close(out.parameters.values[0], 13. / 7., 1e-12);
    assert_eq!(out.multiplier, 0.);
    assert_eq!(out.uniform_unknown_programs, 1);
}

#[test]
fn upper_lower_and_zero_mass_programs() {
    let sh = Shape {
        programs: 2,
        states: 4,
        classes: 2,
    };
    let d = Domain::new(
        sh,
        &[Record {
            states: vec![0, 0],
            label: 0,
        }],
        1.,
        3.,
    )
    .unwrap();
    let mut q = Parameters {
        values: vec![0.; sh.parameters()],
    };
    q.values[0] = 30.;
    q.values[1] = -30.;
    let upper = project(&d, &q, &mut meter()).unwrap();
    close(upper.parameters.values[0], 3., 1e-12);
    close(upper.parameters.values[1], 0., 0.);
    assert!(upper.multiplier > 0.);
    q.values.fill(-3.);
    let lower = project(&d, &q, &mut meter()).unwrap();
    close(lower.parameters.values[0], 0.5, 1e-12);
    close(lower.parameters.values[1], 0.5, 1e-12);
    assert!(lower.multiplier < 0.);
}

fn random(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}
#[test]
fn variational_inequality_and_idempotence_for_nonuniform_full_domain() {
    let sh = Shape {
        programs: 2,
        states: 4,
        classes: 3,
    };
    let records = [
        Record {
            states: vec![0, 1],
            label: 0,
        },
        Record {
            states: vec![1, 2],
            label: 2,
        },
    ];
    let d = Domain::new(sh, &records, 0.5, 3.).unwrap();
    let mut state = 7;
    let mut work = meter();
    for _ in 0..24 {
        let target = Parameters {
            values: (0..sh.parameters())
                .map(|_| random(&mut state) * 12. - 6.)
                .collect(),
        };
        let projected = project(&d, &target, &mut work).unwrap();
        let q = &projected.parameters;
        assert_eq!(projected.uniform_unknown_programs, 0);
        let again = project(&d, q, &mut work).unwrap();
        for (a, b) in q.values.iter().zip(&again.parameters.values) {
            close(*a, *b, 1e-8);
        }
        for _ in 0..20 {
            let mut probe = d.initial(1.).unwrap();
            let mass = 0.5 + 2.5 * random(&mut state);
            let split = random(&mut state);
            probe.values[0] = mass * split;
            probe.values[1] = mass * (1. - split);
            for p in 0..sh.programs {
                for s in 0..sh.states {
                    let r = [random(&mut state), random(&mut state), random(&mut state)];
                    let total = r.iter().sum::<f64>();
                    for (c, &v) in r.iter().enumerate() {
                        probe.values[sh.at(p, s, c)] = match d.labels[p * sh.states + s] {
                            Some(y) => {
                                if c == y {
                                    probe.values[p]
                                } else {
                                    0.
                                }
                            }
                            None => probe.values[p] * v / total,
                        };
                    }
                }
            }
            d.check(&probe, 1e-8).unwrap();
            let dot = target
                .values
                .iter()
                .zip(&q.values)
                .zip(&probe.values)
                .map(|((t, x), z)| (t - x) * (z - x))
                .sum::<f64>();
            assert!(dot <= 1e-7, "projection variational inequality {dot}");
        }
    }
}
