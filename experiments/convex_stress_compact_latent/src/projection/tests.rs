use super::*;
const LIMIT: u64 = 16 * 1024 * 1024;
fn work() -> Work {
    Work::new(2_000_000_000_000).unwrap()
}
fn bits(values: impl IntoIterator<Item = f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}
fn compare(d: &Domain, t: &legacy::projection::Target) -> (f64, usize) {
    let compact = Target::from_legacy(d, t, LIMIT, &mut work()).unwrap();
    assert_eq!(
        bits(t.linear.iter().copied()),
        bits(compact.linear().iter().copied())
    );
    assert_eq!(
        bits(t.free.iter().copied()),
        bits(compact.unknown_rows().iter().copied())
    );
    let (mut wa, mut wb) = (work(), work());
    let a = legacy::projection::project(d, t, &mut wa).unwrap();
    let b = project(d, &compact, LIMIT, &mut wb).unwrap();
    assert_eq!(
        bits(a.parameters.values.iter().copied()),
        bits(b.parameters.values().copied())
    );
    assert_eq!(a.multiplier.to_bits(), b.multiplier.to_bits());
    assert_eq!(
        a.maximum_feasibility_residual.to_bits(),
        b.maximum_feasibility_residual.to_bits()
    );
    assert_eq!(
        a.maximum_kkt_residual.to_bits(),
        b.maximum_kkt_residual.to_bits()
    );
    assert_eq!(a.reduced_objective.to_bits(), b.reduced_objective.to_bits());
    assert_eq!(a.uniform_unknown_programs, b.uniform_unknown_programs);
    assert_eq!(wa.used, wb.used);
    assert_eq!(
        t.objective(d, &a.parameters).unwrap().to_bits(),
        compact.objective(d, &b.parameters).unwrap().to_bits()
    );
    (b.multiplier, b.uniform_unknown_programs)
}
fn domain(classes: usize) -> Domain {
    let mut labels = vec![None; 24];
    for (s, label) in labels[..8].iter_mut().enumerate() {
        *label = Some(s % classes);
    }
    labels[8] = Some(0);
    Domain::from_labels(
        Shape {
            programs: 3,
            states: 8,
            classes,
        },
        labels,
        1.,
        8.,
        &mut work(),
    )
    .unwrap()
}

#[test]
fn arbitrary_complete_proposals_preserve_projection_and_kkt_bits() {
    let mut signs = [false; 3];
    let mut mixed = false;
    for classes in [2, 3, 5, 97] {
        let d = domain(classes);
        for linear in [[-10.; 3], [36., 0., 0.], [100., 200., 300.]] {
            for variant in 0..4 {
                let mut t = legacy::projection::Target {
                    linear: linear.to_vec(),
                    free: vec![0.1; d.unknown.len() * classes],
                };
                for (r, values) in t.free.chunks_mut(classes).enumerate() {
                    match variant {
                        1 if r % 3 == 0 => {
                            for (c, x) in values.iter_mut().enumerate() {
                                *x = (c as f64 - 2.) * 0.125;
                            }
                        }
                        2 => values.fill(if r % 2 == 0 { -0. } else { -0.25 }),
                        3 => {
                            for (c, x) in values.iter_mut().enumerate() {
                                *x = ((r * 17 + c * 13) % 23) as f64 / 17. - 0.5;
                            }
                        }
                        _ => (),
                    }
                }
                let (lambda, uniform) = compare(&d, &t);
                signs[if lambda < 0. {
                    0
                } else if lambda > 0. {
                    2
                } else {
                    1
                }] = true;
                mixed |= uniform > 0 && uniform < d.shape.programs;
            }
        }
    }
    assert_eq!(signs, [true; 3]);
    assert!(mixed);
}

#[test]
fn uniform_prefix_rounding_is_preserved_and_complete_fallback_is_budgeted() {
    let shape = Shape {
        programs: 1,
        states: 4,
        classes: 1000,
    };
    let d = Domain::from_labels(shape, vec![None; 4], 1., 8., &mut work()).unwrap();
    let mut t = legacy::projection::Target {
        linear: vec![4.],
        free: vec![0.1; 4000],
    };
    compare(&d, &t);
    let compact = Target::from_legacy(&d, &t, LIMIT, &mut work()).unwrap();
    let result = project(&d, &compact, 50_000, &mut work()).unwrap();
    assert_ne!(
        result.parameters.masses()[0].to_bits(),
        ((4.0f64 + 4. * 0.1) / (1. + 4. / 1000.)).to_bits()
    );
    assert_eq!(result.uniform_unknown_programs, 1);
    let plans = Plan::prepare(&d, &compact, 50_000, &mut work()).unwrap();
    assert!(Plan::allocation_bytes(&plans) < 4096);
    t.free[17] = 0.2;
    let mixed = Target::from_legacy(&d, &t, LIMIT, &mut work()).unwrap();
    assert!(project(&d, &mixed, 50_000, &mut work()).is_err());
    compare(&d, &t);
    assert_eq!(bits(mixed.unknown_rows().iter().copied()), bits(t.free));
}

#[test]
fn weighted_step_retains_every_unknown_gradient_coordinate() {
    let d = domain(5);
    let mut q = d.initial(4., &mut work()).unwrap();
    let mut gradient = q.clone();
    for (i, x) in q.values.iter_mut().enumerate() {
        *x = (i as f64 - 11.) * 0.1;
    }
    for (i, x) in gradient.values.iter_mut().enumerate() {
        *x = (i % 7) as f64 * 0.17 - 0.31;
    }
    let cq = Parameters::from_legacy(&d, &q, LIMIT, &mut work()).unwrap();
    let cg = Parameters::from_legacy(&d, &gradient, LIMIT, &mut work()).unwrap();
    for rate in [1e-6, 3., 1e6] {
        let (mut wa, mut wb) = (work(), work());
        let a = legacy::projection::Target::step(&d, &q, &gradient, rate, &mut wa).unwrap();
        let b = Target::step(&d, &cq, &cg, rate, LIMIT, &mut wb).unwrap();
        assert_eq!(
            bits(a.linear.iter().copied()),
            bits(b.linear().iter().copied())
        );
        assert_eq!(
            bits(a.free.iter().copied()),
            bits(b.unknown_rows().iter().copied())
        );
        assert_eq!(
            a.objective(&d, &q).unwrap().to_bits(),
            b.objective(&d, &cq).unwrap().to_bits()
        );
        assert_eq!(wa.used, wb.used);
    }
}

#[test]
fn malformed_proposals_limits_and_work_are_honest_refusals() {
    let d = domain(5);
    let mut t = legacy::projection::Target {
        linear: vec![1.; 3],
        free: vec![0.; d.unknown.len() * 5],
    };
    let good = Target::from_legacy(&d, &t, LIMIT, &mut work()).unwrap();
    for cap in [0, 1, 96_000_000_001] {
        assert!(project(&d, &good, cap, &mut work()).is_err());
    }
    assert!(project(&d, &good, LIMIT, &mut Work::new(1).unwrap()).is_err());
    for x in [f64::NAN, f64::INFINITY, 1.1e15] {
        t.free[0] = x;
        assert!(Target::from_legacy(&d, &t, LIMIT, &mut work()).is_err());
    }
    t.free[0] = 0.;
    t.free.pop();
    assert!(Target::from_legacy(&d, &t, LIMIT, &mut work()).is_err());
    let mut bad = d.clone();
    bad.offsets[9] = usize::MAX;
    assert!(project(&bad, &good, LIMIT, &mut work()).is_err());
}
