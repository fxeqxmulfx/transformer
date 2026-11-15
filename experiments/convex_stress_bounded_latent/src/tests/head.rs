use super::*;
use crate::head::{reference_source, source};

#[test]
fn exact_source_gram_and_dense_complete_products() {
    for i in 0..8 {
        for j in 0..8 {
            let (a, b) = source(i, j);
            let (r, s) = reference_source(i, j);
            assert_eq!((a, b), (r as f64 / 2., s as f64 / 2.));
            let (mut re, mut im) = (0i32, 0i32);
            for k in 0..8 {
                let (a, b) = reference_source(i, k);
                let (c, d) = reference_source(j, k);
                re += a as i32 * c as i32 + b as i32 * d as i32;
                im += b as i32 * c as i32 - a as i32 * d as i32;
            }
            assert_eq!((re, im), (if i == j { 4 } else { 0 }, 0));
        }
    }
    let sh = Shape {
        programs: 2,
        states: 8,
        classes: 3,
    };
    let q = Parameters {
        values: (0..sh.parameters())
            .map(|i| ((i * 13) % 29) as f64 / 8. - 2.)
            .collect(),
    };
    let mut work = meter();
    let h = Head::build(sh, &q, &mut work).unwrap();
    assert!(h.imaginary.iter().any(|v| *v != 0.));
    h.audit(&q, &mut work).unwrap();
    for states in [[1, 6], [5, 2], [7, 7]] {
        let logits = h.logits(&states, &mut work).unwrap();
        for (c, &z) in logits.iter().enumerate() {
            let mut dense = 0.;
            for (p, &s) in states.iter().enumerate() {
                for f in 0..sh.states {
                    let (u, v) = reference_source(s, f);
                    let i = h.at(p, f, c);
                    dense += (u as f64 * h.real[i] - v as f64 * h.imaginary[i]) / 2.;
                }
            }
            close(dense, z, 0.);
            close(
                z,
                q.values[sh.at(0, states[0], c)] + q.values[sh.at(1, states[1], c)],
                0.,
            );
        }
    }
    let mut corrupted = h.clone();
    corrupted.imaginary[1] += 0.01;
    assert!(corrupted.audit(&q, &mut work).is_err());
}

#[test]
fn close_unequal_scores_are_exactly_resolved_and_ties_remain_ties() {
    let sh = Shape {
        programs: 1,
        states: 4,
        classes: 2,
    };
    let mut h = Head {
        shape: sh,
        real: vec![0.; sh.cells()],
        imaginary: vec![0.; sh.cells()],
    };
    h.real[0] = 1e12;
    h.real[1] = 1e12;
    h.real[3] = 1e-5;
    h.validate().unwrap();
    let mut work = meter();
    let mut budget = PredictionBudget::new(3).unwrap();
    let p = h.predict(&[0], &mut budget, &mut work).unwrap();
    assert_eq!(p.certificate, Certificate::ExactDyadics);
    assert_eq!(p.winner, 1);
    assert_eq!(p.maximizers, vec![1]);
    close(p.logits[0], -5e-6, 1e-20);
    assert_eq!(p.logits[1], 0.);
    assert!(!p.target_realizability_certified);
    h.real[2] = 1e-5;
    let tie = h.predict(&[0], &mut budget, &mut work).unwrap();
    assert_eq!(tie.maximizers, vec![0, 1]);
    assert_eq!(tie.winner, 0);
    h.real.fill(0.);
    h.real[1] = f64::from_bits(1);
    assert!(
        h.predict(&[0], &mut budget, &mut work)
            .unwrap_err()
            .to_string()
            .contains("strict gap")
    );
    assert!(
        h.predict(&[0], &mut budget, &mut work)
            .unwrap_err()
            .to_string()
            .contains("budget")
    );
}

#[test]
fn separated_winner_uses_intervals_and_full_column_audit() {
    let (d, _) = fixture();
    let mut work = meter();
    let q = d.initial(4.).unwrap();
    let h = Head::build(d.shape, &q, &mut work).unwrap();
    let p = h
        .predict(&[0, 1], &mut PredictionBudget::new(0).unwrap(), &mut work)
        .unwrap();
    assert_eq!(p.certificate, Certificate::OutwardIntervals);
    assert_eq!(p.winner, 0);
    let audit = h.audit(&q, &mut work).unwrap();
    assert_eq!(audit.coefficients, 2 * d.shape.cells());
    assert!(
        audit
            .column_l1
            .iter()
            .all(|v| *v <= 2. * d.maximum_mass * d.shape.states as f64)
    );
}
