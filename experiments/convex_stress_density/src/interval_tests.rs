use crate::{
    C,
    interval::{self, Input, Options},
    interval_audit, matrix,
};

fn input(n: usize, frequencies: &[usize], epsilon: f64) -> Input {
    Input {
        n,
        rank: frequencies.len(),
        features: frequencies
            .iter()
            .flat_map(|k| {
                (0..n).map(move |x| {
                    C::from_polar(1., std::f64::consts::TAU * (x * k) as f64 / n as f64)
                })
            })
            .collect(),
        rows: [1, 0],
        options: Options {
            model_tolerance: epsilon,
            ..Options::default()
        },
    }
}

fn rotated() -> Input {
    let mut out = input(13, &[1, 3, 9], 0.);
    out.rows = [0, 1];
    let old = out.features.clone();
    for x in 0..13 {
        let p = (5 * x + 3) % 13;
        out.features[x] = (old[p] * 0.6 + old[p + 13] * 0.8) * C::from_polar(1., 0.37);
        out.features[x + 13] = (old[p] * -0.8 + old[p + 13] * 0.6) * C::from_polar(1., 0.74);
        out.features[x + 26] = old[p + 26] * C::from_polar(1., 1.11);
    }
    out
}

#[test]
fn interval_zero_recovers_a_complex_rotated_and_row_permuted_mode() {
    let original = rotated();
    let out = interval_audit::run(original.clone()).unwrap();
    assert!(out.outcome.accepted, "{:?}", out.outcome.trace.last());
    let column = out.outcome.kernel_column.as_ref().unwrap();
    let error = [1, 3, 9]
        .iter()
        .map(|&k| {
            (0..13)
                .map(|x| {
                    let expected = C::from_polar(
                        1.,
                        std::f64::consts::TAU * k as f64 * (((5 * x + 3) % 13) as f64 - 3.) / 13.,
                    );
                    (column[x] - expected).norm()
                })
                .fold(0., f64::max)
        })
        .fold(f64::INFINITY, f64::min);
    assert!(error < 1e-6, "character error {error}");
    interval_audit::replay(&original, &out).unwrap();
}

#[test]
fn centered_normalized_noisy_subspace_can_be_admitted_as_an_empirical_candidate() {
    let mut original = rotated();
    let (n, r) = (original.n, original.rank);
    let b = &mut original.features;
    let mut random = crate::random::Random::new(311);
    for z in b.iter_mut() {
        *z += C::new(random.gauss(0., 0.02), random.gauss(0., 0.02));
    }
    for j in 0..r {
        let mean = (0..n).map(|x| b[x + n * j]).sum::<C>() / n as f64;
        for x in 0..n {
            b[x + n * j] -= mean;
        }
        for k in 0..j {
            let dot = (0..n)
                .map(|x| b[x + n * k].conj() * b[x + n * j])
                .sum::<C>()
                / n as f64;
            for x in 0..n {
                let z = b[x + n * k] * dot;
                b[x + n * j] -= z;
            }
        }
        let scale = ((0..n).map(|x| b[x + n * j].norm_sqr()).sum::<f64>() / n as f64).sqrt();
        for x in 0..n {
            b[x + n * j] /= scale;
        }
    }
    let strict =
        crate::solver::extract(b, n, r, original.rows, crate::solver::Options::default()).unwrap();
    assert!(strict.centered_residual < 1e-12 && strict.gram_residual < 1e-12);
    assert_eq!(strict.stages[0].reason, "inconsistent_affine_constraints");
    original.options.model_tolerance = 0.1;
    let out = interval_audit::run(original.clone()).unwrap();
    assert!(out.outcome.accepted, "{:?}", out.outcome.trace.last());
    assert!(out.outcome.trace.last().unwrap().interval_violation <= 1e-8);
    assert_eq!(out.raw_audit["certified_mode"], false);
    interval_audit::replay(&original, &out).unwrap();
}

#[test]
fn interval_domain_admits_a_real_perturbation_without_loosening_numerical_thresholds() {
    let mut original = input(4, &[1], 0.21);
    original.features.iter_mut().for_each(|z| *z *= 1.1);
    let a = interval_audit::run(original.clone()).unwrap();
    assert!(a.outcome.accepted, "{:?}", a.outcome.trace.last());
    assert!(a.raw_audit["interval_violation"].as_f64().unwrap() < 1e-12);
    assert_eq!(a.raw_audit["certified_mode"], false);
    interval_audit::replay(&original, &a).unwrap();
    original.options.model_tolerance = 0.;
    original.options.iterations = 32;
    let b = interval_audit::run(original.clone()).unwrap();
    assert!(!b.outcome.accepted);
    assert!(b.raw_audit["interval_violation"].as_f64().unwrap() > 0.2);
    assert_eq!(b.outcome.trace.len(), 32);
    interval_audit::replay(&original, &b).unwrap();
}

#[test]
fn exact_zero_interval_selects_a_mode_and_rejects_raw_and_replay_tampering() {
    let original = input(4, &[0, 1], 0.);
    let a = interval_audit::run(original.clone()).unwrap();
    assert!(a.outcome.accepted, "{:?}", a.outcome.trace.last());
    assert!(a.outcome.purity_error < 1e-7);
    assert!((a.outcome.trace.last().unwrap().objective - 1.).abs() < 1e-7);
    assert!(
        a.outcome
            .kernel_column
            .as_ref()
            .unwrap()
            .iter()
            .all(|z| (*z - 1.).norm() < 1e-7)
    );
    interval_audit::replay(&original, &a).unwrap();
    let restored = serde_json::from_slice(&serde_json::to_vec(&a).unwrap()).unwrap();
    interval_audit::replay(&original, &restored).unwrap();
    let mut bad = a.clone();
    bad.outcome.matrix[0].re += 1e-4;
    assert!(interval_audit::endpoint(&original, &bad.outcome).is_err());
    assert!(interval_audit::replay(&original, &bad).is_err());
    let mut bad = a.clone();
    bad.outcome.multipliers[0] += 0.1;
    assert!(interval_audit::endpoint(&original, &bad.outcome).is_err());
    let mut bad = a.clone();
    bad.outcome.trace[0].objective += 0.01;
    assert!(interval_audit::replay(&original, &bad).is_err());
    let mut changed = original;
    changed.features[0] += 0.001;
    assert!(interval_audit::replay(&changed, &a).is_err());
}

#[test]
fn real_ties_keep_mixed_densities_and_do_not_claim_purity() {
    let original = input(13, &[1, 12], 0.);
    let a = interval_audit::run(original.clone()).unwrap();
    assert!(a.outcome.accepted, "{:?}", a.outcome.trace.last());
    let e = matrix::eigen(&a.outcome.matrix, 2).unwrap();
    assert!(e.values[0] > 0.49 && e.values[1] < 0.51);
    assert!(a.outcome.purity_error > 0.49);
    interval_audit::replay(&original, &a).unwrap();
}

#[test]
fn bounded_interval_refusals_and_support_signs_are_explicit() {
    let mut original = input(4, &[0, 1], 0.1);
    let a = interval::BoxRows::new(&original);
    let y = [1., -2., 3., -4.];
    // Each raw row has norm two: upper=.55, lower=.45. The support
    // at [1,-2,3,-4] is 4*.55-6*.45=-.5, not a centered absolute-value sum.
    assert!((a.support(&y) + 0.5).abs() < 1e-13);
    original.options.iterations = 1;
    let out = interval_audit::run(original.clone()).unwrap();
    assert!(!out.outcome.accepted);
    assert_eq!(out.outcome.trace.len(), 1);
    interval_audit::replay(&original, &out).unwrap();
    original.options.model_tolerance = f64::NAN;
    assert!(interval::solve(original.clone()).is_err());
    original.options.model_tolerance = 1.1;
    assert!(interval::solve(original.clone()).is_err());
    original.options.model_tolerance = 0.1;
    original.options.iterations = 16_385;
    assert!(interval::solve(original).is_err());
}
