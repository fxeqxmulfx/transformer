use crate::{
    C,
    affine::Affine,
    matrix,
    solver::{self, Options},
};

/// True frequencies/permutations are fixture generators only, never solver inputs.
fn features(n: usize, frequencies: &[usize], permutation: &[usize]) -> Vec<C> {
    let r = frequencies.len();
    let mut q = vec![C::default(); r * r];
    for j in 0..r {
        q[j + r * j] = C::from_polar(1., 0.37 * (j + 1) as f64);
    }
    if r >= 2 {
        let a = q[0];
        let b = q[1 + r];
        q[0] = a * 0.6;
        q[1] = a * 0.8;
        q[r] = b * (-0.8);
        q[1 + r] = b * 0.6;
    }
    let mut out = vec![C::default(); n * r];
    for j in 0..r {
        for x in 0..n {
            for i in 0..r {
                out[x + n * j] += C::from_polar(
                    1.,
                    std::f64::consts::TAU * (permutation[x] * frequencies[i]) as f64 / n as f64,
                ) * q[i + r * j];
            }
        }
    }
    out
}

#[test]
fn eigensolve_coordinates_and_complex_objectives_match_direct_algebra() {
    let a = vec![
        C::new(2., 0.),
        C::new(0., -1.),
        C::new(0., 1.),
        C::new(2., 0.),
    ];
    let e = matrix::eigen(&a, 2).unwrap();
    assert!((e.values[0] - 1.).abs() < 1e-13 && (e.values[1] - 3.).abs() < 1e-13);
    let rec = matrix::reconstruct(&e, &e.values);
    for (a, b) in a.iter().zip(&rec) {
        assert!((*a - *b).norm() < 1e-13);
    }
    for j in 0..2 {
        for k in 0..2 {
            let inner: C = (0..2)
                .map(|i| e.vectors[i + 2 * j].conj() * e.vectors[i + 2 * k])
                .sum();
            assert!((inner - C::new(f64::from(j == k), 0.)).norm() < 1e-13);
        }
    }
    let v = matrix::pack(&a, 2);
    let back = matrix::unpack(&v, 2);
    for (a, b) in a.iter().zip(back) {
        assert!((*a - b).norm() < 1e-13);
    }
    let b = vec![
        C::new(1., 0.2),
        C::new(-0.4, 0.7),
        C::new(0.6, -0.9),
        C::new(2., 0.1),
    ];
    let direct = matrix::reading(&b, 2, 2, &a, 0, 1);
    assert!((matrix::dot(&matrix::objective(&b, 2, 2, 0, 1, false), &v) - direct.re).abs() < 1e-13);
    assert!((matrix::dot(&matrix::objective(&b, 2, 2, 0, 1, true), &v) - direct.im).abs() < 1e-13);
    let density = matrix::density(&a, 2).unwrap();
    let de = matrix::eigen(&density, 2).unwrap();
    assert!(de.values[0].abs() < 1e-13 && (de.values[1] - 1.).abs() < 1e-13);
    let mut wrong = a;
    wrong[1] = C::new(0., 2.);
    assert!(matrix::eigen(&wrong, 2).is_err());
}

#[test]
fn affine_projection_handles_dependent_and_inconsistent_readings() {
    let rows = vec![vec![1., 1.], vec![2., 2.]];
    let a = Affine::new(rows.clone(), vec![1., 2.], 1e-11).unwrap();
    assert_eq!(a.rank, 1);
    let (p, y) = a.project(&[3., -1.]);
    assert!(a.residual(&p) < 1e-13);
    assert!((p[0] - 2.5).abs() < 1e-13 && (p[1] + 1.5).abs() < 1e-13);
    let correction = a.transpose(&y);
    assert!((correction[0] - 0.5).abs() < 1e-13);
    let bad = Affine::new(rows, vec![1., 3.], 1e-11).unwrap();
    assert!(bad.consistency > 0.1);
}

#[test]
fn exact_rotated_subspace_recovers_an_actual_character_without_receiving_its_index() {
    let n = 13;
    let r = 3;
    let permutation: Vec<_> = (0..n).map(|x| (5 * x + 3) % n).collect();
    let b = features(n, &[1, 3, 9], &permutation);
    let out = solver::extract(&b, n, r, [0, 1], Options::default()).unwrap();
    assert!(
        out.accepted,
        "{} {:?}",
        out.reason,
        out.stages
            .iter()
            .map(|s| s.trace.last())
            .collect::<Vec<_>>()
    );
    let phase = out.phase_column.as_ref().unwrap();
    let error = [1, 3, 9]
        .iter()
        .map(|&k| {
            (0..n)
                .map(|x| {
                    let truth = C::from_polar(
                        1.,
                        std::f64::consts::TAU
                            * k as f64
                            * (permutation[x] as f64 - permutation[0] as f64)
                            / n as f64,
                    );
                    (phase[x] - truth).norm()
                })
                .fold(0., f64::max)
        })
        .fold(f64::INFINITY, f64::min);
    assert!(error < 1e-6, "character error {error}");
    for s in &out.stages {
        assert!(s.accepted && s.trace.len() <= out.options.iterations);
    }
    solver::replay(&b, &out).unwrap();
    crate::audit::endpoint(&b, &out).unwrap();
    let bytes = serde_json::to_vec(&out).unwrap();
    let restored = serde_json::from_slice(&bytes).unwrap();
    solver::replay(&b, &restored).unwrap();
    let mut bad = out.clone();
    bad.stages[0].matrix[0].re += 1e-4;
    assert!(solver::replay(&b, &bad).is_err());
    assert!(crate::audit::endpoint(&b, &bad).is_err());
    let mut bad = out;
    bad.stages[1].multipliers[0] += 1e-4;
    assert!(solver::replay(&b, &bad).is_err());
    assert!(crate::audit::endpoint(&b, &bad).is_err());
}

#[test]
fn imaginary_stage_breaks_a_real_tie_and_does_not_impose_rank_one_in_stage_one() {
    let n = 13;
    let b = features(n, &[1, 12], &(0..n).collect::<Vec<_>>());
    let out = solver::extract(&b, n, 2, [0, 1], Options::default()).unwrap();
    assert!(
        out.accepted,
        "{} {:?}",
        out.reason,
        out.stages
            .iter()
            .map(|s| s.trace.last())
            .collect::<Vec<_>>()
    );
    let mixed = matrix::eigen(&out.stages[0].matrix, 2).unwrap();
    assert!(
        mixed.values[0] > 0.49 && mixed.values[1] < 0.51,
        "{:?}",
        mixed.values
    );
    let phase = out.phase_column.unwrap();
    let score = phase[1].conj();
    assert!(
        score.im > 0.4,
        "second objective must choose the positive imaginary score"
    );
}

#[test]
fn capped_perturbed_and_invalid_inputs_remain_explicit_failures() {
    let b = features(13, &[1, 3, 9], &(0..13).collect::<Vec<_>>());
    let out = solver::extract(
        &b,
        13,
        3,
        [0, 1],
        Options {
            iterations: 1,
            ..Options::default()
        },
    )
    .unwrap();
    assert!(!out.accepted && !out.stages[0].accepted);
    assert_eq!(out.stages[0].trace.len(), 1);
    let mut perturbed = b.clone();
    perturbed[0] += 0.1;
    let refused = solver::extract(&perturbed, 13, 3, [0, 1], Options::default()).unwrap();
    assert_eq!(refused.reason, "features_not_centered_and_normalized");
    assert!(solver::extract(&b, 13, 3, [0, 0], Options::default()).is_err());
    assert!(
        solver::extract(
            &b,
            13,
            3,
            [0, 1],
            Options {
                iterations: usize::MAX,
                ..Options::default()
            }
        )
        .is_err()
    );
    assert!(crate::memory_bound(194, 3).is_err());
    assert!(crate::memory_bound(13, 9).is_err());
    assert!(crate::memory_bound(193, 8).unwrap() < 80 * 1024 * 1024);
}

#[test]
fn collision_density_is_feasible_and_has_nonzero_cross_terms() {
    let b = vec![
        C::new(1., 0.),
        C::new(1., 0.),
        C::new(1., 0.),
        C::new(-1., 0.),
    ];
    let m = vec![
        C::new(0.5, 0.),
        C::new(0., -0.5),
        C::new(0., 0.5),
        C::new(0.5, 0.),
    ];
    let e = matrix::eigen(&m, 2).unwrap();
    assert!(e.values[0].abs() < 1e-13 && (e.values[1] - 1.).abs() < 1e-13);
    for x in 0..2 {
        assert!((matrix::reading(&b, 2, 2, &m, x, x) - 1.).norm() < 1e-13);
    }
    assert!(m[1].norm() > 0.4);
}

#[test]
fn normalized_noisy_subspace_is_not_silently_treated_as_an_exact_fourier_space() {
    let n = 13;
    let r = 3;
    let mut b = features(n, &[1, 3, 9], &(0..n).collect::<Vec<_>>());
    let mut rng = crate::random::Random::new(997);
    for z in &mut b {
        *z += C::new(rng.gauss(0., 0.02), rng.gauss(0., 0.02));
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
                let value = b[x + n * k] * dot;
                b[x + n * j] -= value;
            }
        }
        let scale = ((0..n).map(|x| b[x + n * j].norm_sqr()).sum::<f64>() / n as f64).sqrt();
        for x in 0..n {
            b[x + n * j] /= scale;
        }
    }
    let out = solver::extract(&b, n, r, [0, 1], Options::default()).unwrap();
    assert!(out.centered_residual < 1e-12 && out.gram_residual < 1e-12);
    assert!(
        !out.accepted,
        "noise must not be presented as exact mode extraction"
    );
    assert_eq!(out.stages[0].reason, "inconsistent_affine_constraints");
    assert!(out.stages[0].affine_consistency > 1e-6);
    crate::audit::endpoint(&b, &out).unwrap();
    solver::replay(&b, &out).unwrap();
}

#[test]
fn self_contained_artifact_binds_inputs_audit_and_every_stage() {
    use crate::artifact::{self, Input};
    let input = Input {
        n: 13,
        rank: 2,
        features: features(13, &[1, 12], &(0..13).collect::<Vec<_>>()),
        rows: [0, 1],
        options: Options::default(),
    };
    let saved = artifact::run(input.clone()).unwrap();
    let bytes = serde_json::to_vec(&saved).unwrap();
    let restored = serde_json::from_slice(&bytes).unwrap();
    artifact::replay(&input, &restored).unwrap();
    let mut wrong = saved.clone();
    wrong.input.features[0] += 0.01;
    assert!(artifact::replay(&input, &wrong).is_err());
    let mut wrong = saved.clone();
    wrong.raw_audit["accepted"] = false.into();
    assert!(artifact::replay(&input, &wrong).is_err());
    let mut wrong = saved;
    wrong.extraction.stages[0].trace[0].dual_gap += 0.01;
    assert!(artifact::replay(&input, &wrong).is_err());
}
