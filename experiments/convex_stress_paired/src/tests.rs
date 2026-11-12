use super::*;
use crate::{
    projection::{Projector, distance, simplex, simplex_even},
    solver::{Solver, latent_gradient},
    transform::Cosine,
};
use convex_stress_relabel::{ideal_loss, loss};

/// The full table over Z/q of an even half, state by state.
pub(crate) fn expand(shape: Shape, half: &[f64]) -> Vec<f64> {
    let q = shape.latent_states();
    half.chunks_exact(shape.orbits)
        .flat_map(|row| (0..q).map(move |z| row[z.min(q - z)]))
        .collect()
}
fn wave(count: usize, salt: usize) -> Vec<f64> {
    (0..count).map(|i| ((i * 17 + salt) as f64).sin()).collect()
}
/// Lean `pairedPhase`: equal weights on the states r(i) + s(j) and -(r(i) + s(j)).
pub(crate) fn paired_phase(shape: Shape, rows: &[usize], columns: &[usize]) -> Vec<f64> {
    let q = shape.latent_states();
    let mut half = vec![0.; shape.cells() * shape.orbits];
    for (j, s) in columns.iter().enumerate() {
        for (i, r) in rows.iter().enumerate() {
            let z = (r + s) % q;
            let orbit = z.min(q - z);
            half[(i + shape.side * j) * shape.orbits + orbit] = 1. / weight(orbit);
        }
    }
    half
}
/// Lean `foldedLabels`: |i - j| on a 3 x 3 grid, five latent states, three classes.
pub(crate) fn folded() -> (Shape, Decoder, Vec<f64>) {
    let shape = Shape {
        side: 3,
        orbits: 3,
        vocab: 3,
    };
    (
        shape,
        Decoder::fold(3),
        paired_phase(shape, &[0, 1, 2], &[0, 4, 3]),
    )
}
fn train_ce(model: &Model, samples: &[Sample]) -> f64 {
    let vocab = model.shape.vocab;
    samples
        .iter()
        .map(|s| loss(&model.output[s.cell * vocab..(s.cell + 1) * vocab], s.label))
        .sum::<f64>()
        / samples.len() as f64
}

#[test]
fn three_sizes_are_separate_and_preflight_refuses_before_allocation() -> Result<()> {
    let shape = Shape {
        side: 4,
        orbits: 3,
        vocab: 7,
    };
    assert_eq!((shape.cells(), shape.latent_states()), (16, 5));
    let decoder = Decoder {
        orbit_class: vec![6, 2, 6],
    };
    let model = Model::uniform(shape, decoder.clone(), 4., 4.)?;
    assert_eq!(model.latent_half.len(), 16 * 3);
    assert_eq!(model.output.len(), 16 * 7);
    // Classes without a latent state stay in the vocabulary with probability zero.
    let expected = [0., 0., 0.4, 0., 0., 0., 0.6];
    assert!(
        model.output[..7]
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() < 1e-15)
    );
    assert_eq!(model.output[..2], [0., 0.]);
    assert_eq!(metrics::Predictor::vocab(&model), 7);
    assert_eq!(metrics::Predictor::logits(&model, &[3, 1]).len(), 7);
    assert_eq!(model.cell(&[9, 3, 1]), 1 + 4 * 3);
    assert_eq!(
        decoder.fibers(7),
        [vec![], vec![], vec![1], vec![], vec![], vec![], vec![0, 2]]
    );
    for z in 0..5 {
        assert_eq!(
            decoder.class_of_state(z),
            decoder.class_of_state((5 - z) % 5)
        );
    }
    assert!(
        Decoder {
            orbit_class: vec![0, 7, 1]
        }
        .validate(shape)
        .is_err()
    );
    assert!(Decoder::fold(3).validate(shape).is_ok());
    assert!(Decoder::fold(4).validate(shape).is_err());
    for bad in [
        Shape { side: 1, ..shape },
        Shape { orbits: 1, ..shape },
        Shape { vocab: 1, ..shape },
        Shape { side: 194, ..shape },
        Shape {
            orbits: 98,
            ..shape
        },
        Shape {
            vocab: 194,
            ..shape
        },
    ] {
        assert!(bad.validate().is_err());
        assert!(memory_bound(bad).is_err());
    }
    let largest = Shape {
        side: 193,
        orbits: 97,
        vocab: 193,
    };
    let bytes = memory_bound(largest)?;
    assert!(bytes < 2 * 1024 * 1024 * 1024);
    assert_eq!(preflight(largest, bytes)?, bytes);
    assert!(preflight(largest, bytes - 1).is_err());
    let original = Shape {
        side: 97,
        orbits: 97,
        vocab: 97,
    };
    assert!(memory_bound(original)? < memory_bound(largest)?);
    Ok(())
}

#[test]
fn cosine_moments_are_the_complex_fourier_sums_of_the_expanded_table() -> Result<()> {
    for (side, orbits) in [(2, 2), (4, 3), (3, 6)] {
        let shape = Shape {
            side,
            orbits,
            vocab: 2,
        };
        let (cells, q) = (shape.cells(), shape.latent_states());
        let half = wave(cells * orbits, side + orbits);
        let full = expand(shape, &half);
        let cosine = Cosine::new(orbits)?;
        let moments = cosine.moments(&half, cells)?;
        for f in 0..q {
            for cell in 0..cells {
                let sum: C = (0..q)
                    .map(|z| {
                        full[cell * q + z]
                            * C::from_polar(1., std::f64::consts::TAU * (f * z) as f64 / q as f64)
                    })
                    .sum();
                assert!(sum.im.abs() < 1e-12);
                assert!((sum.re - moments[f.min(q - f) * cells + cell]).abs() < 1e-12);
            }
        }
        assert!(distance(shape, &half, &cosine.latent(&moments, cells)?) < 1e-12);
        // Parseval in the latent metric: q * ||P||^2 = sum over all frequencies.
        let energy: f64 = moments
            .chunks_exact(cells)
            .enumerate()
            .map(|(f, m)| weight(f) * m.iter().map(|x| x * x).sum::<f64>())
            .sum();
        let zero = vec![0.; half.len()];
        assert!((energy - q as f64 * distance(shape, &half, &zero).powi(2)).abs() < 1e-10);
    }
    assert!(Cosine::new(1).is_err());
    assert!(Cosine::new(98).is_err());
    assert!(Cosine::new(3)?.moments(&[f64::NAN; 12], 4).is_err());
    assert!(Cosine::new(3)?.latent(&[0.; 11], 4).is_err());
    Ok(())
}

#[test]
fn row_projection_is_the_simplex_projection_of_the_expanded_row() {
    let shape = Shape {
        side: 2,
        orbits: 4,
        vocab: 2,
    };
    for salt in 0..20 {
        let half = wave(4, salt);
        let mut out = [0.; 4];
        simplex_even(&half, None, &mut out);
        let full = expand(Shape { side: 1, ..shape }, &half);
        let mut reference = vec![0.; 7];
        simplex(&full, &mut reference);
        assert!(
            expand(Shape { side: 1, ..shape }, &out)
                .iter()
                .zip(&reference)
                .all(|(a, b)| (a - b).abs() < 1e-14)
        );
        simplex_even(&half, Some(&[0, 3]), &mut out);
        assert_eq!((out[1], out[2]), (0., 0.));
        assert!((out[0] + 2. * out[3] - 1.).abs() < 1e-14);
        assert!(out[0] >= 0. && out[3] >= 0.);
        simplex_even(&half, Some(&[2]), &mut out);
        assert_eq!((out[0], out[1], out[3]), (0., 0., 0.));
        assert!((out[2] - 0.5).abs() < 1e-15);
        simplex_even(&half, Some(&[0]), &mut out);
        assert_eq!((out[1], out[2], out[3]), (0., 0., 0.));
        assert!((out[0] - 1.).abs() < 1e-15);
    }
}

#[test]
fn spectral_projection_agrees_with_the_complex_projector_on_the_expanded_table() -> Result<()> {
    // Side 5 equals the five latent states, so the CM5 projector applies to the
    // expanded table; it knows nothing of the symmetry or of the cosine transform.
    let shape = Shape {
        side: 5,
        orbits: 3,
        vocab: 2,
    };
    let projector = Projector::new(shape)?;
    let mut reference = convex_stress_relabel::projection::Projector::new(5, 1)?;
    for (salt, radius, budget) in [(1, 1., 1.5), (2, 0.7, 5.), (3, 5., 0.9), (4, 50., 50.)] {
        let half = wave(25 * 3, salt);
        let (projected, decompositions) = projector.ball(&half, radius, budget)?;
        assert_eq!(decompositions == 0, salt == 4);
        let expected = reference.ball(&expand(shape, &half), radius, budget)?;
        assert!(
            expand(shape, &projected)
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() < 1e-11)
        );
        let norms = projector.norms(&projected)?;
        assert_eq!(norms.len(), 2);
        assert!(
            norms
                .iter()
                .all(|s| s.operator <= radius + 1e-11 && s.nuclear <= budget + 1e-11)
        );
        let all = reference.norms(&expand(shape, &projected))?;
        for f in 1..5_usize {
            let own = norms[f.min(5 - f) - 1];
            assert!((own.operator - all[f - 1].operator).abs() < 1e-11);
            assert!((own.nuclear - all[f - 1].nuclear).abs() < 1e-11);
        }
        let (again, _) = projector.ball(&projected, radius, budget)?;
        assert!(distance(shape, &projected, &again) < 1e-11);
        // Row sums are the constant moment, which no budget constrains.
        for (a, b) in half
            .as_chunks::<3>()
            .0
            .iter()
            .zip(projected.as_chunks::<3>().0)
        {
            assert!((a[0] + 2. * (a[1] + a[2]) - b[0] - 2. * (b[1] + b[2])).abs() < 1e-12);
        }
    }
    assert!(projector.ball(&wave(75, 0), 0., 1.).is_err());
    assert!(projector.ball(&wave(75, 0), 1., f64::NAN).is_err());
    Ok(())
}

#[test]
fn lean_fold_witness_is_feasible_fits_and_reaches_the_ce_lower_bound() -> Result<()> {
    let (shape, decoder, half) = folded();
    let projector = Projector::new(shape)?;
    let model = Model::new(shape, decoder, 3., 3., half)?;
    let norms = model.validate(&projector)?;
    assert!(
        norms
            .iter()
            .all(|s| s.operator <= 3. + 1e-12 && s.nuclear <= 3. + 1e-12)
    );
    let labels = [0, 1, 2, 1, 0, 1, 2, 1, 0];
    let samples: Vec<_> = (0..9)
        .map(|cell| Sample {
            cell,
            label: labels[cell],
        })
        .collect();
    for s in &samples {
        let row = &model.output[s.cell * 3..(s.cell + 1) * 3];
        assert!((0..3).all(|c| row[c] == f64::from(c == s.label)));
    }
    assert!((train_ce(&model, &samples) - ideal_loss(3)).abs() < 1e-15);
    // The same table violates a budget below the proved one.
    let tight = Model::new(
        shape,
        model.decoder.clone(),
        1.,
        1.,
        model.latent_half.clone(),
    )?;
    assert!(tight.validate(&projector).is_err());
    Ok(())
}

#[test]
fn lean_ambiguity_pair_shares_the_observed_fit_and_differs_unseen() -> Result<()> {
    // Lean `paired_unseen_ambiguity`: side 2, three latent states, two classes,
    // cells (0,0), (0,1), (1,0) observed; phases i + j and i - j.
    let shape = Shape {
        side: 2,
        orbits: 2,
        vocab: 2,
    };
    let projector = Projector::new(shape)?;
    let samples = [(0, 0), (2, 1), (1, 1)].map(|(cell, label)| Sample { cell, label });
    let build = |columns: &[usize]| {
        Model::new(
            shape,
            Decoder::fold(2),
            2.,
            2.,
            paired_phase(shape, &[0, 1], columns),
        )
    };
    let (sum, difference) = (build(&[0, 1])?, build(&[0, 2])?);
    for model in [&sum, &difference] {
        model.validate(&projector)?;
        assert!((train_ce(model, &samples) - ideal_loss(2)).abs() < 1e-15);
    }
    assert_eq!(sum.output[3 * 2..], [0., 1.]);
    assert_eq!(difference.output[3 * 2..], [1., 0.]);
    Ok(())
}

#[test]
fn latent_gradient_is_the_derivative_of_full_vocabulary_ce_through_the_decoder() {
    let decoder = Decoder {
        orbit_class: vec![4, 1, 4, 0],
    };
    let shape = Shape {
        side: 2,
        orbits: 4,
        vocab: 5,
    };
    let half: Vec<_> = wave(4, 5).iter().map(|x| 0.2 + 0.1 * x).collect();
    let ce = |half: &[f64]| loss(&decoder.decode(Shape { side: 1, ..shape }, half), 4);
    let mut g = [0.; 4];
    latent_gradient(&decoder, 5, &half, 4, &mut g);
    for orbit in 0..4 {
        let (mut up, mut down) = (half.clone(), half.clone());
        up[orbit] += 1e-6;
        down[orbit] -= 1e-6;
        // A half coordinate moves every state of its orbit.
        let derivative = (ce(&up) - ce(&down)) / 2e-6;
        assert!((derivative - weight(orbit) * g[orbit]).abs() < 1e-9);
    }
    // The classes 2 and 3 have no latent state but stay in the softmax.
    let decoded = decoder.decode(Shape { side: 1, ..shape }, &half);
    assert_eq!((decoded[2], decoded[3]), (0., 0.));
    let mass: f64 = decoded.iter().map(|x| x.exp()).sum();
    assert!((g[1] - decoded[1].exp() / mass).abs() < 1e-15);
    assert!((g[0] - (decoded[4].exp() / mass - 1.)).abs() < 1e-15);
    assert_eq!(g[0], g[2]);
}

pub(crate) fn folded_training() -> Vec<Sample> {
    let labels = [0, 1, 2, 1, 0, 1, 2, 1, 0];
    (0..9)
        .filter(|cell| *cell != 5)
        .map(|cell| Sample {
            cell,
            label: labels[cell],
        })
        .collect()
}

#[test]
fn three_modes_lower_full_vocabulary_ce_on_the_fold_fixture() -> Result<()> {
    let (shape, decoder, _) = folded();
    let samples = folded_training();
    for mode in [Mode::Full, Mode::Mini, Mode::Forward] {
        let initial = Model::uniform(shape, decoder.clone(), 3., 3.)?;
        let start = train_ce(&initial, &samples);
        let mut solver = Solver::new(&initial, samples.clone(), mode)?;
        assert_eq!(solver.step, 1.);
        for update in 0..400 {
            let batch = [update % 8, (update + 3) % 8];
            solver.step(&batch)?;
        }
        let (model, alpha, _) = solver.snapshot()?;
        model.validate(&solver.projector)?;
        let end = train_ce(&model, &samples);
        println!(
            "mode {} start {start:.6} end {end:.6} ideal {:.6} alpha {alpha} residual {:.3e} defect {:.3e} heldout {:?}",
            mode as u8,
            ideal_loss(3),
            solver.residual,
            solver.fit_defect(&model),
            &model.output[5 * 3..6 * 3]
        );
        assert!(end < start);
        assert!(end >= ideal_loss(3) - 1e-12);
        let costs = solver.costs;
        assert_eq!(costs.spectral_projections, 400);
        assert_eq!(costs.row_projections, 800);
        assert_eq!(
            costs.gradient_evaluations,
            match mode {
                Mode::Full => 400 * 8,
                Mode::Mini => 400 * 2,
                Mode::Forward => 0,
            }
        );
        if mode == Mode::Mini {
            assert!(solver.step(&[8]).is_err());
        }
    }
    Ok(())
}

#[test]
fn solver_rejects_invalid_observations_and_unreachable_forward_labels() -> Result<()> {
    let shape = Shape {
        side: 2,
        orbits: 2,
        vocab: 3,
    };
    let model = Model::uniform(shape, Decoder::fold(2), 2., 2.)?;
    let sample = |cell, label| Sample { cell, label };
    assert!(Solver::new(&model, vec![], Mode::Full).is_err());
    assert!(Solver::new(&model, vec![sample(4, 0)], Mode::Full).is_err());
    assert!(Solver::new(&model, vec![sample(0, 3)], Mode::Full).is_err());
    assert!(Solver::new(&model, vec![sample(1, 0), sample(1, 1)], Mode::Full).is_err());
    // Class 2 has no latent state: CE modes may observe it, forward-only cannot fit it.
    assert!(Solver::new(&model, vec![sample(0, 2)], Mode::Full).is_ok());
    assert!(Solver::new(&model, vec![sample(0, 2)], Mode::Mini).is_ok());
    assert!(Solver::new(&model, vec![sample(0, 2)], Mode::Forward).is_err());
    let mut solver = Solver::new(&model, vec![sample(0, 1)], Mode::Full)?;
    assert!(solver.restore_diagnostic_state(vec![0.; 7]).is_err());
    assert!(solver.restore_diagnostic_state(vec![f64::NAN; 8]).is_err());
    Ok(())
}

#[test]
fn saved_tables_are_checked_for_symmetry_mass_and_decoding() -> Result<()> {
    let (shape, decoder, half) = folded();
    let model = Model::new(shape, decoder.clone(), 3., 3., half.clone())?;
    let mut changed = model.clone();
    changed.output[0] = 0.5;
    assert!(changed.validate_shape().is_err());
    let mut changed = model.clone();
    changed.latent_half[1] = 0.25;
    assert!(changed.validate_shape().is_err());
    let mut changed = model.clone();
    changed.decoder.orbit_class = vec![0, 2, 1];
    assert!(changed.validate_shape().is_err());
    let mut changed = model.clone();
    changed.radius = 0.;
    assert!(changed.validate_shape().is_err());
    assert!(Model::new(shape, decoder.clone(), 3., 3., half[1..].to_vec()).is_err());
    let mut negative = half;
    negative[0] = -0.5;
    negative[1] = 0.75;
    assert!(Model::new(shape, decoder, 3., 3., negative).is_err());
    let text = serde_json::to_string(&model)?;
    assert_eq!(serde_json::from_str::<Model>(&text)?, model);
    Ok(())
}

/// Timing and behaviour at the original sizes on a synthetic table; not a result.
#[test]
#[ignore]
fn synthetic_original_size_exploration() -> Result<()> {
    let n = 97;
    let shape = Shape {
        side: n,
        orbits: n,
        vocab: n,
    };
    let q = shape.latent_states();
    let mut rng = convex_stress_relabel::random::Random::new(4242);
    let rows: Vec<usize> = (0..n).map(|_| rng.below(q as u32) as usize).collect();
    let columns: Vec<usize> = (0..n).map(|_| rng.below(q as u32) as usize).collect();
    let label = |cell: usize| {
        let z = (rows[cell % n] + columns[cell / n]) % q;
        z.min(q - z)
    };
    let mut cells: Vec<usize> = (0..n * n).collect();
    rng.shuffle(&mut cells);
    let samples: Vec<_> = cells[..n * n / 2]
        .iter()
        .map(|&cell| Sample {
            cell,
            label: label(cell),
        })
        .collect();
    let mode = match std::env::var("PAIRED_MODE").as_deref() {
        Ok("mini") => Mode::Mini,
        Ok("forward") => Mode::Forward,
        _ => Mode::Full,
    };
    let limit: usize = std::env::var("PAIRED_UPDATES")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(128);
    let initial = Model::uniform(shape, Decoder::fold(n), n as f64, n as f64)?;
    let mut solver = Solver::new(&initial, samples.clone(), mode)?;
    let started = std::time::Instant::now();
    let mut order: Vec<usize> = (0..samples.len()).collect();
    let per_epoch = samples.len().div_ceil(256);
    for update in 0..=limit {
        if update % 19 == 0 || update == limit {
            let (model, alpha, _) = solver.snapshot()?;
            let hits = cells[n * n / 2..]
                .iter()
                .filter(|&&cell| {
                    let row = &model.output[cell * n..(cell + 1) * n];
                    let best = (0..n)
                        .max_by(|&a, &b| row[a].total_cmp(&row[b]).then(b.cmp(&a)))
                        .unwrap();
                    best == label(cell)
                })
                .count();
            println!(
                "update {update} seconds {:.1} excess {:.3e} alpha {alpha:.6} residual {:.3e} heldout {hits}/{} svd {}",
                started.elapsed().as_secs_f64(),
                train_ce(&model, &samples) - ideal_loss(n),
                solver.residual,
                n * n - n * n / 2,
                solver.costs.decompositions
            );
        }
        let within = update % per_epoch;
        if mode == Mode::Mini && within == 0 {
            rng.shuffle(&mut order);
        }
        let batch = if mode == Mode::Mini {
            &order[within * 256..((within + 1) * 256).min(samples.len())]
        } else {
            &order[..]
        };
        solver.step(batch)?;
    }
    Ok(())
}
