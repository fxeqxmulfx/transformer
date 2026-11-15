use crate::*;
use features::{Bank, Encoder};
fn fixture() -> Input {
    let n = 9;
    Input {
        parent_sha256: "1".repeat(64),
        source_sha256: "2".repeat(64),
        n,
        vocabulary: n,
        blocks: 3,
        route: (0..n * n).map(|cell| cell / n / 3).collect(),
        records: (0..85)
            .map(|id| {
                let cell = id % (n * n);
                let b = cell / n / 3;
                Record {
                    id,
                    cell,
                    label: if cell % n < 3 + b { b + 1 } else { b + 6 },
                }
            })
            .collect(),
    }
}
fn arbitrary() -> Result<Model> {
    let input = fixture();
    let geometry =
        geometry::Geometry::new(&input, geometry::Frame::Grid, geometry::Bank::Diagonals)?;
    let q = 16 + 2 * geometry.bank.dimension();
    Ok(Model {
        input,
        protocol_sha256: protocol_hash()?,
        bank: Bank::Quadratic,
        geometry,
        partition: partition::Partition {
            leaves: 2,
            root: partition::Node::Split {
                feature: 1,
                threshold: -0.5,
                left: Box::new(partition::Node::Leaf { slot: 0 }),
                right: Box::new(partition::Node::Leaf { slot: 1 }),
            },
        },
        common: 8.,
        decoder: vec![vec![1, 6], vec![2, 7], vec![3, 8]],
        theta: (0..q).map(|i| ((i * 7 % 11) as f64 - 5.) / 19.).collect(),
    })
}
fn argmax(z: &[f64]) -> usize {
    (0..z.len())
        .max_by(|&a, &b| z[a].total_cmp(&z[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}
#[test]
fn complete_products_budget_and_input_only_geometry() -> Result<()> {
    let mut model = arbitrary()?;
    for frame in geometry::FRAMES {
        for bank in geometry::BANKS {
            let g = geometry::Geometry::new(&model.input, frame, bank)?;
            let mut renamed = model.input.clone();
            for r in &mut renamed.records {
                r.label = (r.label + 3) % renamed.vocabulary;
            }
            assert_eq!(g, geometry::Geometry::new(&renamed, frame, bank)?);
            assert!(g.descriptors.iter().flatten().all(|x| x.abs() <= 1.));
            let mut corrupt = g.clone();
            corrupt.descriptors[0][0] += 0.1;
            assert!(corrupt.validate(&model.input).is_err());
        }
    }
    model.validate()?;
    let (w, t) = model.weights();
    assert!(model.expanded_mass() <= t + projection::mass(&model.theta, &w) + 1e-9);
    for cell in 0..model.input.n.pow(2) {
        assert!(training::audit(&model, cell)? < 1e-10);
        assert!(
            Encoder::new(model.bank)
                .row(&model.input, cell, 0)
                .phi
                .iter()
                .all(|p| p.norm() <= 1. + 1e-12)
        );
    }
    for x in &mut model.theta {
        *x *= 1e8;
    }
    assert!(model.validate().is_err());
    model.project()?;
    model.validate()?;
    Ok(())
}
#[test]
fn ordinary_ce_gradient_real_tail_and_convex_segment() -> Result<()> {
    let mut model = arbitrary()?;
    let rows = Encoder::new(model.bank).records(&model.input);
    let all: Vec<_> = (0..rows.len()).collect();
    let (ce, g) = training::gradient(&model, &rows, &all)?;
    let mut pooled = vec![0.; g.len()];
    let mut loss = 0.;
    for batch in all.chunks(7) {
        let (c, bg) = training::gradient(&model, &rows, batch)?;
        let weight = batch.len() as f64 / rows.len() as f64;
        loss += c * weight;
        for (p, b) in pooled.iter_mut().zip(bg) {
            *p += b * weight;
        }
    }
    assert_eq!(all.chunks(7).last().unwrap().len(), 1);
    assert!((loss - ce).abs() < 1e-12);
    for (a, b) in pooled.iter().zip(&g) {
        assert!((a - b).abs() < 1e-12);
    }
    for (j, expected) in g.iter().enumerate() {
        let original = model.theta[j];
        let eps = 1e-5;
        model.theta[j] = original + eps;
        let upper = fit::score(&model, &rows, &all).ce_sum / rows.len() as f64;
        model.theta[j] = original - eps;
        let lower = fit::score(&model, &rows, &all).ce_sum / rows.len() as f64;
        model.theta[j] = original;
        assert!(((upper - lower) / (2. * eps) - expected).abs() < 1e-8);
    }
    let mut other = model.clone();
    for x in &mut other.theta {
        *x *= -2.;
    }
    let mut mix = model.clone();
    for (x, y) in mix.theta.iter_mut().zip(&other.theta) {
        *x = 0.3 * *x + 0.7 * y;
    }
    assert!(
        fit::score(&mix, &rows, &all).ce_sum
            <= 0.3 * fit::score(&model, &rows, &all).ce_sum
                + 0.7 * fit::score(&other, &rows, &all).ce_sum
                + 1e-9
    );
    Ok(())
}
#[test]
fn categorical_constructor_invariance_and_exact_forward_replay() -> Result<()> {
    let input = fixture();
    let outcome = training::run(input.clone(), Policy::forward())?;
    assert_eq!(outcome.initialization.ce_gradients, 0);
    assert_eq!(outcome.work.gradient_batches, 0);
    assert_eq!(outcome.initialization.candidates.len(), 2304);
    assert_eq!(outcome.initialization.pooled_scores.len(), 6144);
    assert_eq!(
        outcome
            .initialization
            .fold_record_ids
            .iter()
            .flatten()
            .count(),
        85
    );
    assert!(outcome.model.theta.len() <= 52);
    let all: Vec<_> = (0..input.records.len()).collect();
    let rows = Encoder::new(outcome.model.bank).records(&input);
    assert_eq!(fit::score(&outcome.model, &rows, &all).correct, 85);
    training::replay(&input, &outcome)?;
    let mut renamed = input.clone();
    for r in &mut renamed.records {
        r.label = (r.label + 3) % renamed.vocabulary;
    }
    let renamed = training::run(renamed, Policy::forward())?;
    // Summation order of categorical CE can perturb exact floating-point ties;
    // compare represented predictions and parameters with numerical tolerance.
    assert_eq!(
        outcome.initialization.refit.kind,
        renamed.initialization.refit.kind
    );
    assert_eq!(
        outcome.initialization.refit.geometry,
        renamed.initialization.refit.geometry
    );
    for (a, b) in outcome.model.theta.iter().zip(&renamed.model.theta) {
        assert!((a - b).abs() < 1e-8);
    }
    for cell in 0..input.n.pow(2) {
        assert_eq!(
            (argmax(&outcome.model.logits_cell(cell)?) + 3) % input.vocabulary,
            argmax(&renamed.model.logits_cell(cell)?)
        );
    }
    let mut bad = outcome.clone();
    bad.model.theta[0] += 0.01;
    assert!(training::replay(&input, &bad).is_err());
    Ok(())
}

#[test]
fn actual_full_and_minibatch_schedules_keep_final_parameters() -> Result<()> {
    for mode in [Mode::Full, Mode::Mini] {
        let input = fixture();
        let policy = Policy {
            mode,
            iterations: 2,
            batch: 7,
            learning_rate: 0.1,
        };
        let outcome = training::run(input.clone(), policy)?;
        let expected = if mode == Mode::Full { 2 } else { 26 };
        assert_eq!(outcome.work.gradient_batches, expected);
        assert_eq!(outcome.work.gradient_records, 170);
        assert_eq!(outcome.epoch_ce.len(), 3);
        assert_eq!(outcome.final_ce, *outcome.epoch_ce.last().unwrap());
        if mode == Mode::Full {
            assert!(outcome.epoch_ce.windows(2).all(|w| w[1] <= w[0] + 1e-12));
        }
        assert!(outcome.steps.iter().any(|s| s.gradient_l2 > 0.));
        if mode == Mode::Mini {
            assert_eq!(outcome.steps.last().unwrap().record_ids.len(), 1);
        }
        training::replay(&input, &outcome)?;
    }
    Ok(())
}
#[test]
fn original_dimensions_are_forecast_before_fitting() -> Result<()> {
    let mut input = fixture();
    input.n = 256;
    input.vocabulary = 256;
    input.blocks = 389;
    input.route = (0..65536).map(|i| i % 389).collect();
    input.records = (0..4704)
        .map(|id| Record {
            id,
            cell: id,
            label: id % 256,
        })
        .collect();
    for policy in training::policies()? {
        let b = limits::forecast(&input, &policy)?;
        assert_eq!(b.maximum_stored_parameters, 52);
        assert_eq!(b.candidate_fits, 2305);
        assert!(b.buffer_bytes < 56 * 1024u64.pow(3));
        eprintln!(
            "prospective {:?}: work={} bytes={}",
            policy.mode, b.run_and_replay_kernel_work, b.buffer_bytes
        );
    }
    input.route[0] = 389;
    assert!(limits::forecast(&input, &Policy::forward()).is_err());
    Ok(())
}

#[test]
fn partition_learns_generic_discontinuity_and_preserves_trials() -> Result<()> {
    let points: Vec<_> = (0..11)
        .map(|block| {
            let x = block as f64 / 5. - 1.;
            partition::Point {
                block,
                descriptor: vec![1., x],
                target: 2. * x + if x < 0. { -1. } else { 1. },
            }
        })
        .collect();
    let (tree, eta, attempts) = partition::fit(&points, 0.000001, 1)?;
    assert_eq!(tree.leaves, 2);
    assert_eq!(attempts.len(), 10);
    assert!(attempts.iter().all(|a| a.refusal.is_none()));
    for x in [-0.9, -0.3, 0.3, 0.9] {
        let row = [1., x];
        let leaf = tree.root.leaf(&row);
        let prediction = eta[2 * leaf] + eta[2 * leaf + 1] * x;
        assert!((prediction - (2. * x + if x < 0. { -1. } else { 1. })).abs() < 1e-4);
    }
    let mut bad = tree.clone();
    bad.leaves = 1;
    assert!(bad.validate(2).is_err());
    assert!(partition::fit(&points, 0.000001, 3).is_err());
    Ok(())
}

#[test]
fn compact_ce_is_full_vocabulary_ce_including_absent_classes() -> Result<()> {
    for scale in [0., 1., 100.] {
        let mut model = arbitrary()?;
        for x in &mut model.theta {
            *x *= scale;
        }
        for decoder in [vec![1, 6], vec![1], vec![]] {
            model.decoder[0] = decoder;
            for label in 0..model.input.vocabulary {
                let row = Encoder::new(model.bank).row(&model.input, 0, label);
                let full = model.logits_row(&row);
                let compact = model.compact_score(&row);
                assert!((cross_entropy(&full, label) - compact.0).abs() < 1e-10);
                assert_eq!(argmax(&full), compact.1);
            }
        }
    }
    Ok(())
}
