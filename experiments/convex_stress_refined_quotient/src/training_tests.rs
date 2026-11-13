use crate::*;
use training::{gradient, score};
fn nontrivial() -> Result<(Model, Vec<[usize; 2]>)> {
    let (d, _, _) = tests::fixture()?;
    let mut m = initialize(&d)?.model;
    for (i, t) in m.table.iter_mut().enumerate() {
        *t = ((i * 17 + 3) % 23) as f64 / 10. - 1.1;
    }
    m.project();
    m.validate()?;
    Ok((m, d.request.input.observations))
}
#[test]
fn both_complex_coordinates_and_table_pullback_match_finite_differences() -> Result<()> {
    let (model, samples) = nontrivial()?;
    let batch = model.encode(&samples[..19])?;
    let g = gradient(&model, &batch)?;
    let active: Vec<_> = (0..2)
        .flat_map(|coordinate| {
            g.head
                .iter()
                .enumerate()
                .filter(move |(i, g)| (i / model.vocabulary) % 2 == coordinate && g.abs() > 1e-3)
                .take(12)
                .map(|(i, _)| i)
        })
        .collect();
    assert_eq!(active.len(), 24);
    assert!(active.iter().any(|i| (i / model.vocabulary) % 2 == 1));
    assert!(active.iter().any(|i| (i / model.vocabulary) % 2 == 0));
    for i in active {
        let mut a = model.clone();
        let mut b = model.clone();
        a.coefficients[i] += 1e-5;
        b.coefficients[i] -= 1e-5;
        let derivative = (score(&a, &batch) - score(&b, &batch)) / 2e-5;
        assert!(
            (derivative - g.head[i]).abs() < 2e-7,
            "head {i}: {derivative} vs {}",
            g.head[i]
        );
    }
    let active: Vec<_> = g
        .table
        .iter()
        .enumerate()
        .filter(|(_, g)| g.abs() > 1e-3)
        .take(12)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(active.len(), 12);
    for i in active {
        let mut a = model.clone();
        let mut b = model.clone();
        a.table[i] += 1e-5;
        b.table[i] -= 1e-5;
        a.materialize();
        b.materialize();
        let derivative = (score(&a, &batch) - score(&b, &batch)) / 2e-5;
        assert!(
            (derivative - g.table[i]).abs() < 2e-7,
            "table {i}: {derivative} vs {}",
            g.table[i]
        );
    }
    Ok(())
}
#[test]
fn weighted_unequal_minibatches_reconstruct_the_full_actual_derivative() -> Result<()> {
    let (model, samples) = nontrivial()?;
    let encoded = model.encode(&samples[..20])?;
    let all = gradient(&model, &encoded)?;
    let a = gradient(&model, &encoded[..7])?;
    let b = gradient(&model, &encoded[7..])?;
    for ((x, y), z) in all.head.iter().zip(&a.head).zip(&b.head) {
        assert!((x - (7. * y + 13. * z) / 20.).abs() < 1e-10);
    }
    for ((x, y), z) in all.table.iter().zip(&a.table).zip(&b.table) {
        assert!((x - (7. * y + 13. * z) / 20.).abs() < 1e-10);
    }
    assert!((all.loss - (7. * a.loss + 13. * b.loss) / 20.).abs() < 1e-10);
    Ok(())
}
#[test]
fn full_and_mini_steps_reduce_nontrivial_ce_and_replay_every_actual_state() -> Result<()> {
    let (model, samples) = nontrivial()?;
    for policy in training::policies(samples.len(), 64)? {
        let out = training::run(&model, &samples, policy.clone())?;
        training::replay(&model, &samples, &out)?;
        if policy.mode == training::Mode::Forward {
            assert_eq!(out.model, model);
        } else {
            assert!(out.final_ce < out.initial_ce - 1e-4);
            assert!(
                out.steps
                    .iter()
                    .any(|s| s.head_gradient_l2 > 0.01 && s.table_gradient_l2 > 0.01)
            );
            assert_ne!(out.model.table, model.table);
            for coordinate in 0..2 {
                assert!(
                    out.model
                        .coefficients
                        .iter()
                        .zip(&model.coefficients)
                        .enumerate()
                        .any(|(i, (after, before))| {
                            (i / model.vocabulary) % 2 == coordinate
                                && (after - before).abs() > 1e-9
                        })
                );
            }
        }
    }
    Ok(())
}
#[test]
fn trajectory_hash_loss_gradient_and_work_mutations_are_rejected() -> Result<()> {
    let (model, samples) = nontrivial()?;
    let policy = training::policies(samples.len(), 64)?[0].clone();
    let out = training::run(&model, &samples, policy)?;
    for kind in 0..8 {
        let mut bad = out.clone();
        match kind {
            0 => bad.steps[0].table_sha256.push('0'),
            1 => bad.steps[0].coefficients_sha256.push('0'),
            2 => bad.steps[0].table_gradient_l2 += 0.01,
            3 => bad.steps[0].batch_ce += 0.01,
            4 => bad.work.gradient_examples -= 1,
            5 => bad.model.coefficients[0] += 0.01,
            6 => bad.steps[0].count -= 1,
            _ => bad.scalar_work_forecast -= 1,
        }
        assert!(training::check(&model, &samples, &bad).is_err());
    }
    Ok(())
}
#[test]
fn finite_policy_validation_and_missing_training_inputs_fail_before_updates() -> Result<()> {
    let (model, samples) = nontrivial()?;
    for rate in [f64::NAN, f64::INFINITY, 0., -1., 1.1] {
        let policy = training::Policy {
            mode: training::Mode::Full,
            updates: 1,
            batch: 64,
            learning_rate: rate,
        };
        assert!(training::run(&model, &samples, policy).is_err());
    }
    assert!(model.encode(&[[0, 1]]).is_err());
    assert!(model.encode(&[[model.n * model.n, 1]]).is_err());
    assert!(gradient(&model, &[]).is_err());
    assert!(training::policies(0, 64).is_err());
    assert!(training::policies(samples.len(), 0).is_err());
    Ok(())
}
