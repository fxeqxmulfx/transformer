use super::*;
fn fixture() -> Result<(Input, Model)> {
    let request = crate::tests::pair_request();
    let discovery = discovery::run(request.clone())?;
    let mut model = model::initialize(&discovery)?.model;
    // Start away from the nearly interpolating initializer so a zero or wrongly
    // signed derivative cannot pass by changing an already negligible loss.
    for c in &mut model.coefficients {
        *c *= 0.02;
    }
    Ok((request.input, model))
}
#[test]
fn full_vocabulary_complex_ce_derivatives_match_differences_and_batch_weights() -> Result<()> {
    let (input, model) = fixture()?;
    let samples = model.observations(&input)?;
    let (loss, g) = gradient(&model, &samples)?;
    let (independent_loss, independent) = check::reference(&model, &samples);
    assert!((loss - independent_loss).abs() < 1e-12);
    assert!(
        g.iter()
            .zip(independent)
            .all(|(a, b)| (a - b).abs() < 1e-12)
    );
    let eps = 1e-5;
    for (i, &expected) in g.iter().enumerate() {
        let mut plus = model.clone();
        let mut minus = model.clone();
        plus.coefficients[i] += eps;
        minus.coefficients[i] -= eps;
        let finite = (score(&plus, &samples) - score(&minus, &samples)) / (2. * eps);
        assert!((finite - expected).abs() < 1e-9, "coordinate {i}");
    }
    let (_, a) = gradient(&model, &samples[..1])?;
    let (_, b) = gradient(&model, &samples[1..])?;
    for i in 0..g.len() {
        assert!((g[i] - (a[i] + 3. * b[i]) / 4.).abs() < 1e-12);
    }
    assert!(
        g.iter()
            .skip(model.vocabulary)
            .take(model.vocabulary)
            .all(|x| x.abs() < 1e-15)
    );
    assert!(
        g.iter()
            .enumerate()
            .any(|(i, v)| i / model.vocabulary % 2 == 1 && v.abs() > 0.01)
    );
    Ok(())
}
#[test]
fn real_updates_and_forward_identity_replay_with_cyclic_minibatches() -> Result<()> {
    let (input, model) = fixture()?;
    for mut policy in policies() {
        if policy.mode != Mode::Forward {
            policy.updates = 12;
        }
        policy.batch = 3;
        let out = run(&model, &input, policy.clone())?;
        replay(&model, &input, &out)?;
        let bytes = serde_json::to_vec(&out)?;
        assert_eq!(serde_json::from_slice::<Outcome>(&bytes)?, out);
        if policy.mode == Mode::Forward {
            assert_eq!(out.model, model);
            assert_eq!(out.work.gradient_examples, 0);
            assert_eq!(out.work.coefficient_updates, 0);
            assert_eq!(out.work.projection_columns, 0);
            assert_eq!(out.initial_ce, out.final_ce);
        } else {
            assert!(out.final_ce < out.initial_ce - 0.05);
            assert_ne!(out.model.coefficients, model.coefficients);
            assert!(out.work.gradient_examples > 0);
            if policy.mode == Mode::Mini {
                assert_eq!(
                    out.steps
                        .iter()
                        .take(4)
                        .map(|s| s.start)
                        .collect::<Vec<_>>(),
                    [0, 3, 2, 1]
                );
                assert_eq!(
                    out.work.score_examples,
                    2 * input.observations.len() + 12 * 3
                );
            }
        }
    }
    Ok(())
}
#[test]
fn independent_trajectory_check_rejects_changed_state_labels_and_work() -> Result<()> {
    let (input, model) = fixture()?;
    let mut policy = policies()[1].clone();
    policy.updates = 4;
    policy.batch = 3;
    let out = run(&model, &input, policy)?;
    for i in 0..7 {
        let mut bad = out.clone();
        match i {
            0 => bad.model.coefficients[0] += 0.01,
            1 => bad.steps[0].gradient_l2 += 0.01,
            2 => bad.steps[1].start = 0,
            3 => bad.work.gradient_examples += 1,
            4 => bad.final_ce += 0.01,
            5 => bad.model.blocks[0].decoder.labels[0] = 0,
            _ => bad.scalar_work_forecast += 1,
        }
        assert!(check(&model, &input, &bad).is_err());
    }
    let mut altered = input;
    altered.observations[0].label = 0;
    assert!(check(&model, &altered, &out).is_err());
    Ok(())
}
