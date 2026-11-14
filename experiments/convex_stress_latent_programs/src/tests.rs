use crate::*;
mod boundary;
mod gradients;

fn fixture() -> Input {
    let mut rows = vec![];
    for length in 1..=3 {
        for word in 0..3usize.pow(length) {
            let mut tokens = vec![0];
            let mut n = word;
            for _ in 0..length {
                tokens.push(1 + n % 3);
                n /= 3;
            }
            let sums: Vec<i64> = (1..=tokens.len())
                .map(|end| {
                    tokens[..end]
                        .iter()
                        .map(|&t| i64::from(t == 1) - i64::from(t == 2))
                        .sum()
                })
                .collect();
            let label = if sums.iter().any(|s| *s < 0) {
                6
            } else if sums.last() == Some(&0) {
                4
            } else {
                5
            };
            rows.push(Observation { tokens, label });
        }
    }
    Input {
        vocabulary: 8,
        context: 4,
        observations: (0..67).map(|i| rows[i % rows.len()].clone()).collect(),
    }
}

#[test]
fn exhaustive_independent_summary_interpreters_and_state_bounds() -> Result<()> {
    let bank = program::catalogue(3)?;
    assert_eq!(bank.len(), 54);
    for length in 1..=5 {
        for word in 0..3usize.pow(length) {
            let mut n = word;
            let tokens: Vec<_> = (0..length)
                .map(|_| {
                    let t = n % 3;
                    n /= 3;
                    t
                })
                .collect();
            for p in &bank {
                let v = p.run(&tokens);
                assert_eq!(v.state, p.interpret(&tokens));
                assert!(v.state < STATES);
                assert_eq!(v.positions, tokens.len() as u64);
            }
        }
    }
    assert!(program::catalogue(164).is_err());
    Ok(())
}

#[test]
fn learned_many_to_one_complete_heads_all_modes_and_real_short_tails() -> Result<()> {
    let input = fixture();
    let report = latent::run(input.clone())?;
    latent::check(&input, &report)?;
    assert!(report.initialization_error.is_none());
    assert!(!report.target_realizability_certified);
    assert!(
        report
            .discovery
            .retained
            .iter()
            .any(|p| { p.decoder.iter().flatten().filter(|&&y| y == 6).count() > 1 })
    );
    for (endpoint, (batches, examples)) in
        report.endpoints.iter().zip([(16, 1072), (12, 268), (0, 0)])
    {
        let out = endpoint.training.as_ref().unwrap();
        assert_eq!(out.work.gradient_batches, batches);
        assert_eq!(out.work.gradient_examples, examples);
        let model = &out.model;
        for (tokens, target) in [(vec![0, 1, 3, 1, 2, 3, 2], 4), (vec![0, 2, 1, 1, 3], 6)] {
            let states: Vec<_> = model.states(&tokens).iter().map(|v| v.state).collect();
            assert!(model.labels(&states).iter().all(|&y| y == Some(target)));
            let z = model.logits_states(&states);
            assert_eq!(argmax(&z), target);
            assert!((ce(&z, target) - (1. + 7. * (-24f64).exp()).ln()).abs() < 1e-10);
        }
        assert!(model.head.code.iter().any(|v| v.im.abs() > 0.1));
        assert!(model.head.coefficients.iter().any(|b| {
            b.iter()
                .enumerate()
                .any(|(i, v)| (i / 8) % 2 == 1 && v.abs() > 1.)
        }));
    }
    let mini = report.endpoints[1].training.as_ref().unwrap();
    assert_eq!(
        mini.steps.iter().map(|s| s.count).collect::<Vec<_>>(),
        vec![32, 32, 3, 32, 32, 3, 32, 32, 3, 32, 32, 3]
    );
    let forward = report.endpoints[2].training.as_ref().unwrap();
    assert_eq!(forward.model, *report.initial.as_ref().unwrap());
    assert_eq!(forward.work.cache_products, 0);
    assert_eq!(forward.work.derivative_terms, 0);
    assert_eq!(forward.work.projected_steps, 0);
    Ok(())
}

#[test]
fn unknown_rows_conflicts_caps_and_source_mutations_are_visible() -> Result<()> {
    let input = Input {
        vocabulary: 3,
        context: 1,
        observations: vec![Observation {
            tokens: vec![0],
            label: 2,
        }],
    };
    let found = discovery::run(input.clone())?;
    discovery::check(&input, &found)?;
    assert_eq!(found.retained.len(), 54);
    let model = model::Model::new(&found)?;
    let states: Vec<_> = model.states(&[1]).iter().map(|s| s.state).collect();
    assert!(model.labels(&states).contains(&None));
    let mut changed = found.clone();
    changed.retained[0].source_rows.fill(None);
    assert!(discovery::check(&input, &changed).is_err());
    let mut collision = input.clone();
    collision.observations.push(Observation {
        tokens: vec![0],
        label: 1,
    });
    let conflict = latent::run(collision.clone())?;
    latent::check(&collision, &conflict)?;
    assert!(conflict.discovery.retained.is_empty());
    assert_eq!(conflict.endpoints.len(), 3);
    assert!(
        conflict
            .endpoints
            .iter()
            .all(|e| e.error.is_some() && e.training.is_none())
    );
    let cap = discovery::run(Input {
        vocabulary: 8,
        ..input
    })?;
    assert_eq!(cap.error.as_deref(), Some("retained latent program cap128"));
    Ok(())
}

#[test]
fn coefficient_decoder_source_and_work_mutations_refuse() -> Result<()> {
    let input = fixture();
    let report = latent::run(input.clone())?;
    let round: latent::Report = serde_json::from_slice(&serde_json::to_vec(&report)?)?;
    assert_eq!(report, round);
    let mut bad = report.clone();
    bad.initial.as_mut().unwrap().head.coefficients[0][1] += 0.1;
    assert!(latent::check(&input, &bad).is_err());
    let mut bad = report.clone();
    bad.initial.as_mut().unwrap().head.decoders[0][0] = Some(7);
    assert!(latent::check(&input, &bad).is_err());
    let mut bad = report.clone();
    bad.endpoints[2]
        .training
        .as_mut()
        .unwrap()
        .work
        .gradient_examples = 1;
    assert!(latent::check(&input, &bad).is_err());
    let mut changed = input.clone();
    changed.observations[0].label = 7;
    assert!(latent::check(&changed, &report).is_err());
    Ok(())
}
