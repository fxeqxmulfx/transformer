use crate::*;
mod cohort;
mod gradients;

fn fixture() -> Input {
    let mut rows = vec![];
    for a in 4..=6 {
        for b in 4..=6 {
            if a != b {
                for reverse in [false, true] {
                    for query in [2, 3] {
                        let mut tokens = if reverse {
                            vec![1, 3, b, 2, a]
                        } else {
                            vec![1, 2, a, 3, b]
                        };
                        tokens.push(query);
                        rows.push(Observation {
                            tokens,
                            label: if query == 2 { a } else { b },
                        });
                    }
                }
            }
        }
    }
    Input {
        vocabulary: 8,
        context: 6,
        observations: (0..67).map(|i| rows[i % rows.len()].clone()).collect(),
    }
}

#[test]
fn every_short_program_matches_independent_interpreter() -> Result<()> {
    let bank = program::catalogue(3, 5)?;
    assert_eq!(bank.len(), 133);
    for length in 1..=5 {
        for word in 0..3usize.pow(length) {
            let mut n = word;
            let tokens: Vec<_> = (0..length)
                .map(|_| {
                    let x = n % 3;
                    n /= 3;
                    x
                })
                .collect();
            for p in &bank {
                let actual = p.run(&tokens);
                assert_eq!((actual.state, actual.missing), p.interpret(&tokens));
            }
        }
    }
    // The final query must not masquerade as a completed stored value.
    let walk = program::Program::Walk {
        hops: 1,
        offset: 1,
        marker: None,
        latest: false,
    };
    assert_eq!(walk.interpret(&[2, 2]), (0, true));
    assert_eq!(walk.interpret(&[2, 1, 2]), (1, false));
    Ok(())
}

#[test]
fn all_modes_copy_unobserved_raw_symbol_and_replay_real_complex_states() -> Result<()> {
    let input = fixture();
    let report = binding::run(input.clone())?;
    binding::check(&input, &report)?;
    assert_eq!(report.discovery.retained.len(), 2);
    assert!(input.observations.iter().all(|s| s.label != 7));
    let expected = [(16, 1072), (12, 268), (0, 0)];
    for (endpoint, (batches, examples)) in report.endpoints.iter().zip(expected) {
        assert!(endpoint.error.is_none());
        let outcome = endpoint.training.as_ref().unwrap();
        assert_eq!(outcome.work.gradient_batches, batches);
        assert_eq!(outcome.work.gradient_examples, examples);
        let z = outcome.model.logits_tokens(&[1, 4, 7, 2, 5, 4]);
        assert_eq!(argmax(&z), 7);
        assert!((ce(&z, 7) - (1. + 7. * (-24f64).exp()).ln()).abs() < 1e-10);
        assert!(outcome.model.head.code.iter().any(|z| z.im.abs() > 0.1));
        assert!(
            outcome
                .model
                .head
                .coefficients
                .iter()
                .enumerate()
                .any(|(i, v)| (i / 8) % 2 == 1 && v.abs() > 1.)
        );
        assert!(
            outcome
                .model
                .states(&[1, 4, 7, 2, 5, 4])
                .iter()
                .all(|v| v.state == 7 && !v.missing)
        );
    }
    let mini = report.endpoints[1].training.as_ref().unwrap();
    assert_eq!(
        mini.steps.iter().map(|s| s.count).collect::<Vec<_>>(),
        vec![32, 32, 3, 32, 32, 3, 32, 32, 3, 32, 32, 3]
    );
    assert_eq!(
        mini.steps.iter().map(|s| s.start).collect::<Vec<_>>(),
        vec![0, 32, 64, 0, 32, 64, 0, 32, 64, 0, 32, 64]
    );
    let forward = report.endpoints[2].training.as_ref().unwrap();
    assert_eq!(forward.model, *report.initial.as_ref().unwrap());
    assert_eq!(forward.work.derivative_terms, 0);
    assert_eq!(forward.work.projected_steps, 0);
    assert_eq!(forward.work.effective_updates, 0);
    assert_eq!(forward.work.cache_products, 0);
    assert!(!report.target_realizability_certified);
    Ok(())
}

#[test]
fn source_discovery_coefficients_weights_and_work_mutations_refuse() -> Result<()> {
    let input = fixture();
    let report = binding::run(input.clone())?;
    let round: binding::Report = serde_json::from_slice(&serde_json::to_vec(&report)?)?;
    assert_eq!(round, report);
    binding::check(&input, &round)?;
    let mut bad = report.clone();
    bad.discovery.positions += 1;
    assert!(binding::check(&input, &bad).is_err());
    let mut bad = report.clone();
    bad.initial.as_mut().unwrap().head.coefficients[0] += 0.1;
    assert!(binding::check(&input, &bad).is_err());
    let mut bad = report.clone();
    bad.endpoints[0].training.as_mut().unwrap().model.weights = vec![1., 0.];
    assert!(binding::check(&input, &bad).is_err());
    let mut bad = report.clone();
    bad.endpoints[2]
        .training
        .as_mut()
        .unwrap()
        .work
        .gradient_batches = 1;
    assert!(binding::check(&input, &bad).is_err());
    let mut bad = report.clone();
    bad.target_realizability_certified = true;
    assert!(binding::check(&input, &bad).is_err());
    let mut bad = input.clone();
    bad.observations[0].label = 7;
    assert!(binding::check(&bad, &report).is_err());
    Ok(())
}

#[test]
fn conflicting_labels_and_caps_remain_explicit_all_mode_outcomes() -> Result<()> {
    let mut input = fixture();
    let mut conflict = input.observations[0].clone();
    conflict.label = 0;
    input.observations.push(conflict);
    let report = binding::run(input.clone())?;
    binding::check(&input, &report)?;
    assert!(report.discovery.retained.is_empty());
    assert_eq!(report.endpoints.len(), 3);
    assert!(
        report
            .endpoints
            .iter()
            .all(|e| e.error.is_some() && e.training.is_none())
    );
    let input = Input {
        vocabulary: 40,
        context: 2,
        observations: vec![Observation {
            tokens: vec![1, 2],
            label: 0,
        }],
    };
    let report = discovery::run(input.clone())?;
    discovery::check(&input, &report)?;
    assert_eq!(report.error.as_deref(), Some("retained program cap1024"));
    let mut input = fixture();
    input.context = 0;
    assert!(discovery::run(input).is_err());
    assert!(program::catalogue(1024, 4096).is_err());
    Ok(())
}

#[test]
fn shared_copy_predictions_follow_vocabulary_permutations() -> Result<()> {
    let input = fixture();
    let report = binding::run(input)?;
    let model = report.initial.unwrap();
    let rename = |t| match t {
        2 => 3,
        3 => 2,
        6 => 7,
        7 => 6,
        _ => t,
    };
    let tokens = vec![1, 2, 7, 3, 5, 2];
    let renamed: Vec<_> = tokens.iter().copied().map(rename).collect();
    let a = model.logits_tokens(&tokens);
    let b = model.logits_tokens(&renamed);
    for c in 0..8 {
        assert!((a[c] - b[rename(c)]).abs() < 1e-10);
    }
    Ok(())
}
