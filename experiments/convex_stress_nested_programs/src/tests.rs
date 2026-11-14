use crate::*;
mod boundary;
mod gradients;

// Exhaustive short words plus explicit mismatched nesting, no original data.
fn fixture() -> Input {
    let oracle = program::Program {
        pairs: vec![(1, 2), (3, 4)],
    };
    let mut words = vec![];
    for length in 1..=3 {
        for word in 0..5usize.pow(length) {
            let mut n = word;
            let mut tokens = vec![0];
            for _ in 0..length {
                tokens.push(1 + n % 5);
                n /= 5;
            }
            words.push(tokens);
        }
    }
    words.extend([
        vec![0, 1, 3, 4, 2],
        vec![0, 1, 3, 2, 4],
        vec![0, 3, 1, 2, 4],
        vec![0, 3, 1, 4, 2],
    ]);
    let observations = (0..167)
        .map(|i| {
            let tokens = words[i % words.len()].clone();
            let label = 6 + oracle.interpret(&tokens).0;
            Observation { tokens, label }
        })
        .collect();
    Input {
        vocabulary: 10,
        context: 5,
        observations,
    }
}
fn deep() -> Vec<usize> {
    let mut tokens = vec![0];
    for _ in 0..10 {
        tokens.extend([1, 3]);
    }
    for _ in 0..10 {
        tokens.extend([4, 2]);
    }
    tokens
}

#[test]
fn exhaustive_independent_stack_and_cancellation_with_complete_bank() -> Result<()> {
    for (n, count) in [(2, 2), (3, 6), (4, 24), (6, 330), (8, 4256), (9, 11664)] {
        let bank = program::catalogue(&(0..n).collect::<Vec<_>>(), n)?;
        assert_eq!(bank.len(), count);
        assert_eq!(
            bank.iter().collect::<std::collections::BTreeSet<_>>().len(),
            count
        );
        for p in bank {
            let symbols: std::collections::BTreeSet<_> =
                p.pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
            assert_eq!(symbols.len(), 2 * p.pairs.len());
            assert!((1..=3).contains(&p.pairs.len()));
        }
    }
    let bank = program::catalogue(&[0, 1, 2, 3], 4)?;
    for length in 0..=5 {
        for word in 0..4usize.pow(length) {
            let mut n = word;
            let tokens: Vec<_> = (0..length)
                .map(|_| {
                    let t = n % 4;
                    n /= 4;
                    t
                })
                .collect();
            for p in &bank {
                let v = p.run(&tokens);
                assert_eq!((v.state, v.peak_depth), p.interpret(&tokens));
                assert!(v.state < 3 && v.peak_depth <= tokens.len());
                assert_eq!(v.positions, tokens.len() as u64);
            }
        }
    }
    for symbols in [vec![0], vec![0, 0], vec![1, 0], (0..10).collect()] {
        assert!(program::catalogue(&symbols, 10).is_err());
    }
    assert!(program::catalogue(&[0, 4], 4).is_err());
    Ok(())
}

#[test]
fn learned_complete_complex_heads_all_modes_short_tails_and_deeper_input() -> Result<()> {
    let input = fixture();
    let report = latent::run(input.clone())?;
    latent::check(&input, &report)?;
    assert!(report.initialization_error.is_none());
    assert!(!report.target_realizability_certified);
    assert_eq!(report.discovery.alphabet, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(report.discovery.attempts.len(), 330);
    assert!(
        report
            .discovery
            .retained
            .iter()
            .any(|p| p.program.pairs == vec![(1, 2), (3, 4)])
    );
    assert_eq!(
        report.audit.as_ref().unwrap().real_head_coordinates,
        report.discovery.retained.len() * 2 * STATES * 10
    );
    for (endpoint, (batches, examples)) in
        report.endpoints.iter().zip([(16, 2672), (24, 668), (0, 0)])
    {
        let out = endpoint.training.as_ref().unwrap();
        assert_eq!(out.work.gradient_batches, batches);
        assert_eq!(out.work.gradient_examples, examples);
        let model = &out.model;
        for (tokens, target) in [(deep(), 6), (vec![0, 1, 3, 2, 4], 8), (vec![0, 1, 3, 5], 7)] {
            let states: Vec<_> = model.states(&tokens).iter().map(|v| v.state).collect();
            assert!(model.labels(&states).iter().all(|&y| y == Some(target)));
            let z = model.logits_states(&states);
            assert_eq!(argmax(&z), target);
            assert!((ce(&z, target) - (1. + 9. * (-24f64).exp()).ln()).abs() < 1e-10);
        }
        assert!(model.head.code.iter().any(|v| v.im.abs() > 0.1));
        assert!(
            model.head.coefficients.iter().any(|b| b
                .iter()
                .enumerate()
                .any(|(i, v)| (i / 10) % 2 == 1 && v.abs() > 1.))
        );
        assert!(model.states(&deep()).iter().all(|v| v.peak_depth == 20));
    }
    let mini = report.endpoints[1].training.as_ref().unwrap();
    assert_eq!(
        mini.steps.iter().map(|s| s.count).collect::<Vec<_>>(),
        [32, 32, 32, 32, 32, 7].repeat(4)
    );
    let forward = report.endpoints[2].training.as_ref().unwrap();
    assert_eq!(forward.model, *report.initial.as_ref().unwrap());
    assert_eq!(forward.work.cache_products, 0);
    assert_eq!(forward.work.derivative_terms, 0);
    assert_eq!(forward.work.projected_steps, 0);
    Ok(())
}

#[test]
fn unknown_states_conflicts_caps_and_source_mutations_are_visible() -> Result<()> {
    let input = Input {
        vocabulary: 3,
        context: 2,
        observations: vec![Observation {
            tokens: vec![0, 1],
            label: 2,
        }],
    };
    let found = discovery::run(input.clone())?;
    discovery::check(&input, &found)?;
    assert_eq!(found.retained.len(), 2);
    let model = model::Model::new(&found)?;
    let states: Vec<_> = model.states(&[0, 0]).iter().map(|s| s.state).collect();
    assert!(model.labels(&states).contains(&None));
    let mut changed = found.clone();
    changed.retained[0].source_rows.fill(None);
    assert!(discovery::check(&input, &changed).is_err());
    let mut collision = input.clone();
    collision.observations.push(Observation {
        tokens: vec![0, 1],
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
        context: 6,
        observations: vec![Observation {
            tokens: (0..6).collect(),
            label: 7,
        }],
    })?;
    assert_eq!(cap.error.as_deref(), Some("retained latent program cap128"));
    let mut oversized = fixture();
    oversized.context = 4096;
    oversized.observations = vec![
        Observation {
            tokens: (0..4096).map(|i| i % 9).collect(),
            label: 9
        };
        32
    ];
    assert!(
        discovery::run(oversized)
            .unwrap_err()
            .to_string()
            .contains("scalar-work")
    );
    Ok(())
}

#[test]
fn complete_state_role_alphabet_decoder_and_work_replay_reject_mutations() -> Result<()> {
    let input = fixture();
    let report = latent::run(input.clone())?;
    let round: latent::Report = serde_json::from_slice(&serde_json::to_vec(&report)?)?;
    assert_eq!(report, round);
    for variant in 0..7 {
        let mut bad = report.clone();
        match variant {
            0 => bad.initial.as_mut().unwrap().head.coefficients[0][1] += 0.1,
            1 => bad.initial.as_mut().unwrap().head.decoders[0][0] = Some(9),
            2 => bad
                .initial
                .as_mut()
                .unwrap()
                .alphabet
                .pop()
                .map(|_| ())
                .unwrap(),
            3 => bad.initial.as_mut().unwrap().programs[0].pairs[0] = (0, 5),
            4 => {
                bad.endpoints[2]
                    .training
                    .as_mut()
                    .unwrap()
                    .work
                    .gradient_examples = 1
            }
            5 => bad.endpoints[0].training.as_mut().unwrap().model.weights[0] += 0.1,
            _ => bad.endpoints.swap(0, 1),
        }
        assert!(latent::check(&input, &bad).is_err(), "variant {variant}");
    }
    let mut changed = input.clone();
    changed.observations[0].label = 9;
    assert!(latent::check(&changed, &report).is_err());
    Ok(())
}
