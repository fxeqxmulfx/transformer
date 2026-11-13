use crate::{
    comparison,
    interval_search::{self, Choice, Input, Policy, RetainedChoice},
    random::Random,
    search,
    search_tests::fixture,
};

#[test]
fn masked_interval_attempts_retain_exact_refusals_and_earlier_success() {
    let (truth, categorical) = fixture(true);
    let input = Input {
        categorical: categorical.clone(),
        policy: Policy::default(),
    };
    let out = interval_search::propose(input.clone()).unwrap();
    assert_eq!(out.exact, search::propose(categorical.clone()).unwrap());
    assert!(out.exact.selected.is_none());
    assert_eq!(out.attempts.len(), 18);
    assert!(
        out.attempts
            .iter()
            .any(|a| a.extraction.as_ref().is_some_and(|e| e.outcome.accepted))
    );
    assert!(
        !out.attempts
            .iter()
            .any(|a| a.reason == "interval_iteration_budget_exhausted")
    );
    println!(
        "{}",
        serde_json::json!({"masked_interval_diagnostics":out.attempts.iter().map(|a|
        serde_json::json!({"exact_attempt":a.exact_attempt,"variant":a.variant,"reason":a.reason,
            "last":a.extraction.as_ref().and_then(|e|e.outcome.trace.last())})).collect::<Vec<_>>()})
    );
    assert_eq!(out.cost.ce_gradients, 0);
    assert_eq!(
        out.cost.iterations_charged,
        out.attempts
            .iter()
            .map(|a| a.iterations_charged)
            .sum::<usize>()
    );
    assert!(out.cost.iterations_charged <= input.policy.total_iterations);
    assert_eq!(
        out.cost.latent_frequency_audit_allowance,
        out.cost.model_audits * categorical.n
    );
    let errors = match out.selected {
        Some(Choice::Interval(i)) => out.attempts[i].model.as_ref().map(|m| {
            m.output_labels
                .iter()
                .zip(&truth.output_labels)
                .filter(|(a, b)| a != b)
                .count()
        }),
        Some(Choice::Exact(_)) => panic!("all exact masked proposals refuse"),
        None => None,
    };
    println!(
        "{}",
        serde_json::json!({"masked_selected":out.selected,"full_table_errors":errors,
        "interval_cost":out.cost,"attempts":out.attempts.iter().map(|a|
            serde_json::json!({"exact_attempt":a.exact_attempt,"variant":a.variant,"reason":a.reason,
                "ce":a.train_ce,"iterations":a.iterations_charged})).collect::<Vec<_>>()})
    );
    interval_search::replay(&input, &out).unwrap();
    let samples = search::validate(&categorical).unwrap();
    let old = convex_stress_decode::search::propose(
        categorical.n,
        categorical.vocab,
        &samples,
        convex_stress_decode::search::Options {
            code_trials: 16,
            power_steps: 64,
            refinement_steps: 24,
            seed: 73,
        },
    )
    .unwrap();
    assert!(old.exact_fit);
    let model = old.model.unwrap();
    assert_eq!(model.output_labels, truth.output_labels);
    let combined = interval_search::compare(out.clone(), vec![model.clone()]).unwrap();
    assert_eq!(combined.proposal, out);
    assert_eq!(
        combined.selected,
        Some(RetainedChoice::Earlier(comparison::Choice::Prior(0)))
    );
    interval_search::replay_comparison(&input, std::slice::from_ref(&model), &combined).unwrap();
    let mut bad = combined;
    bad.selected = None;
    assert!(interval_search::replay_comparison(&input, &[model], &bad).is_err());
}

#[test]
fn complete_exact_success_is_retained_and_output_renaming_preserves_the_interval_search() {
    let (_, categorical) = fixture(false);
    let input = Input {
        categorical,
        policy: Policy {
            total_iterations: 2048,
            ..Policy::default()
        },
    };
    let a = interval_search::propose(input.clone()).unwrap();
    assert!(matches!(a.selected, Some(Choice::Exact(_))));
    let mut changed = input;
    let mut rename: Vec<_> = (0..changed.categorical.vocab).collect();
    Random::new(51).shuffle(&mut rename);
    for row in &mut changed.categorical.observations {
        row[1] = rename[row[1]];
    }
    let b = interval_search::propose(changed).unwrap();
    assert_eq!(a.cost, b.cost);
    assert_eq!(a.selected, b.selected);
    for (a, b) in a.attempts.iter().zip(&b.attempts) {
        assert_eq!(a.extraction, b.extraction);
        assert_eq!(a.reason, b.reason);
        if let (Some(a), Some(b)) = (&a.model, &b.model) {
            assert_eq!(a.row_phases, b.row_phases);
            assert_eq!(a.column_phases, b.column_phases);
            assert!(
                a.output_labels
                    .iter()
                    .zip(&b.output_labels)
                    .all(|(&a, &b)| rename[a] == b)
            );
        }
    }
}

#[test]
fn interval_search_budget_and_input_trace_tampering_are_checked() {
    let (_, categorical) = fixture(true);
    let input = Input {
        categorical,
        policy: Policy {
            total_iterations: 2,
            ..Policy::default()
        },
    };
    let out = interval_search::propose(input.clone()).unwrap();
    assert_eq!(out.attempts.len(), 18);
    assert_eq!(out.cost.iterations_charged, 2);
    assert!(
        out.attempts
            .iter()
            .any(|a| a.reason == "interval_iteration_budget_exhausted")
    );
    let restored = serde_json::from_slice(&serde_json::to_vec(&out).unwrap()).unwrap();
    interval_search::replay(&input, &restored).unwrap();
    let mut bad = out;
    bad.attempts[0].iterations_charged = 0;
    assert!(interval_search::replay(&input, &bad).is_err());
    let mut changed = input.clone();
    changed.categorical.options.seed += 1;
    assert!(interval_search::replay(&changed, &restored).is_err());
    let mut invalid = input;
    invalid.policy.variants.push(invalid.policy.variants[0]);
    assert!(interval_search::propose(invalid).is_err());
}
