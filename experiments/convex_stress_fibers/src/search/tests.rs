use super::*;

fn fixture(q: usize, fibers: &[&[usize]], withheld: &[usize]) -> (Input, Model) {
    let vocab = fibers.len() * 3 + 5;
    let mut d = vec![usize::MAX; q];
    for (class, fiber) in fibers.iter().enumerate() {
        for &z in fiber.iter() {
            d[z] = class * 3 + 2;
        }
    }
    assert!(d.iter().all(|&z| z < vocab));
    // Only this fixture builder knows the hidden phases and decoder. The
    // learner receives category identifiers and observed equality classes.
    let rows = (0..q).map(|i| (5 * i + 2) % q).collect();
    let columns = (0..q).map(|j| (3 * j + 4) % q).collect();
    let truth = Model::construct(vocab, rows, columns, d).unwrap();
    let observations = truth
        .output_labels
        .iter()
        .enumerate()
        .filter(|(i, _)| !withheld.contains(i))
        .map(|(i, &v)| [i, v])
        .collect();
    (
        Input {
            n: q,
            vocab,
            observations,
            policy: Policy::default(),
        },
        truth,
    )
}
fn triples() -> (Input, Model) {
    fixture(7, &[&[0], &[1, 2, 4], &[3, 5, 6]], &[3, 10, 25, 40])
}

#[test]
fn masked_permuted_three_and_four_fibers_recover_without_latent_inputs() {
    for (input, truth) in [
        triples(),
        fixture(
            13,
            &[&[0], &[1, 5, 8, 12], &[2, 3, 10, 11], &[4, 6, 7, 9]],
            &[0, 8, 17, 39, 71, 102, 131, 162],
        ),
    ] {
        let report = run(input.clone()).unwrap();
        check(&input, &report).unwrap();
        replay(&input, &report).unwrap();
        let chosen = &report.candidates[report.selected.expect("balanced fiber recovery")];
        assert_eq!(chosen.model.output_labels, truth.output_labels);
        assert_ne!(chosen.common_mass, 0);
        assert_eq!(chosen.audit["training_errors"], 0);
        assert_eq!(chosen.audit["vocabulary"], input.vocab);
        assert!(
            report
                .attempts
                .iter()
                .any(|a| a.elimination.as_ref().is_some_and(|s| !s.free.is_empty()))
        );
        assert!(report.attempts.iter().any(|a| {
            a.trials
                .iter()
                .any(|t| matches!(t.decision, Decision::Collision { .. }))
        }));
        let bytes = serde_json::to_vec(&report).unwrap();
        assert_eq!(serde_json::from_slice::<Report>(&bytes).unwrap(), report);
    }
}

#[test]
fn paired_fibers_and_class_renaming_preserve_categorical_discovery() {
    let (input, truth) = fixture(
        13,
        &[
            &[0],
            &[1, 12],
            &[2, 11],
            &[3, 10],
            &[4, 9],
            &[5, 8],
            &[6, 7],
        ],
        &[2, 20, 50, 82, 144],
    );
    let report = run(input.clone()).unwrap();
    check(&input, &report).unwrap();
    assert_eq!(
        report.candidates[report.selected.unwrap()]
            .model
            .output_labels,
        truth.output_labels
    );
    let mut renamed = input.clone();
    for s in &mut renamed.observations {
        s[1] = input.vocab - 1 - s[1];
    }
    let second = run(renamed.clone()).unwrap();
    check(&renamed, &second).unwrap();
    let (a, b) = (&report.candidates[0].model, &second.candidates[0].model);
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.columns, b.columns);
    assert_eq!(
        a.decoder
            .iter()
            .map(|&v| input.vocab - 1 - v)
            .collect::<Vec<_>>(),
        b.decoder
    );
    assert_eq!(report.work, second.work);
}

#[test]
fn complete_training_fit_still_leaves_two_different_unseen_completions() {
    let input = Input {
        n: 3,
        vocab: 7,
        observations: vec![[0, 4], [1, 6], [2, 6], [3, 6], [6, 6]],
        policy: Policy::default(),
    };
    let report = run(input.clone()).unwrap();
    let audit = check(&input, &report).unwrap();
    replay(&input, &report).unwrap();
    assert_eq!(report.candidates.len(), 2);
    assert_eq!(audit["unseen_answers_identified"], false);
    assert_ne!(
        report.candidates[0].model.output_labels,
        report.candidates[1].model.output_labels
    );
    for c in &report.candidates {
        assert_eq!(c.audit["training_errors"], 0);
        assert!(
            input
                .observations
                .iter()
                .all(|s| c.model.output_labels[s[0]] == s[1])
        );
    }
}

#[test]
fn incomplete_fiber_guesses_and_uncovered_decoder_states_are_retained_as_refusals() {
    let input = Input {
        n: 3,
        vocab: 8,
        observations: (0..9).filter(|&i| i != 8).map(|i| [i, 7]).collect(),
        policy: Policy::default(),
    };
    let report = run(input.clone()).unwrap();
    check(&input, &report).unwrap();
    let wrong = report.attempts.iter().find(|a| a.size == 2).unwrap();
    assert!(
        wrong
            .trials
            .iter()
            .any(|t| matches!(t.decision, Decision::FiberMismatch { cardinality: 3, .. }))
    );
    assert!(
        !wrong
            .trials
            .iter()
            .any(|t| matches!(t.decision, Decision::Accepted { .. }))
    );
    let mut missing = input;
    missing.observations = (0..9)
        .filter(|&c| (c % 3 + c / 3) % 3 != 2)
        .map(|c| [c, 7])
        .collect();
    let report = run(missing.clone()).unwrap();
    check(&missing, &report).unwrap();
    replay(&missing, &report).unwrap();
    assert!(report.selected.is_none());
    assert!(report.attempts.iter().any(|a| {
        a.trials
            .iter()
            .any(|t| matches!(t.decision, Decision::Uncovered { .. }))
    }));
}

#[test]
fn budget_stops_missing_axes_and_unsupported_domains_make_no_success_claim() {
    let (base, _) = triples();
    for which in 0..6 {
        let mut input = base.clone();
        match which {
            0 => input.policy.guesses = 0,
            1 => input.policy.enumeration = 0,
            2 => input.policy.models = 0,
            3 => input.policy.elimination.scalar_updates = 0,
            4 => input.policy.elimination.row_operations = 0,
            _ => input.policy.constraints = 1,
        }
        let report = run(input.clone()).unwrap();
        check(&input, &report).unwrap();
        replay(&input, &report).unwrap();
        assert!(report.selected.is_none());
        assert!(report.candidates.is_empty());
    }
    let mut missing = base.clone();
    missing.observations.retain(|s| s[0] % missing.n != 0);
    let report = run(missing.clone()).unwrap();
    assert_eq!(report.reason, "missing_observed_axis");
    check(&missing, &report).unwrap();
    let composite = Input {
        n: 4,
        vocab: 2,
        observations: vec![[0, 0]],
        policy: Policy::default(),
    };
    let report = run(composite.clone()).unwrap();
    assert_eq!(report.reason, "unsupported_prime_domain");
    check(&composite, &report).unwrap();
    let mut invalid = base;
    invalid.policy.elimination.nullity = 4;
    assert!(run(invalid).is_err());
}

#[test]
fn altered_groups_equations_enumeration_models_audits_and_ledgers_are_rejected() {
    let (input, _) = triples();
    let report = run(input.clone()).unwrap();
    let mut wrong = report.clone();
    wrong.partition.groups[0].label = (wrong.partition.groups[0].label + 1) % input.vocab;
    assert!(check(&input, &wrong).is_err());
    wrong = report.clone();
    wrong.attempts[0]
        .elimination
        .as_mut()
        .unwrap()
        .problem
        .system
        .right[0] = 1;
    assert!(check(&input, &wrong).is_err());
    wrong = report.clone();
    wrong.attempts[0].trials[0].coefficients[0] = 1;
    assert!(check(&input, &wrong).is_err());
    wrong = report.clone();
    wrong.candidates[0].common_mass = (wrong.candidates[0].common_mass + 1) % input.n;
    assert!(check(&input, &wrong).is_err());
    wrong = report.clone();
    wrong.candidates[0].model.output_labels[0] =
        (wrong.candidates[0].model.output_labels[0] + 1) % input.vocab;
    assert!(check(&input, &wrong).is_err());
    wrong = report.clone();
    wrong.candidates[0].audit["training_errors"] = 1.into();
    assert!(check(&input, &wrong).is_err());
    wrong = report.clone();
    wrong.work.permutation_entries += 1;
    assert!(replay(&input, &wrong).is_err());
    let mut changed = input.clone();
    changed.observations[0][1] = (changed.observations[0][1] + 1) % input.vocab;
    assert!(replay(&changed, &report).is_err());
}
