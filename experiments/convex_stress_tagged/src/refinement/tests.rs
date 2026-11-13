use super::*;

fn fixture() -> (Input, Model) {
    let truth = Model::construct(9, vec![0, 0, 1, 2], vec![0, 1, 2, 0], vec![2, 5, 7]).unwrap();
    let seed = Model::construct(
        9,
        vec![0, 1, 1, 2],
        truth.columns.clone(),
        truth.decoder.clone(),
    )
    .unwrap();
    let observations = truth
        .output_labels
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 2)
        .map(|(cell, &label)| [cell, label])
        .collect();
    (
        Input {
            seed,
            observations,
            rounds: 8,
            axis_budget: 16,
        },
        truth,
    )
}

#[test]
fn repeated_phase_coordinate_update_recovers_a_masked_table_with_exact_integer_checks() {
    let (input, truth) = fixture();
    let out = run(input.clone()).unwrap();
    let audit = check(&input, &out).unwrap();
    replay(&input, &out).unwrap();
    assert!(out.steps[0].state.errors > 0);
    assert_eq!(out.reason, "exact_observed_fit");
    let best = out.steps[out.best.unwrap()]
        .state
        .model(truth.vocab)
        .unwrap();
    assert_eq!(best.output_labels, truth.output_labels);
    assert_eq!(best.rows[0], best.rows[1]);
    assert_ne!(best.n, best.q);
    assert_eq!(audit["endpoint"]["training_errors"], 0);
    assert_eq!(audit["endpoint"]["raw_frequency_audits"], 3);
    assert_eq!(
        audit["axis_observation_comparisons"],
        out.cost.axis_steps * input.observations.len() * 3
    );
    assert!(
        out.steps
            .windows(2)
            .all(|s| s[1].state.errors <= s[0].state.errors)
    );
    let raw = serde_json::to_vec(&out).unwrap();
    assert_eq!(serde_json::from_slice::<Outcome>(&raw).unwrap(), out);
}

#[test]
fn raw_minimum_state_coverage_ledger_and_input_tampering_are_rejected() {
    let (input, _) = fixture();
    let out = run(input.clone()).unwrap();
    let index = out.steps.iter().position(|s| s.kind == Kind::Row).unwrap();
    let mut wrong = out.clone();
    wrong.steps[index].minimum_costs.as_mut().unwrap()[0] += 1;
    assert!(check(&input, &wrong).is_err());
    wrong = out.clone();
    wrong.steps[index].state.columns[0] = 1;
    assert!(check(&input, &wrong).is_err());
    wrong = out.clone();
    wrong.steps[0].state.unseen_latents.push(0);
    assert!(check(&input, &wrong).is_err());
    wrong = out.clone();
    wrong.cost.axis_observation_comparisons -= 1;
    assert!(check(&input, &wrong).is_err());
    wrong = out.clone();
    wrong.reason = "global_optimum".into();
    assert!(replay(&input, &wrong).is_err());
    let mut changed = input.clone();
    changed.observations[0][1] = 8;
    assert!(check(&changed, &out).is_err());
    assert!(replay(&changed, &out).is_err());
}

#[test]
fn missing_latent_answers_are_refused_and_caps_and_dimensions_remain_explicit() {
    let (mut input, _) = fixture();
    input.axis_budget = 0;
    let out = run(input.clone()).unwrap();
    assert_eq!(out.reason, "axis_budget_exhausted");
    assert_eq!(out.steps.len(), 2);
    assert_eq!(out.cost.axis_steps, 0);
    check(&input, &out).unwrap();
    replay(&input, &out).unwrap();
    input.observations = vec![[0, input.seed.output_labels[0]]];
    let out = run(input.clone()).unwrap();
    assert_eq!(out.reason, "exact_fit_uncovered");
    assert_eq!(out.best, None);
    assert_eq!(out.steps[0].state.unseen_latents, vec![1, 2]);
    assert!(check(&input, &out).unwrap()["endpoint"].is_null());
    replay(&input, &out).unwrap();
    input.rounds = 129;
    assert!(run(input.clone()).is_err());
    input.rounds = 1;
    input.axis_budget = 257;
    assert!(run(input.clone()).is_err());
    input.axis_budget = 1;
    input.observations.push(input.observations[0]);
    assert!(run(input).is_err());
}

#[test]
fn covered_flat_exact_fits_can_still_disagree_on_an_unseen_axis() {
    let first = Model::construct(3, vec![0, 0, 0], vec![0, 1, 2], vec![0, 1, 2]).unwrap();
    let second = Model::construct(3, vec![0, 1, 0], vec![0, 1, 2], vec![0, 1, 2]).unwrap();
    let observations: Vec<_> = (0..3)
        .map(|c| [3 * c, first.output_labels[3 * c]])
        .collect();
    assert_ne!(first.output_labels[1], second.output_labels[1]);
    for seed in [first, second] {
        let input = Input {
            seed,
            observations: observations.clone(),
            rounds: 8,
            axis_budget: 16,
        };
        let out = run(input.clone()).unwrap();
        assert_eq!(out.reason, "exact_observed_fit");
        assert_eq!(out.best, Some(0));
        assert_eq!(
            check(&input, &out).unwrap()["endpoint"]["training_errors"],
            0
        );
        replay(&input, &out).unwrap();
    }
}

#[test]
fn integer_minima_hold_for_every_small_binary_observation_table() {
    // All 512 binary 3x3 tables, including inconsistent phase targets and ties.
    // This independently checks actual table errors, not just the implementation
    // formula, across all returned coordinate updates and decoder updates.
    for mask in 0..512 {
        let seed = Model::construct(2, vec![0, 1, 0], vec![0, 1, 1], vec![0, 1]).unwrap();
        let observations = (0..9).map(|cell| [cell, (mask >> cell) & 1]).collect();
        let input = Input {
            seed,
            observations,
            rounds: 3,
            axis_budget: 6,
        };
        let out = run(input.clone()).unwrap();
        check(&input, &out).unwrap();
        replay(&input, &out).unwrap();
    }
}
