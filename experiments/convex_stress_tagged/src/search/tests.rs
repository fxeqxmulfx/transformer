use super::*;

fn fixture() -> (Input, Model) {
    let interior = Model::construct(9, vec![0, 1, 2], vec![0, 2, 1], vec![2, 5, 7]).unwrap();
    let axes = vec![None, Some(0), Some(1), Some(2)];
    let truth = interior.lift(8, &axes, &axes).unwrap();
    let observations = truth
        .output_labels
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 13)
        .map(|(cell, &label)| [cell, label])
        .collect();
    let policy = Policy {
        minimum_strip_support: 3,
        code_trials: 8,
        seed_budget: 16,
        axis_budget: 256,
        refinement_rounds: 12,
        ..Policy::default()
    };
    (
        Input {
            n: 4,
            vocab: 9,
            observations,
            policy,
        },
        truth,
    )
}

#[test]
fn generic_observed_only_discovery_recovers_masked_constant_strips() {
    let (input, truth) = fixture();
    let report = run(input.clone()).unwrap();
    replay(&input, &report).unwrap();
    assert_eq!(report.views.len(), 2);
    assert_eq!(report.views[1].constant, Some(8));
    assert_eq!(report.views[1].stripped_observations, 7);
    let best = &report.attempts[report.selected.unwrap()];
    assert_eq!(best.audit.as_ref().unwrap()["training_errors"], 0);
    assert_eq!(
        best.model.as_ref().unwrap().output_labels,
        truth.output_labels
    );
    assert_eq!(best.model.as_ref().unwrap().q, 9);
    assert_eq!(report.work.ce_gradients, 0);
    assert!(report.work.axis_steps <= input.policy.axis_budget);
    assert!(report.work.seeds <= input.policy.seed_budget);
    let raw = serde_json::to_vec(&report).unwrap();
    assert_eq!(serde_json::from_slice::<Report>(&raw).unwrap(), report);
}

#[test]
fn categorical_label_renaming_preserves_the_search_and_phase_states() {
    let (input, _) = fixture();
    let first = run(input.clone()).unwrap();
    let mut renamed = input;
    for o in &mut renamed.observations {
        o[1] = (o[1] + 4) % 9;
    }
    let second = run(renamed.clone()).unwrap();
    assert_eq!(first.work, second.work);
    assert_eq!(first.selected, second.selected);
    for (a, b) in first.attempts.iter().zip(&second.attempts) {
        assert_eq!(a.reason, b.reason);
        if let (Some(a), Some(b)) = (&a.model, &b.model) {
            assert_eq!(a.rows, b.rows);
            assert_eq!(a.columns, b.columns);
            assert_eq!(a.latent_labels, b.latent_labels);
            assert_eq!(
                a.output_labels
                    .iter()
                    .map(|c| (c + 4) % 9)
                    .collect::<Vec<_>>(),
                b.output_labels
            );
        } else {
            assert!(a.model.is_none() && b.model.is_none());
        }
    }
    replay(&renamed, &second).unwrap();
}

#[test]
fn missing_axes_padding_and_false_constant_evidence_remain_explicit() {
    // Category0 is wholly unobserved on the row axis; column0 is observed
    // constant. Another row is only apparently constant on its observed part.
    let samples = vec![
        Sample { cell: 1, label: 2 },
        Sample { cell: 2, label: 2 },
        Sample { cell: 4, label: 0 },
        Sample { cell: 5, label: 1 },
        Sample { cell: 7, label: 1 },
        Sample { cell: 8, label: 0 },
    ];
    let found = views::propose(3, 3, &samples, 2).unwrap();
    let view = found.iter().find(|v| v.constant == Some(2)).unwrap();
    assert_eq!(view.missing_rows, vec![0]);
    assert_eq!(view.rows, vec![None, Some(0), Some(1)]);
    assert_eq!(view.columns, vec![None, Some(0), Some(1)]);
    assert_eq!(view.stripped_observations, 2);
    views::replay(3, 3, &samples, 2, &found).unwrap();
    let mut wrong = found.clone();
    wrong[1].rows[1] = None;
    assert!(views::replay(3, 3, &samples, 2, &wrong).is_err());
    let sparse = vec![
        Sample { cell: 0, label: 2 },
        Sample { cell: 3, label: 2 },
        Sample { cell: 1, label: 0 },
        Sample { cell: 4, label: 1 },
        Sample { cell: 8, label: 0 },
    ];
    let uneven = views::propose(3, 3, &sparse, 2).unwrap();
    let padded = uneven.iter().find(|v| v.constant == Some(2)).unwrap();
    assert_eq!(
        (padded.n, padded.row_padding, padded.column_padding),
        (3, 1, 0)
    );
    // A partially observed row supports a hypothesis only; nothing in the
    // proposal says its missing third label is2.
    assert_eq!(padded.row_support[0], 2);
}

#[test]
fn zero_sketch_budget_and_domain_refusals_preserve_every_slot() {
    let (mut input, _) = fixture();
    input.policy.seed_budget = 0;
    let out = run(input.clone()).unwrap();
    assert!(out.selected.is_none());
    assert_eq!(out.attempts.len(), 16);
    assert_eq!(out.work.seeds, 0);
    assert_eq!(out.work.sketches, 0);
    assert!(
        out.attempts
            .iter()
            .all(|a| a.reason == "seed_budget_exhausted")
    );
    replay(&input, &out).unwrap();
    input.policy.seed_budget = 8;
    input.policy.max_views = 1;
    let out = run(input.clone()).unwrap();
    assert!(
        out.sketches
            .iter()
            .filter(|s| s.view == 1)
            .all(|s| s.reason == "view_budget_exhausted")
    );
    input.observations = (0..16).map(|c| [c, 1]).collect();
    let out = run(input.clone()).unwrap();
    assert!(out.selected.is_none());
    assert!(
        out.sketches
            .iter()
            .any(|s| s.reason == "left_normalization_failed")
    );
    assert!(
        out.attempts
            .iter()
            .any(|a| a.reason == "latent_domain_out_of_bounds")
    );
    replay(&input, &out).unwrap();
    input.policy.code_trials = 33;
    assert!(run(input).is_err());
}

#[test]
fn power_trace_candidate_partition_and_ledger_tampering_fail_full_replay() {
    let (input, _) = fixture();
    let out = run(input.clone()).unwrap();
    let mut wrong = out.clone();
    wrong.sketches[0].sketch.as_mut().unwrap().states[1].left[0][0] += 0.1;
    assert!(replay(&input, &wrong).is_err());
    wrong = out.clone();
    wrong.views[1].stripped_observations += 1;
    assert!(replay(&input, &wrong).is_err());
    wrong = out.clone();
    wrong.work.power_products += 1;
    assert!(replay(&input, &wrong).is_err());
    wrong = out;
    let i = wrong.selected.unwrap();
    wrong.attempts[i].model.as_mut().unwrap().output_labels[0] = 0;
    assert!(replay(&input, &wrong).is_err());
}

#[test]
fn angular_quantization_supports_repeated_phases_and_explicit_missing_entries() {
    let x = vec![[1., 0.], [1., 0.], [-0.5, 0.8660254037844386], [0., 0.]];
    assert_eq!(
        sketch::quantize(&x, &[true, true, true, false], 3).unwrap(),
        vec![0, 0, 1, 0]
    );
    assert!(sketch::quantize(&x, &[true; 4], 3).is_err());
    assert!(sketch::quantize(&x, &[false; 4], 3).is_err());
    assert!(sketch::quantize(&x, &[true; 4], 258).is_err());
}
