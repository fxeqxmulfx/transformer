use crate::{
    Model, Sample,
    comparison::{self, Choice},
    random::Random,
    search::{self, Input, SearchOptions},
    solver::Options,
};

fn fixture(masked: bool) -> (Model, Input) {
    let n = 11;
    let (mut rows, mut columns) = ((0..n).collect::<Vec<_>>(), (0..n).collect::<Vec<_>>());
    let mut rng = Random::new(381);
    rng.shuffle(&mut rows);
    rng.shuffle(&mut columns);
    let truth = Model::construct(n, rows, columns, vec![1, 5, 1, 7, 2, 2, 8, 5, 3, 1, 6]).unwrap();
    let input = Input {
        n,
        vocab: n,
        observations: (0..n * n)
            .filter(|cell| !masked || cell % 17 != 0)
            .map(|cell| [cell, truth.output_labels[cell]])
            .collect(),
        options: SearchOptions {
            code_trials: 2,
            ranks: vec![1, 2, 3],
            seed: 73,
            extraction: Options {
                iterations: 256,
                ..Options::default()
            },
            total_iterations: 2048,
            model_audits: 4,
        },
    };
    (truth, input)
}

#[test]
fn categorical_complete_table_pipeline_uses_one_mode_for_both_axes() {
    let (truth, input) = fixture(false);
    let out = search::propose(input.clone()).unwrap();
    assert_eq!(out.attempts.len(), 6);
    let chosen = out
        .selected
        .expect("complete categorical sketch must propose a mode");
    let model = out.attempts[chosen].model.as_ref().unwrap();
    assert_eq!(model.output_labels, truth.output_labels);
    assert_eq!(
        out.attempts[chosen].model_audit.as_ref().unwrap()["training_errors"],
        0
    );
    assert_eq!(out.cost.ce_gradients, 0);
    assert!(out.cost.iteration_budget_used <= input.options.total_iterations);
    assert_eq!(
        out.cost.iteration_budget_used,
        out.attempts
            .iter()
            .map(|a| a.iterations_charged)
            .sum::<usize>()
    );
    assert_eq!(
        out.cost.latent_frequency_audit_allowance,
        out.cost.model_audits * input.n
    );
    search::replay(&input, &out).unwrap();
    let restored = serde_json::from_slice(&serde_json::to_vec(&out).unwrap()).unwrap();
    search::replay(&input, &restored).unwrap();
    let mut bad = out.clone();
    bad.selected = None;
    assert!(search::replay(&input, &bad).is_err());
    let mut bad = out;
    bad.attempts[0].features.as_mut().unwrap()[0].re += 1e-4;
    assert!(search::replay(&input, &bad).is_err());
}

#[test]
fn categorical_output_renaming_preserves_observed_search_and_choices() {
    let (_, input) = fixture(false);
    let a = search::propose(input.clone()).unwrap();
    let mut changed = input;
    let mut rename: Vec<_> = (0..changed.vocab).collect();
    Random::new(51).shuffle(&mut rename);
    for item in &mut changed.observations {
        item[1] = rename[item[1]];
    }
    let b = search::propose(changed).unwrap();
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
                    .all(|(&x, &y)| rename[x] == y)
            );
        }
    }
}

#[test]
fn masked_subspace_refusals_preserve_earlier_successful_candidate() {
    let (truth, input) = fixture(true);
    let out = search::propose(input.clone()).unwrap();
    assert_eq!(out.attempts.len(), 6);
    assert!(
        out.selected.is_none(),
        "exact density equalities should expose the masked-subspace gap"
    );
    assert!(
        out.attempts
            .iter()
            .all(|a| a.extraction.as_ref().is_some_and(|e| !e.accepted))
    );
    let samples: Vec<_> = input
        .observations
        .iter()
        .map(|&[cell, label]| Sample { cell, label })
        .collect();
    let old = convex_stress_decode::search::propose(
        input.n,
        input.vocab,
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
    let combined = comparison::compare(out.clone(), vec![model.clone()]).unwrap();
    assert_eq!(combined.proposal, out);
    assert_eq!(combined.prior[0].model, model);
    assert_eq!(combined.selected, Some(Choice::Prior(0)));
    comparison::replay(&input, std::slice::from_ref(&model), &combined).unwrap();
    let mut bad = combined;
    bad.selected = None;
    assert!(comparison::replay(&input, std::slice::from_ref(&model), &bad).is_err());
    search::replay(&input, &out).unwrap();
}

#[test]
fn budgets_and_rank_refusals_retain_the_complete_declared_attempt_scope() {
    let (_, mut input) = fixture(false);
    input.options.total_iterations = 2;
    let out = search::propose(input.clone()).unwrap();
    assert_eq!(out.attempts.len(), 6);
    assert!(out.cost.iteration_budget_used <= 2);
    assert!(
        out.attempts
            .iter()
            .any(|a| a.reason == "iteration_budget_exhausted")
    );
    input.observations.iter_mut().for_each(|s| s[1] = 1);
    input.options.total_iterations = 2048;
    let rank_zero = search::propose(input.clone()).unwrap();
    assert_eq!(rank_zero.attempts.len(), 6);
    assert!(
        rank_zero
            .attempts
            .iter()
            .all(|a| a.reason == "subspace_rank_deficient")
    );
    assert!(rank_zero.selected.is_none());
    input.options.ranks = vec![1, 1];
    assert!(search::propose(input.clone()).is_err());
    input.options.ranks = vec![1];
    input.observations.push(input.observations[0]);
    assert!(search::propose(input).is_err());
}
