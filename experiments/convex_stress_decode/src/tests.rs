use crate::{
    C, Model, Sample, assignment, linalg,
    metrics::Predictor,
    random::Random,
    search::{self, Options},
};

#[test]
fn decoded_corner_is_feasible_but_its_output_moment_exceeds_the_old_bound() {
    let model = Model::construct(2, vec![0, 1], vec![0, 1], vec![1, 1, 0]).unwrap();
    assert_eq!(model.output_labels, [1, 1, 1, 0]);
    let samples: Vec<_> = (0..3).map(|cell| Sample { cell, label: 1 }).collect();
    let audit = model.audit(&samples).unwrap();
    assert_eq!(audit["exact_observed_fit"], true);
    for norm in audit["latent_moment_norms"].as_array().unwrap() {
        assert!((norm["operator"].as_f64().unwrap() - 2.).abs() < 1e-12);
        assert!((norm["nuclear"].as_f64().unwrap() - 2.).abs() < 1e-12);
    }
    let decoded: Vec<_> = model
        .output_labels
        .iter()
        .map(|&c| C::new(if c == 1 { 1. } else { -1. }, 0.))
        .collect();
    assert!(linalg::svd(&decoded, 2).unwrap().values.iter().sum::<f64>() > 2.8);
    let other = Model::construct(2, vec![0, 0], vec![0, 0], vec![1, 1, 0]).unwrap();
    assert_eq!(
        other.audit(&samples).unwrap()["train_ce"],
        audit["train_ce"]
    );
    assert_ne!(model.logits(&[1, 1]), other.logits(&[1, 1]));
    let encoded = serde_json::to_vec(&model).unwrap();
    let restored: Model = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(restored.audit(&samples).unwrap(), audit);
    let mut wrong = restored.clone();
    wrong.output_labels[3] = 1;
    assert!(wrong.audit(&samples).is_err());
    wrong = restored;
    wrong.latent_labels[3] = 0;
    assert!(wrong.audit(&samples).is_err());
}

#[test]
fn unit_scale_ordinary_ce_excess_equals_error_fraction_in_full_vocabulary() {
    let model = Model::construct(11, vec![0, 1], vec![0, 1], vec![1, 1, 0]).unwrap();
    let samples = [Sample { cell: 0, label: 1 }, Sample { cell: 2, label: 7 }];
    let audit = model.audit(&samples).unwrap();
    let ideal = (1. + (model.vocab - 1) as f64 / std::f64::consts::E).ln();
    assert!((audit["train_ce"].as_f64().unwrap() - ideal - 0.5).abs() < 1e-12);
    assert_eq!(audit["training_errors"], 1);
    assert_eq!(audit["exact_observed_fit"], false);
}

fn permutations(values: &mut [usize], at: usize, visit: &mut impl FnMut(&[usize])) {
    if at == values.len() {
        visit(values);
        return;
    }
    for i in at..values.len() {
        values.swap(at, i);
        permutations(values, at + 1, visit);
        values.swap(at, i);
    }
}
#[test]
fn assignment_cost_matches_exhaustive_permutations() {
    let mut rng = Random::new(149);
    for n in 2..=5 {
        for _ in 0..20 {
            let costs: Vec<_> = (0..n * n)
                .map(|_| rng.below((n * n + 1) as u32) as usize)
                .collect();
            let assignment = assignment::minimize(&costs, n).unwrap();
            let cost: usize = assignment
                .iter()
                .enumerate()
                .map(|(i, &j)| costs[i * n + j])
                .sum();
            let mut best = usize::MAX;
            permutations(&mut (0..n).collect::<Vec<_>>(), 0, &mut |p| {
                best = best.min(p.iter().enumerate().map(|(i, &j)| costs[i * n + j]).sum());
            });
            assert_eq!(cost, best);
            let mut sorted = assignment;
            sorted.sort_unstable();
            assert_eq!(sorted, (0..n).collect::<Vec<_>>());
        }
    }
}

fn fixture() -> (Model, Vec<Sample>) {
    let n = 11;
    let (mut rows, mut columns) = ((0..n).collect::<Vec<_>>(), (0..n).collect::<Vec<_>>());
    let mut rng = Random::new(381);
    rng.shuffle(&mut rows);
    rng.shuffle(&mut columns);
    let truth = Model::construct(n, rows, columns, vec![1, 5, 1, 7, 2, 2, 8, 5, 3, 1, 6]).unwrap();
    let samples = (0..n * n)
        .filter(|cell| cell % 17 != 0)
        .map(|cell| Sample {
            cell,
            label: truth.output_labels[cell],
        })
        .collect();
    (truth, samples)
}
fn options() -> Options {
    Options {
        code_trials: 16,
        power_steps: 64,
        refinement_steps: 24,
        seed: 73,
    }
}

#[test]
fn categorical_sketch_discovers_a_coarsened_fixture_without_all_classes_observed() {
    let (truth, samples) = fixture();
    let proposal = search::propose(truth.n, truth.vocab, &samples, options()).unwrap();
    assert!(
        proposal.exact_fit,
        "{} {:?}",
        proposal.reason, proposal.best_errors
    );
    let model = proposal.model.unwrap();
    assert_eq!(
        model.output_labels, truth.output_labels,
        "withheld fixture cells"
    );
    assert_eq!(model.audit(&samples).unwrap()["training_errors"], 0);
    assert_eq!(model.vocab(), 11);
    assert!(
        samples
            .iter()
            .all(|s| s.label != 0 && s.label != 4 && s.label != 9 && s.label != 10)
    );
    assert_eq!(
        model.logits(&[0, 0]).len(),
        11,
        "full original CE vocabulary"
    );
    assert!(proposal.improvement_history.windows(2).all(|w| w[1] < w[0]));
    assert!(proposal.cost.code_trials <= options().code_trials);
    assert_eq!(proposal.cost.code_trials, proposal.trials.len());
    assert_eq!(proposal.cost.validation_observations, samples.len());
    assert_eq!(proposal.cost.class_observations, samples.len());
    assert_eq!(
        proposal.cost.sketch_observations,
        2 * proposal.trials.len() * samples.len()
    );
    assert_eq!(
        proposal.cost.verification_observations,
        proposal.improvement_history.len() * samples.len()
    );
    assert_eq!(
        proposal.cost.power_steps,
        proposal.trials.iter().map(|t| t.power_steps).sum::<usize>()
    );
    assert_eq!(
        proposal.cost.power_matvecs,
        proposal
            .trials
            .iter()
            .map(|t| t.power_matvecs)
            .sum::<usize>()
    );
    assert_eq!(
        proposal.cost.assignment_sweeps,
        proposal
            .trials
            .iter()
            .map(|t| t.assignment_sweeps)
            .sum::<usize>()
    );
    assert_eq!(proposal.trials.last().unwrap().reason, "exact_observed_fit");
    assert!(
        proposal.cost.assignment_sweeps <= 2 * options().code_trials * options().refinement_steps
    );
}

#[test]
fn output_renaming_preserves_the_training_only_search_and_full_predictions() {
    let (truth, samples) = fixture();
    let mut rename: Vec<_> = (0..truth.vocab).collect();
    Random::new(51).shuffle(&mut rename);
    let changed: Vec<_> = samples
        .iter()
        .map(|s| Sample {
            cell: s.cell,
            label: rename[s.label],
        })
        .collect();
    let a = search::propose(truth.n, truth.vocab, &samples, options()).unwrap();
    let b = search::propose(truth.n, truth.vocab, &changed, options()).unwrap();
    assert_eq!(a.exact_fit, b.exact_fit);
    assert_eq!(a.improvement_history, b.improvement_history);
    assert_eq!(a.cost, b.cost);
    assert_eq!(a.trials, b.trials);
    let (a, b) = (a.model.unwrap(), b.model.unwrap());
    assert_eq!(a.row_phases, b.row_phases);
    assert_eq!(a.column_phases, b.column_phases);
    assert_eq!(
        b.output_labels,
        a.output_labels
            .iter()
            .map(|&x| rename[x])
            .collect::<Vec<_>>()
    );
}

#[test]
fn failure_and_invalid_input_are_explicit_and_no_missing_class_is_dropped() {
    let samples = vec![Sample { cell: 0, label: 1 }, Sample { cell: 1, label: 1 }];
    let failed = search::propose(
        3,
        5,
        &samples,
        Options {
            code_trials: 2,
            ..options()
        },
    )
    .unwrap();
    assert!(!failed.exact_fit);
    assert!(failed.model.is_none());
    assert_eq!(failed.cost.code_trials, 2);
    assert_eq!(failed.reason, "no_covered_candidate");
    assert_eq!(failed.trials.len(), 2);
    assert!(
        failed
            .trials
            .iter()
            .all(|t| t.reason == "power_or_clock_failed" && t.states_scored == 0)
    );
    assert_eq!(failed.cost.power_matvecs, 2);
    assert!(crate::memory_bound(usize::MAX, 2, 2).is_err());
    assert!(crate::memory_bound(2, 194, 2).is_err());
    assert!(assignment::minimize(&[0; 4], usize::MAX).is_err());
    assert!(Model::construct(2, vec![0, 3], vec![0, 1], vec![1, 1, 0]).is_err());
    assert!(search::propose(2, 2, &[Sample { cell: 0, label: 2 }], options()).is_err());
    assert!(
        search::propose(
            2,
            2,
            &[Sample { cell: 0, label: 1 }, Sample { cell: 0, label: 1 }],
            options()
        )
        .is_err()
    );
    assert!(search::propose(2, 2, &[], options()).is_err());
    assert!(
        search::propose(
            2,
            2,
            &samples,
            Options {
                code_trials: 0,
                ..options()
            }
        )
        .is_err()
    );
}
