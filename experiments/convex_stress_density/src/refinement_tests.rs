use crate::{
    assignment_certificate as assignment,
    refinement::{self, Input},
    refinement_audit,
    search_tests::fixture,
};

#[test]
fn dual_certificates_match_all_binary_three_by_three_assignment_optima_and_reject_corruption() {
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for bits in 0..512usize {
        let cost: Vec<_> = (0..9).map(|i| (bits >> i) & 1).collect();
        let c = assignment::solve(&cost, 3).unwrap();
        let optimum = permutations
            .iter()
            .map(|p| (0..3).map(|i| cost[3 * i + p[i]]).sum::<usize>())
            .min()
            .unwrap();
        assert_eq!(c.objective, optimum);
        assignment::check(&cost, 3, &c).unwrap();
        let mut bad = c.clone();
        bad.row_dual[0] += 1;
        assert!(assignment::check(&cost, 3, &bad).is_err());
        let mut bad = c;
        bad.row_dual[0] = i64::MAX;
        bad.column_dual[0] = i64::MAX;
        assert!(assignment::check(&cost, 3, &bad).is_err());
    }
}

#[test]
fn decoder_refit_recovers_observed_many_to_one_labels_and_preserves_full_ce_and_spectra() {
    let (truth, categorical) = fixture(false);
    let mut seed = truth.clone();
    seed.decoder.fill(0);
    seed = crate::Model::construct(
        seed.vocab,
        seed.row_phases,
        seed.column_phases,
        seed.decoder,
    )
    .unwrap();
    let input = Input {
        seed,
        observations: categorical.observations,
        rounds: 24,
    };
    let out = refinement::run(input.clone()).unwrap();
    let audit = refinement_audit::check(&input, &out).unwrap();
    assert_eq!(
        out.steps[out.best.unwrap()]
            .state
            .model(truth.vocab)
            .unwrap(),
        truth
    );
    assert_eq!(audit["endpoint"]["training_errors"], 0);
    assert_eq!(out.cost.ce_gradients, 0);
    refinement::replay(&input, &out).unwrap();
    let mut bad = out.clone();
    bad.steps[1].state.decoder[0] = 0;
    assert!(refinement_audit::check(&input, &bad).is_err());
    let mut bad = out.clone();
    bad.steps[2].certificate.as_mut().unwrap().row_dual[0] += 1;
    assert!(refinement_audit::check(&input, &bad).is_err());
    let mut bad = out;
    bad.cost.assignment_relaxations += 1;
    assert!(refinement::replay(&input, &bad).is_err());
}

#[test]
fn masked_interval_seed_refines_with_certificates_without_assuming_hidden_success() {
    let (truth, categorical) = fixture(true);
    let parent = crate::interval_search::propose(crate::interval_search::Input {
        categorical: categorical.clone(),
        policy: Default::default(),
    })
    .unwrap();
    let seed = parent
        .attempts
        .iter()
        .find_map(|a| a.model.clone())
        .unwrap();
    let input = Input {
        seed,
        observations: categorical.observations,
        rounds: 24,
    };
    let out = refinement::run(input.clone()).unwrap();
    let audit = refinement_audit::check(&input, &out).unwrap();
    assert!(
        out.steps
            .windows(2)
            .all(|s| s[1].state.errors <= s[0].state.errors)
    );
    assert!(out.steps.len() <= 121);
    let last = &out.steps[out.best.unwrap()].state;
    let model = last.model(truth.vocab).unwrap();
    println!(
        "{}",
        serde_json::json!({"seed_errors":out.steps[0].state.errors,"refined_errors":last.errors,
        "full_table_errors":model.output_labels.iter().zip(&truth.output_labels).filter(|(a,b)|a!=b).count(),
        "reason":out.reason,"steps":out.steps.len(),"cost":out.cost,"audit":audit})
    );
    refinement::replay(&input, &out).unwrap();
    let mut changed = input.clone();
    changed.observations[0][1] = (changed.observations[0][1] + 1) % truth.vocab;
    assert!(refinement::replay(&changed, &out).is_err());
    let mut short = input;
    short.rounds = 1;
    let capped = refinement::run(short.clone()).unwrap();
    assert_eq!(capped.steps.len(), 6);
    refinement_audit::check(&short, &capped).unwrap();
}

#[test]
fn retained_refinement_search_replays_parents_ties_and_every_budget_refusal() {
    use crate::{
        interval_search,
        refinement_search::{self, Choice, Policy},
    };
    let (truth, categorical) = fixture(true);
    let input = interval_search::Input {
        categorical,
        policy: Default::default(),
    };
    let proposal = interval_search::propose(input.clone()).unwrap();
    let empty = interval_search::compare(proposal.clone(), vec![]).unwrap();
    let refined = refinement_search::run(&input, &[], empty.clone(), Policy::default()).unwrap();
    assert!(matches!(refined.selected, Some(Choice::Refined(_))));
    assert_eq!(refined.parent, empty);
    assert_eq!(refined.attempts.len(), 24);
    refinement_search::replay(&input, &[], &empty, &refined).unwrap();
    let parent = interval_search::compare(proposal, vec![truth.clone()]).unwrap();
    let retained = refinement_search::run(
        &input,
        std::slice::from_ref(&truth),
        parent.clone(),
        Policy::default(),
    )
    .unwrap();
    assert_eq!(retained.parent, parent);
    assert_eq!(retained.selected, parent.selected.map(Choice::Earlier));
    let roundtrip = serde_json::from_slice(&serde_json::to_vec(&retained).unwrap()).unwrap();
    refinement_search::replay(&input, std::slice::from_ref(&truth), &parent, &roundtrip).unwrap();
    let capped = refinement_search::run(
        &input,
        std::slice::from_ref(&truth),
        parent.clone(),
        Policy {
            seeds: 1,
            ..Policy::default()
        },
    )
    .unwrap();
    assert_eq!(capped.attempts.len(), 25);
    assert!(
        capped
            .attempts
            .iter()
            .any(|a| a.reason == "seed_budget_exhausted")
    );
    let mut changed = retained;
    changed.selected = None;
    assert!(
        refinement_search::replay(&input, std::slice::from_ref(&truth), &parent, &changed).is_err()
    );
    let bad_seed = crate::Model::construct(
        truth.vocab,
        truth.row_phases.clone(),
        truth.column_phases.clone(),
        vec![0; truth.q],
    )
    .unwrap();
    let prior = vec![bad_seed.clone(), bad_seed];
    let parent = interval_search::compare(empty.proposal.clone(), prior.clone()).unwrap();
    let limited = refinement_search::run(
        &input,
        &prior,
        parent.clone(),
        Policy {
            assignments: 2,
            ..Policy::default()
        },
    )
    .unwrap();
    assert_eq!(limited.assignments_charged, 2);
    assert!(
        limited
            .attempts
            .iter()
            .any(|a| a.reason == "assignment_budget_exhausted")
    );
    refinement_search::replay(&input, &prior, &parent, &limited).unwrap();
}

#[test]
fn unobserved_states_invalid_seeds_and_output_renaming_are_explicit() {
    let (truth, categorical) = fixture(false);
    let input = Input {
        seed: truth.clone(),
        observations: vec![categorical.observations[0]],
        rounds: 1,
    };
    let out = refinement::run(input.clone()).unwrap();
    assert!(out.best.is_none());
    assert!(refinement_audit::check(&input, &out).unwrap()["endpoint"].is_null());
    let mut invalid = input.clone();
    invalid.rounds = 129;
    assert!(refinement::run(invalid).is_err());
    let mut invalid = input;
    invalid.seed.row_phases[0] = invalid.seed.row_phases[1];
    assert!(refinement::run(invalid).is_err());
    let mut seed = truth;
    seed.decoder.rotate_left(1);
    seed = crate::Model::construct(
        seed.vocab,
        seed.row_phases,
        seed.column_phases,
        seed.decoder,
    )
    .unwrap();
    let a = Input {
        seed,
        observations: categorical.observations,
        rounds: 3,
    };
    let mut b = a.clone();
    let rename = |c: usize| (c + 3) % 11;
    b.seed.decoder.iter_mut().for_each(|c| *c = rename(*c));
    b.seed
        .output_labels
        .iter_mut()
        .for_each(|c| *c = rename(*c));
    b.observations.iter_mut().for_each(|s| s[1] = rename(s[1]));
    let x = refinement::run(a).unwrap();
    let y = refinement::run(b).unwrap();
    assert_eq!(x.cost, y.cost);
    assert_eq!(x.best, y.best);
    for (x, y) in x.steps.iter().zip(&y.steps) {
        assert_eq!(x.certificate, y.certificate);
        assert_eq!(x.state.rows, y.state.rows);
        assert_eq!(x.state.columns, y.state.columns);
        assert_eq!(x.state.errors, y.state.errors);
        assert_eq!(
            x.state
                .decoder
                .iter()
                .map(|&c| rename(c))
                .collect::<Vec<_>>(),
            y.state.decoder
        );
    }
}
