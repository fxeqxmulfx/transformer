use super::*;
use crate::{random::Random, search_tests::fixture};

fn parent(input: &interval_search::Input, prior: &[Model]) -> refinement_search::Report {
    let p = interval_search::propose(input.clone()).unwrap();
    let p = interval_search::compare(p, prior.to_vec()).unwrap();
    refinement_search::run(
        input,
        prior,
        p,
        refinement_search::Policy {
            rounds: 4,
            seeds: 4,
            assignments: 16,
        },
    )
    .unwrap()
}
fn small(masked: bool) -> (Model, interval_search::Input) {
    let (truth, cat) = fixture(masked);
    (
        truth,
        interval_search::Input {
            categorical: cat,
            policy: interval_search::Policy {
                total_iterations: 2,
                ..interval_search::Policy::default()
            },
        },
    )
}
fn policy() -> Policy {
    Policy {
        variants: vec![interval::Options {
            model_tolerance: 0.1,
            iterations: 2048,
            ..interval::Options::default()
        }],
        total_iterations: 12288,
        model_audits: 12,
        refinement: refinement_search::Policy {
            rounds: 12,
            seeds: 12,
            assignments: 128,
        },
    }
}

#[test]
fn actual_row_balance_admits_uniform_density_and_exposes_zero_row_policy() {
    let b = vec![
        C::new(3., 4.),
        C::default(),
        C::new(2., 0.),
        C::new(0., 2.),
        C::default(),
        C::new(0., 3.),
    ];
    let a = feature_balance::build(&b, 3, 2, &[true, false, true]).unwrap();
    let raw = feature_balance::audit(&a).unwrap();
    assert_eq!(raw["completed_rows"], serde_json::json!([1]));
    assert!(raw["diagonal_error"].as_f64().unwrap() < 1e-12);
    assert_eq!(raw["objective_optimality_claim"], false);
    assert!(feature_balance::build(&b, 3, 2, &[true, true, true]).is_err());
    let mut bad = a.clone();
    bad.features[0] += C::new(0.01, 0.);
    assert!(feature_balance::audit(&bad).is_err());
    let mut bad = a;
    bad.uniform[0] = C::new(0.1, 0.);
    assert!(feature_balance::audit(&bad).is_err());
    assert!(feature_balance::build(&[], 194, 2, &[]).is_err());
}

#[test]
fn partial_clock_fills_only_missing_axes_and_preserves_observed_choices() {
    let x = vec![
        C::default(),
        C::from_polar(2., -2.),
        C::from_polar(3., 1.),
        C::from_polar(1., 2.),
    ];
    assert!(search::clock(&x).is_none());
    let a = feature_balance::clock(&x, &[false, true, true, true]).unwrap();
    assert_eq!(a.phases, vec![3, 0, 1, 2]);
    assert_eq!(a.missing, vec![0]);
    let mut other = x.clone();
    other[0] = C::new(f64::NAN, 0.);
    assert_eq!(
        a,
        feature_balance::clock(&other, &[false, true, true, true]).unwrap()
    );
    assert!(feature_balance::clock(&x, &[true, true, true, true]).is_err());
    let mut collision = x;
    collision[2] = collision[1] * 3.;
    assert!(feature_balance::clock(&collision, &[false, true, true, true]).is_err());
}

#[test]
fn balanced_proposals_retain_earlier_success_and_all_failed_parent_slots() {
    let (truth, input) = small(true);
    let prior = vec![truth];
    let old = parent(&input, &prior);
    let out = run(&input, &prior, old.clone(), policy()).unwrap();
    assert_eq!(out.parent, old);
    assert_eq!(out.subspaces.len(), 6);
    assert_eq!(out.attempts.len(), 12);
    assert_eq!(out.selected.as_deref(), Some("prior-0"));
    assert!(out.subspaces.iter().all(|s| s.audit.is_some()));
    assert!(out.cost.model_audits > 0);
    assert_eq!(out.cost.ce_gradients, 0);
    assert_eq!(
        out.cost.iterations_charged,
        out.attempts
            .iter()
            .map(|a| a.iterations_charged)
            .sum::<usize>()
    );
    assert_eq!(
        out.cost.assignments_charged,
        out.attempts
            .iter()
            .map(|a| a.assignments_charged)
            .sum::<usize>()
    );
    println!(
        "{}",
        serde_json::json!({"balanced_masked_fixture":out.selected,"cost":out.cost,
        "candidates":candidates(&out).unwrap().iter().map(|(id,m,ce)|serde_json::json!({"id":id,"ce":ce,
            "full_errors":m.output_labels.iter().zip(&prior[0].output_labels).filter(|(a,b)|a!=b).count()})).collect::<Vec<_>>() })
    );
    let saved = serde_json::from_slice(&serde_json::to_vec(&out).unwrap()).unwrap();
    replay(&input, &prior, &old, &saved).unwrap();
    let mut bad = out;
    bad.cost.iterations_charged += 1;
    assert!(replay(&input, &prior, &old, &bad).is_err());
}

#[test]
fn missing_row_proposals_and_renamed_labels_replay_without_answer_rules() {
    let (_, mut input) = small(false);
    let n = input.categorical.n;
    input.categorical.observations.retain(|s| s[0] % n != 0);
    let old = parent(&input, &[]);
    let a = run(&input, &[], old.clone(), policy()).unwrap();
    assert!(!a.observed_rows[0]);
    assert_eq!(a.observed_rows.iter().filter(|&&b| b).count(), n - 1);
    assert!(
        a.attempts
            .iter()
            .any(|a| a.row_clock.as_ref().is_some_and(|c| c.missing == vec![0]))
    );
    replay(&input, &[], &old, &a).unwrap();
    let mut rename: Vec<_> = (0..input.categorical.vocab).collect();
    Random::new(51).shuffle(&mut rename);
    let mut changed = input.clone();
    for s in &mut changed.categorical.observations {
        s[1] = rename[s[1]];
    }
    let b = run(&changed, &[], parent(&changed, &[]), policy()).unwrap();
    assert_eq!(a.cost, b.cost);
    assert_eq!(a.selected, b.selected);
    assert_eq!(a.subspaces, b.subspaces);
    let (ca, cb) = (candidates(&a).unwrap(), candidates(&b).unwrap());
    assert_eq!(ca.len(), cb.len());
    for ((ia, ma, _), (ib, mb, _)) in ca.iter().zip(&cb) {
        assert_eq!(ia, ib);
        assert!(
            ma.output_labels
                .iter()
                .zip(&mb.output_labels)
                .all(|(&a, &b)| rename[a] == b)
        );
    }
    let mut bad = a;
    bad.observed_rows[0] = true;
    assert!(replay(&input, &[], &old, &bad).is_err());
}

#[test]
fn balanced_caps_refusals_and_original_input_binding_remain_explicit() {
    let (_, input) = small(true);
    let old = parent(&input, &[]);
    let p = Policy {
        total_iterations: 1,
        model_audits: 1,
        refinement: refinement_search::Policy {
            seeds: 1,
            rounds: 1,
            assignments: 2,
        },
        ..policy()
    };
    let out = run(&input, &[], old.clone(), p).unwrap();
    assert_eq!(out.attempts.len(), 12);
    assert_eq!(out.cost.iterations_charged, 1);
    assert!(
        out.cost.model_audits <= 1 && out.cost.seeds_run <= 1 && out.cost.assignments_charged <= 2
    );
    assert!(
        out.attempts
            .iter()
            .any(|a| a.reason == "balanced_iteration_budget_exhausted")
    );
    replay(&input, &[], &old, &out).unwrap();
    let mut changed = input.clone();
    changed.categorical.options.seed += 1;
    assert!(replay(&changed, &[], &old, &out).is_err());
    let mut invalid = policy();
    invalid.variants.push(invalid.variants[0]);
    assert!(run(&input, &[], old, invalid).is_err());
}
