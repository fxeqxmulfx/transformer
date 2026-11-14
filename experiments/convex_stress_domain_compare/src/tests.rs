use super::*;
mod causal_route;
#[path = "../../convex_stress_shared_compare/src/tests/fixtures.rs"]
mod fixtures;
mod runner;
#[path = "../../convex_stress_shared_compare/src/tests/sources.rs"]
mod sources;
#[path = "../../convex_stress_shared_compare/src/tests/tensors.rs"]
mod tensors;
use fixtures::*;
#[test]
fn frozen_sources_and_prospective_protocol_match() -> Result<()> {
    parents::sources()?;
    let p = protocol()?;
    assert_eq!(
        p["required_done_tasks"],
        json!([237, 246, 252, 274, 275, 276, 278, 279, 280])
    );
    assert_eq!(p["memory_limit_bytes"], 68719476736_u64);
    assert_eq!(p["metric_work_limit"], 32000000000_u64);
    assert_eq!(planned()?.len(), 23);
    assert_eq!(tasks(&suite()?)?.len(), 35);
    assert_eq!(catalogue().len(), 55);
    Ok(())
}
#[test]
fn primary_sequence_success_is_independent_of_token_and_auxiliary_diagnostics() -> Result<()> {
    let scores = json!({"train":{"sequence_accuracy":0.},"validation":{"sequence_accuracy":0.995,"token_accuracy":0.},
      "long":{"sequence_accuracy":0.1,"token_accuracy":1.}});
    assert_eq!(passes(&scores, "validation")?, (true, false));
    assert_eq!(passes(&scores, "long")?, (false, false));
    assert!(passes(&scores, "missing").is_err());
    assert!(passes(&json!({"train":{"sequence_accuracy":1.}}), "train").is_err());
    Ok(())
}
#[test]
fn original_selector_keeps_both_arms_parent_ties_and_ascending_primes() -> Result<()> {
    let all = tasks(&suite()?)?;
    let mut p = parent(&all);
    let offer = |prime, ce| selection::Offer {
        task: all[0].clone(),
        mode: training::Mode::Full,
        prime,
        train_ce: ce,
        model_sha256: "1".repeat(64),
        stage_sha256: "2".repeat(64),
    };
    let tied = selection::select(&p, &[offer(3, 4. - 1e-12)], &all)?;
    for policy in POLICIES {
        assert_eq!(
            cell(&tied, policy, &all[0], MODES[0])?["selected"],
            cell(&p, policy, &all[0], MODES[0])?["reference"]
        );
    }
    let offers = vec![offer(5, 3.), offer(3, 3.)];
    let selected = selection::select(&p, &offers, &all)?;
    for policy in POLICIES {
        assert_eq!(
            cell(&selected, policy, &all[0], MODES[0])?["selected"],
            offers[1].reference()
        );
        assert_eq!(
            selected["arms"]
                .as_array()
                .unwrap()
                .iter()
                .find(|a| a["initializer"] == policy)
                .unwrap()["choices"]
                .as_array()
                .unwrap()
                .len(),
            105
        );
    }
    p["arms"][0]["choices"][0]["metrics"]["validation"]["sequence_accuracy"] = json!(0.);
    assert_eq!(selected, selection::select(&p, &offers, &all)?);
    assert!(selection::select(&p, &[offer(3, 3.), offer(3, 2.)], &all).is_err());
    Ok(())
}
#[test]
fn complete_render_labels_inheritance_and_never_fills_failed_cases_with_success() -> Result<()> {
    let all = tasks(&suite()?)?;
    let parent = parent(&all);
    let pin = reference(PathBuf::from("/fixture"));
    let mut cases = inherited_cases(&parent, &all)?;
    let arms = summary::render(&parent, &cases, &all, &pin)?;
    assert!(
        arms.iter()
            .all(|a| a["choices"].as_array().unwrap().len() == 105)
    );
    assert!(
        arms.iter().all(|a| a["by_mode"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["primary_passes"] == 35
                && m["inherited_metrics"] == 35
                && m["fresh_metrics"] == 0))
    );
    cases[0]["status"] = json!("process_failure");
    let failed = summary::render(&parent, &cases, &all, &pin)?;
    assert!(failed.iter().all(|a| {
        a["by_mode"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["primary_passes"] == 34 && m["no_model"] == 1)
    }));
    cases[0]["status"] = json!("complete");
    cases[0]["result"]["seal"]["choices"]["choices"][0]["selected"] = json!("changed:missing");
    assert!(summary::render(&parent, &cases, &all, &pin).is_err());
    cases.pop();
    assert!(summary::render(&parent, &cases, &all, &pin).is_err());
    Ok(())
}
