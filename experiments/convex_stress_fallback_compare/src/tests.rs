use super::*;
use metrics::Predictor;
mod fixtures;
mod runner;
use fixtures::{Fixture, input, overwrite};
#[test]
fn frozen_sources_and_prospective_protocol_match() -> Result<()> {
    let p = protocol()?;
    assert_eq!(p["metric_scalar_work_limit"], outcomes::METRIC_WORK_LIMIT);
    assert_eq!(
        p["parent"]["source_fingerprint"],
        convex_stress_readout_compare::fingerprint()
    );
    assert_eq!(
        p["staging"]["source_fingerprint"],
        convex_stress_causal_pairs_cohort::fingerprint()
    );
    assert_eq!(p["core_commit"], "027237cdddde7f8b4019501b26d52f0f270b1001");
    let original: Value = serde_json::from_str(include_str!(
        "../../convex_stress_causal_pairs/cohort/protocol.json"
    ))?;
    assert_eq!(p["discovery"], original["discovery"]);
    assert_eq!(p["training"], original["training"]);
    Ok(())
}

#[test]
fn combined_metric_forecast_uses_the_uniform_allowance_and_keeps_its_cap() -> Result<()> {
    let mut x = input("forecast-fixture")?;
    let request = &x.configuration.as_ref().unwrap().discovery;
    let candidate =
        component::model::initialize(&component::discovery::run(request.clone())?)?.model;
    let parent = fallback::Parent::from_checkpoint(json!({"kind":"Counts","model":{
        "features":{"base":{"vocab":7,"context":4096}},"scale":10.,"total":0,
        "shared":{"total":0,"keys":{},"active":[]},"local":{}}}))?;
    let model = fallback::Model::new(candidate, parent, &request.input)?;
    let sample = x.task.splits["validation"][0].clone();
    x.task
        .splits
        .insert("validation".into(), vec![sample.clone(); 3000]);
    let work = outcomes::forecast(&x.task, &model)?;
    assert!(work > 2_000_000_000 && work <= outcomes::METRIC_WORK_LIMIT);
    x.task
        .splits
        .insert("validation".into(), vec![sample; 40_000]);
    assert!(outcomes::forecast(&x.task, &model).is_err());
    Ok(())
}
#[test]
fn complete_original_scope_preserves_baseline_and_uses_primary_success() -> Result<()> {
    let f = Fixture::new()?;
    let inv = f.prepare()?;
    parents::load_with(&f.request.parent, false, &f.expected_parent, &f.all, true)?;
    parents::load_with(&f.request.staging, true, &f.expected_stage, &f.all, true)?;
    for task in &f.all {
        f.run(&inv, task)?;
    }
    let summary = summary::summarize_with(&f.destination, &inv, f.parent.clone(), &f.stage, input)?;
    assert_eq!(summary["baseline_arms"], f.parent["arms"]);
    assert_eq!(summary["complete_parent"], f.parent);
    assert_eq!(summary["records"].as_array().unwrap().len(), 35);
    for arm in summary["arms"].as_array().unwrap() {
        assert_eq!(arm["choices"].as_array().unwrap().len(), 105);
        assert_eq!(arm["full35_pass"], true);
        assert_eq!(arm["all_split_diagnostic_pass"], false);
        for m in arm["by_mode"].as_array().unwrap() {
            assert_eq!(m["primary_passes"], 35);
            assert_eq!(m["all_split_passes"], 34);
        }
    }
    let result: Value = read(&f.destination.join(&f.all[0]).join("result.json"))?;
    assert_eq!(result["evaluation"].as_array().unwrap().len(), 6);
    for r in result["evaluation"].as_array().unwrap() {
        assert_eq!(r["prediction_limit"], 4096);
        assert_eq!(r["new_gradient_updates"], 0);
        assert_eq!(r["covered_records"], 3);
        assert_eq!(r["primary_pass"], true);
        assert_eq!(r["all_splits_pass"], false);
        assert_eq!(
            r["prediction_masks"]["validation"]["supported_mask"],
            json!([false, true])
        );
        assert_eq!(
            r["prediction_masks"]["validation"]["parent_route_mask"],
            json!([true, false])
        );
        assert_eq!(
            r["prediction_masks"]["validation"]["exact_observed_input_mask"],
            json!([false, false])
        );
    }
    Ok(())
}
#[test]
fn process_failures_keep_all_parent_choices_and_partial_evidence() -> Result<()> {
    let f = Fixture::new()?;
    let inv = f.prepare()?;
    for task in &f.all {
        fixtures::files(&f.destination, task, false, None)?;
    }
    fs::create_dir(f.destination.join(&f.all[0]))?;
    fs::write(
        f.destination.join(&f.all[0]).join("partial.txt"),
        "retained allocation refusal\n",
    )?;
    let summary = summary::summarize_with(&f.destination, &inv, f.parent.clone(), &f.stage, input)?;
    assert_eq!(summary["process_failures"], 35);
    for policy in POLICIES {
        for task in &f.all {
            for m in MODES {
                assert_eq!(
                    comparison::cell(&summary, policy, task, m)?,
                    comparison::cell(&f.parent, policy, task, m)?
                );
            }
        }
    }
    assert!(
        summary["records"][0]["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["path"].as_str().unwrap().ends_with("partial.txt"))
    );
    Ok(())
}
#[test]
fn structural_choices_ignore_heldout_scores_and_missing_checkpoint_keeps_the_head() -> Result<()> {
    let f = Fixture::new()?;
    let x = input(&f.all[0])?;
    let b = imported::build(
        &f.request.checkpoint_root,
        &f.request.staging,
        &f.stage,
        &f.parent,
        &f.all[0],
        &x,
    )?;
    let first = comparison::transform(&f.parent, &f.all[0], &b.rows)?;
    let mut parent = f.parent.clone();
    for arm in parent["arms"].as_array_mut().unwrap() {
        for row in arm["choices"].as_array_mut().unwrap() {
            row["metrics"] = json!({"forged_for_selection_independence":true});
        }
    }
    assert_eq!(first, comparison::transform(&parent, &f.all[0], &b.rows)?);
    fs::remove_file(f.request.checkpoint_root.join("actual-parent.json"))?;
    let missing = imported::build(
        &f.request.checkpoint_root,
        &f.request.staging,
        &f.stage,
        &f.parent,
        &f.all[0],
        &x,
    )?;
    assert!(missing.models.is_empty());
    assert!(missing.rows.iter().all(|r| r["status"] == "refused"));
    let retained = comparison::transform(&f.parent, &f.all[0], &missing.rows)?;
    assert!(
        retained["choices"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["selected"] == r["prior_reference"])
    );
    Ok(())
}
#[test]
fn rehashed_wrapper_parameters_support_and_original_input_mutations_are_rejected() -> Result<()> {
    let f = Fixture::new()?;
    let inv = f.prepare()?;
    let task = &f.all[0];
    f.run(&inv, task)?;
    let dir = f.destination.join(task);
    let file = dir.join("minimum_ce-forward_only.json");
    let original: Value = read(&file)?;
    let imports: Vec<Value> = read(&dir.join("imports.json"))?;
    for field in ["parameters", "support"] {
        let mut forged = original.clone();
        if field == "parameters" {
            let n = forged["candidate"]["coefficients"][0].as_f64().unwrap();
            forged["candidate"]["coefficients"][0] = json!(n + 0.125);
        } else {
            forged["candidate"]["blocks"][0]["query"] = json!(0);
        }
        overwrite(&file, &forged);
        let mut rows = imports.clone();
        let row = rows
            .iter_mut()
            .find(|r| r["model_file"] == "minimum_ce-forward_only.json")
            .unwrap();
        row["model_sha256"] = json!(digest(&file)?);
        overwrite(&dir.join("imports.json"), &rows);
        assert!(
            execute::retained(
                &f.destination,
                task,
                &inv,
                &f.parent,
                &f.stage,
                &input(task)?
            )
            .is_err()
        );
    }
    overwrite(&file, &original);
    overwrite(&dir.join("imports.json"), &imports);
    let mut x = input(task)?;
    x.configuration
        .as_mut()
        .unwrap()
        .discovery
        .input
        .observations[0]
        .label = 0;
    assert!(execute::retained(&f.destination, task, &inv, &f.parent, &f.stage, &x).is_err());
    let path = f.request.staging.root.join(task).join("attempt.json");
    let mut forged: Value = read(&path)?;
    let n = forged["report"]["endpoints"][0]["training"]["model"]["coefficients"][0]
        .as_f64()
        .unwrap();
    forged["report"]["endpoints"][0]["training"]["model"]["coefficients"][0] = json!(n + 0.125);
    overwrite(&path, &forged);
    let mut stage = f.stage.clone();
    stage["records"][0]["result"]["attempt_sha256"] = json!(digest(&path)?);
    assert!(
        imported::build(
            &f.request.checkpoint_root,
            &f.request.staging,
            &stage,
            &f.parent,
            task,
            &input(task)?
        )
        .is_err()
    );
    Ok(())
}
#[test]
fn foreign_review_scope_checkpoint_and_mask_corruption_are_rejected() -> Result<()> {
    let f = Fixture::new()?;
    let inv = f.prepare()?;
    let task = &f.all[0];
    f.run(&inv, task)?;
    let mut incomplete = f.parent.clone();
    incomplete["arms"][0]["choices"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(comparison::validate(&incomplete, &f.all).is_err());
    let mut pin = f.request.parent.clone();
    pin.review.exit_sha256 = "0".repeat(64);
    assert!(parents::load_with(&pin, false, &f.expected_parent, &f.all, false).is_err());
    let file = f.destination.join(task).join("result.json");
    let mut result: Value = read(&file)?;
    result["evaluation"][0]["prediction_masks"]["validation"]["supported_mask"][0] = json!(true);
    overwrite(&file, &result);
    assert!(
        execute::retained(
            &f.destination,
            task,
            &inv,
            &f.parent,
            &f.stage,
            &input(task)?
        )
        .is_err()
    );
    fs::write(f.request.checkpoint_root.join("actual-parent.json"), "{}")?;
    assert!(
        execute::inventory_with(
            &f.destination,
            &f.suite,
            &f.parent_inv,
            &f.parent,
            &f.stage_inv,
            &f.stage,
            false
        )
        .is_err()
    );
    assert!(parents::checkpoint_path(&f.request.checkpoint_root, "../escape.json").is_err());
    Ok(())
}
#[test]
fn original_counts_checkpoint_is_imported_by_saved_kind_and_exact_digest() -> Result<()> {
    let f = Fixture::new()?;
    let path = f.request.checkpoint_root.join("counts.json");
    let count = json!({"kind":"Counts","model":{"features":{"base":{"vocab":7,"context":65}},"scale":10.,"total":0,"shared":{"total":0,"keys":{},"active":[]},"local":{}}});
    write(&path, &count)?;
    let old = json!({"saved_endpoint":{"model_path":"counts.json","model_sha256":digest(&path)?}});
    let (model, _) = imported::predecessor(&f.request.checkpoint_root, &old)?.unwrap();
    let absolute = json!({"saved_endpoint":{"model_path":path,"model_sha256":digest(&path)?}});
    let (same, _) = imported::predecessor(&f.request.checkpoint_root, &absolute)?.unwrap();
    assert_eq!(model.prediction_limit(), 65);
    assert!(matches!(model, fallback::Parent::Counts(_)));
    let original: convex_stress_counts::AdaptiveModel =
        serde_json::from_value(count["model"].clone())?;
    assert_eq!(model.logits(&[1; 65])?, original.logits(&[1; 65]));
    assert_eq!(same.logits(&[1; 65])?, original.logits(&[1; 65]));
    let mut changed = count;
    changed["kind"] = json!("Unsupported");
    overwrite(&path, &changed);
    assert!(imported::predecessor(&f.request.checkpoint_root, &old).is_err());
    let updated =
        json!({"saved_endpoint":{"model_path":"counts.json","model_sha256":digest(&path)?}});
    assert!(imported::predecessor(&f.request.checkpoint_root, &updated)?.is_none());
    Ok(())
}
