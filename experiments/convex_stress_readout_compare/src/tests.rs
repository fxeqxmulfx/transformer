use super::*;
mod fixtures;
mod partition;
mod runner;
use fixtures::{Fixture, input, overwrite};
#[test]
fn both_frozen_dependency_sources_match_the_prospective_protocol() -> Result<()> {
    let p = protocol()?;
    assert_eq!(
        p["parent"]["source_fingerprint"],
        convex_stress_kernel_compare::fingerprint()
    );
    assert_eq!(
        p["staging"]["source_fingerprint"],
        convex_stress_readout_kernel_cohort::fingerprint()
    );
    assert_eq!(p["planned"].as_array().unwrap().len(), 23);
    let refined = json!({"complete_parent":{"staged_construction":{"reference":{
        "summary_sha256":p["causal_stage_summary_sha256"],
        "inventory_sha256":p["causal_stage_inventory_sha256"]}}},
        "staging_sha256":p["refined_stage_summary_sha256"],
        "staged_construction":{"inventory_sha256":p["refined_stage_inventory_sha256"]}});
    let mut causal = json!({"complete_parent":refined,"staged_construction":{
        "source_fingerprint":p["kernel_stage"]["source_fingerprint"],
        "request":{"producer_commit":p["kernel_stage"]["producer_commit"]}}});
    parents::causal_stage(&causal)?;
    causal["complete_parent"]["complete_parent"]["staged_construction"]["reference"]["inventory_sha256"] =
        json!("a".repeat(64));
    assert!(parents::causal_stage(&causal).is_err());
    causal["complete_parent"]["complete_parent"]["staged_construction"]["reference"]["inventory_sha256"] =
        p["causal_stage_inventory_sha256"].clone();
    causal["complete_parent"]["staging_sha256"] = json!("b".repeat(64));
    assert!(parents::causal_stage(&causal).is_err());
    Ok(())
}
#[test]
fn primary_full35_pass_survives_a_failed_auxiliary_split_and_unknown_raw_alias() -> Result<()> {
    let f = Fixture::new()?;
    let inv = f.prepare()?;
    for task in &f.all {
        f.run(&inv, task)?;
    }
    let out = summary::summarize_with(&f.destination, &inv, f.parent.clone(), &f.stage, input)?;
    assert_eq!(out["tasks"], 35);
    assert_eq!(out["records"].as_array().unwrap().len(), 35);
    assert_eq!(out["complete_parent"], f.parent);
    assert_eq!(out["staged_construction"], f.stage);
    assert_eq!(
        out["standalone_candidates"]["endpoints"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    for mode in out["standalone_candidates"]["by_mode"].as_array().unwrap() {
        assert_eq!(mode["available_endpoints"], 1);
        assert_eq!(mode["primary_passes"], 1);
        assert_eq!(mode["all_split_passes"], 0);
    }
    for arm in out["selected_changes"]["arms"].as_array().unwrap() {
        assert_eq!(arm["replacements"], 3);
        assert_eq!(arm["new_primary_passes"], 3);
        assert_eq!(arm["primary_regressions"], 0);
    }
    assert_eq!(
        out["preparation_resources"],
        parents::evidence(&f.destination.join("prepare.resources"))?
    );
    for arm in out["arms"].as_array().unwrap() {
        assert_eq!(arm["choices"].as_array().unwrap().len(), 105);
        assert_eq!(arm["full35_pass"], true);
        assert_eq!(arm["all_split_diagnostic_pass"], false);
        for m in arm["by_mode"].as_array().unwrap() {
            assert_eq!(m["primary_passes"], 35);
            assert_eq!(m["all_split_passes"], 34);
        }
    }
    let x = input("fixture-0")?;
    let record = imported::case(&f.request.staging, &f.stage, "fixture-0", &x)?;
    let s = imported::load(
        &f.request.staging,
        "fixture-0",
        &record,
        &record["result"]["stages"][24],
        &x,
        true,
    )?;
    assert_eq!(
        s.attempt
            .discovery
            .as_ref()
            .unwrap()
            .raw_code
            .as_ref()
            .unwrap()
            .dimension,
        2
    );
    let model = &s.attempt.head.as_ref().unwrap().endpoints[2]
        .training
        .as_ref()
        .unwrap()
        .model;
    let e = &x.task.splits["validation"][0];
    let z = model.logits_cell(e.tokens[1] as usize + model.n * e.tokens[0] as usize);
    let raw = (0..z.len()).max_by(|&a, &b| z[a].total_cmp(&z[b])).unwrap();
    assert_ne!(raw, e.targets[1] as usize);
    let saved: Value = read(&f.destination.join("fixture-0/result.json"))?;
    for row in saved["evaluation"].as_array().unwrap() {
        let mask = &row["known_inputs"]["validation"];
        assert_eq!(mask["known_calls"], mask["calls"]);
        assert_eq!(mask["observed_calls"], 0);
        assert_eq!(mask["default_calls"], mask["calls"]);
        assert_eq!(mask["unknown_axis_calls"], 0);
        assert_eq!(mask["unmatched_active_calls"], 0);
    }
    Ok(())
}
#[test]
fn ce_ties_prime_order_and_both_parent_policies_remain_independent() -> Result<()> {
    let all: Vec<_> = (0..35).map(|i| format!("fixture-{i}")).collect();
    let mut p = fixtures::parent(&all);
    p["arms"][1]["choices"][0]["train_ce"] = json!(0.1);
    let offer = |prime, ce| comparison::Offer {
        task: all[0].clone(),
        mode: training::Mode::Full,
        prime,
        train_ce: ce,
        model_sha256: "a".repeat(64),
        stage_sha256: "b".repeat(64),
    };
    let offers = vec![offer(5, 0.5), offer(3, 0.5)];
    let selected = comparison::select(&p, &offers, &all)?;
    assert_eq!(
        selected["arms"][0]["choices"][0]["selected"],
        offers[1].reference()
    );
    assert_eq!(
        selected["arms"][1]["choices"][0]["selected"],
        p["arms"][1]["choices"][0]["reference"]
    );
    let tied = comparison::select(&p, &[offer(3, 1. - 0.5e-12)], &all)?;
    assert_eq!(
        tied["arms"][0]["choices"][0]["selected"],
        p["arms"][0]["choices"][0]["reference"]
    );
    // A sub-tolerance first offer must not hide a later improvement against the parent.
    let later = comparison::select(&p, &[offer(3, 1. - 0.5e-12), offer(5, 1. - 1.2e-12)], &all)?;
    assert_eq!(
        later["arms"][0]["choices"][0]["selected"],
        offer(5, 0.).reference()
    );
    p["arms"][0]["choices"][0]["metrics"] = json!({"arbitrary_heldout":123});
    assert_eq!(comparison::select(&p, &offers, &all)?, selected);
    Ok(())
}
#[test]
fn incomplete_parents_and_duplicate_offers_are_rejected() -> Result<()> {
    let all: Vec<_> = (0..35).map(|i| format!("fixture-{i}")).collect();
    let parent = fixtures::parent(&all);
    for kind in 0..5 {
        let mut bad = parent.clone();
        match kind {
            0 => {
                bad["arms"].as_array_mut().unwrap().pop();
            }
            1 => {
                bad["arms"][0]["choices"].as_array_mut().unwrap().pop();
            }
            2 => bad["arms"][0]["choices"][0] = bad["arms"][0]["choices"][1].clone(),
            3 => {
                bad["complete_parent"]["complete_parent"]["complete_parent"]["complete_parent"]["depth_outcomes"]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            }
            _ => {
                bad["complete_parent"]["complete_parent"]["complete_parent"]["complete_parent"]["transformer_endpoints"] =
                    json!([])
            }
        }
        assert!(comparison::select(&bad, &[], &all).is_err());
    }
    let o = comparison::Offer {
        task: all[0].clone(),
        mode: training::Mode::Mini,
        prime: 3,
        train_ce: 0.5,
        model_sha256: "a".repeat(64),
        stage_sha256: "b".repeat(64),
    };
    assert!(comparison::select(&parent, &[o.clone(), o], &all).is_err());
    Ok(())
}
#[test]
fn actual_import_and_metric_corruption_fail_replay() -> Result<()> {
    let f = Fixture::new()?;
    let inv = f.prepare()?;
    let task = "fixture-0";
    f.run(&inv, task)?;
    let path = f.destination.join(task).join("result.json");
    let original: Value = read(&path)?;
    for kind in 0..7 {
        let mut bad = original.clone();
        match kind {
            0 => bad["evaluation"][0]["metrics"]["validation"]["sequence_accuracy"] = json!(0.5),
            1 => bad["evaluation"][0]["train_ce"] = json!(1.),
            2 => bad["evaluation"][0]["offer_eligible"] = json!(false),
            3 => bad["evaluation"][0]["known_inputs"]["validation"]["known_calls"] = json!(999),
            4 => bad["new_construction_work"] = json!(1),
            5 => {
                bad["evaluation"][0]["known_inputs"]["validation"]["active_mask"][0] = json!(false)
            }
            _ => {
                bad["evaluation"][0]["known_inputs"]["validation"]["unmatched_active_calls"] =
                    json!(1)
            }
        }
        overwrite(&path, &bad);
        assert!(
            execute::replay_with(
                &f.destination,
                task,
                &inv,
                &f.parent,
                &f.stage,
                input(task)?
            )
            .is_err()
        );
    }
    overwrite(&path, &original);
    execute::replay_with(
        &f.destination,
        task,
        &inv,
        &f.parent,
        &f.stage,
        input(task)?,
    )?;
    let model = f.request.staging.root.join(task).join("stage-24-p97.json");
    let mut changed: Value = read(&model)?;
    changed["head"]["endpoints"][0]["training"]["model"]["coefficients"][0] = json!(123.);
    overwrite(&model, &changed);
    assert!(
        execute::replay_with(
            &f.destination,
            task,
            &inv,
            &f.parent,
            &f.stage,
            input(task)?
        )
        .is_err()
    );
    // Rehashing the altered file passes the outer byte checks. Independent
    // complex-state replay must still reject the changed coefficient.
    let x = input(task)?;
    let mut record = f.stage["records"][0].clone();
    let mut saved = record["result"]["stages"][24].clone();
    saved["sha256"] = json!(digest(&model)?);
    saved["bytes"] = json!(fs::metadata(&model)?.len());
    record["result"]["stages"][24] = saved.clone();
    imported::load(&f.request.staging, task, &record, &saved, &x, false)?;
    assert!(imported::load(&f.request.staging, task, &record, &saved, &x, true).is_err());
    // An altered complete basis must fail even when every enclosing digest is
    // recomputed and no trained head remains to expose the disagreement.
    changed["discovery"]["grouped"]["field"]["space"]["directions"]
        .as_array_mut()
        .unwrap()
        .pop();
    changed["head"] = Value::Null;
    changed["head_error"] = json!("simulated failure after discovery");
    overwrite(&model, &changed);
    saved["sha256"] = json!(digest(&model)?);
    saved["bytes"] = json!(fs::metadata(&model)?.len());
    saved["retained"]["head_error"] = changed["head_error"].clone();
    record["result"]["stages"][24] = saved.clone();
    imported::load(&f.request.staging, task, &record, &saved, &x, false)?;
    assert!(imported::load(&f.request.staging, task, &record, &saved, &x, true).is_err());
    Ok(())
}
#[test]
fn incompatible_readouts_and_actual_mode_refusals_keep_all_outcomes() -> Result<()> {
    let f = Fixture::new()?;
    let mut x = input("fixture-0")?;
    let source: Value = read(&f.request.staging.root.join("fixture-0/stage-24-p97.json"))?;
    let discovery: discovery::Report = serde_json::from_value(source["discovery"].clone())?;
    let identity: Vec<_> = (0..x.task.vocab).collect();
    assert!(!outcomes::compatible(&discovery, &identity));
    assert!(outcomes::compatible(
        &discovery,
        x.readout.as_ref().unwrap()
    ));
    // Exercise a real per-mode preflight refusal; other modes remain valid.
    let mut policies = x.policies.clone().unwrap();
    policies[0].updates = 4097;
    let report = binding::run(&discovery, policies.clone())?;
    binding::check(&discovery, &policies, &report)?;
    let before = serde_json::to_vec(&report)?;
    let sha = "b".repeat(64);
    assert!(outcomes::offers("fixture-0", 97, &sha, &report, false)?.is_empty());
    assert_eq!(
        outcomes::offers("fixture-0", 97, &sha, &report, true)?.len(),
        2
    );
    let rows = outcomes::evaluate("fixture-0", 97, &sha, &x, &report, false)?;
    outcomes::check("fixture-0", 97, &sha, &x.record, &report, false, &rows)?;
    assert_eq!(rows[0]["status"], "refused");
    for row in &rows[1..] {
        assert_eq!(row["primary_pass"], true);
        assert_eq!(row["offer_eligible"], false);
    }
    assert_eq!(before, serde_json::to_vec(&report)?);
    // Both encoded axes can still form an unmatched complete vector code.
    x.task.splits.get_mut("validation").unwrap()[0].tokens[1] = 6;
    let rows = outcomes::evaluate("fixture-0", 97, &sha, &x, &report, true)?;
    for row in &rows[1..] {
        let mask = &row["known_inputs"]["validation"];
        assert_eq!(mask["known_calls"], 0);
        assert_eq!(mask["unknown_calls"], mask["calls"]);
        assert_eq!(mask["default_calls"], mask["calls"]);
        assert_eq!(mask["unknown_axis_calls"], 0);
        assert_eq!(mask["unmatched_active_calls"], mask["calls"]);
    }
    // A genuinely unseen categorical axis is recorded separately.
    x.task.splits.get_mut("validation").unwrap()[0].tokens[1] = 7;
    let rows = outcomes::evaluate("fixture-0", 97, &sha, &x, &report, true)?;
    for row in &rows[1..] {
        let mask = &row["known_inputs"]["validation"];
        assert_eq!(mask["unknown_axis_calls"], mask["calls"]);
        assert_eq!(mask["unmatched_active_calls"], 0);
        assert_eq!(mask["known_calls"], 0);
    }
    x.task.splits.get_mut("validation").unwrap()[0].tokens[1] = x.task.vocab as i64;
    assert!(outcomes::evaluate("fixture-0", 97, &sha, &x, &report, true).is_err());
    Ok(())
}
#[test]
fn review_pins_process_failures_and_missing_models_are_retained() -> Result<()> {
    let f = Fixture::new()?;
    let inv = f.prepare()?;
    // Input binding applies even when the original construction failed before
    // producing a model; a failed case is not permission to change its data.
    let mut bad_stage = f.stage.clone();
    bad_stage["inputs"][1]["original_batch"] = json!(99);
    assert!(
        imported::case(
            &f.request.staging,
            &bad_stage,
            "fixture-1",
            &input("fixture-1")?
        )
        .is_err()
    );
    fs::write(&f.request.staging.review.exit, "1\n")?;
    assert!(
        parents::load_with(
            &f.request.staging,
            parents::Kind::Kernel,
            &f.expected_stage,
            &f.all,
            &f.planned,
            true
        )
        .is_err()
    );
    fs::write(&f.request.staging.review.exit, "0\n")?;
    let reviewed = fs::read(&f.request.staging.review.commit)?;
    fs::write(&f.request.staging.review.commit, "b".repeat(40))?;
    assert!(
        parents::load_with(
            &f.request.staging,
            parents::Kind::Kernel,
            &f.expected_stage,
            &f.all,
            &f.planned,
            true
        )
        .is_err()
    );
    fs::write(&f.request.staging.review.commit, reviewed)?;
    for task in &f.all {
        if task == "fixture-0" {
            fs::create_dir(f.destination.join(task))?;
            write(
                &f.destination.join(task).join("partial.json"),
                &json!({"partial_metric_work":12}),
            )?;
            fixtures::files(&f.destination, task, false, None)?;
        } else {
            f.run(&inv, task)?;
        }
    }
    let out = summary::summarize_with(&f.destination, &inv, f.parent.clone(), &f.stage, input)?;
    assert_eq!(out["process_failures"], 1);
    assert_eq!(out["records"][0]["prior_choices_retained"], 6);
    assert_eq!(
        out["staged_construction"]["constructor_accounting"],
        f.stage["constructor_accounting"]
    );
    for arm in out["arms"].as_array().unwrap() {
        assert_eq!(arm["choices"].as_array().unwrap().len(), 105);
        assert_eq!(arm["full35_pass"], false);
    }
    let mut parent = f.parent.clone();
    for arm in parent["arms"].as_array_mut().unwrap() {
        arm["choices"][0] = json!({"task":"fixture-0","training_mode":"full_batch","status":"no_model","reference":null,"train_ce":null});
    }
    let choice = comparison::select(&parent, &[], &f.all)?;
    assert!(choice["arms"][0]["choices"][0]["selected"].is_null());
    Ok(())
}
