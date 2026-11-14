use super::*;

fn snapshot() -> (Value, Vec<Value>) {
    let results = (0..35)
        .map(|t| json!({"task":format!("task{t}"),"status":"complete","outcomes":[]}))
        .collect::<Vec<_>>();
    let choices=(0..35).flat_map(|t| ["full_batch","minibatch","forward_only"].map(move |mode|
        json!({"task":format!("task{t}"),"training_mode":mode,"reference":format!("old:{t}:{mode}"),"status":"selected",
            "primary_split":"validation","primary_pass":true,"all_splits_pass":true,"metrics":{"validation":{"sequence_accuracy":1.,"native_token_accuracy":0.6,"loss":8.}}}))).collect::<Vec<_>>();
    (
        json!({"tasks":35,"choices_per_arm":105,"process_failures":0,"arms":[
        {"initializer":"minimum_ce","choices":choices},{"initializer":"first_small_ce","choices":choices}],
        "baselines":{"transformer_endpoints":(0..35).collect::<Vec<_>>(),"depth_outcomes":(0..336).collect::<Vec<_>>()}}),
        results,
    )
}

#[test]
fn complete_scope_keeps_both_policies_and_explicit_inheritance() -> Result<()> {
    let (p, r) = snapshot();
    let out = summary::combine(p.clone(), &r, &json!("pin"))?;
    assert_eq!(out["baselines"], p["baselines"]);
    for arm in out["arms"].as_array().unwrap() {
        assert_eq!(arm["choices"].as_array().unwrap().len(), 105);
        assert_eq!(arm["full35_pass"], true);
        assert!(
            arm["choices"]
                .as_array()
                .unwrap()
                .iter()
                .all(|r| r["metric_provenance"]["kind"] == "inherited_reviewed_296")
        );
    }
    Ok(())
}

#[test]
fn raw_ce_improvement_does_not_override_primary_success_or_hide_regression() -> Result<()> {
    let (p, mut r) = snapshot();
    r[0]["outcomes"] = json!([{"task":"task0","parent_reference":"old:0:full_batch","reference":"new:0:full_batch",
        "training_mode":"full_batch","primary_pass":false,"all_splits_pass":false,"train_ce":0.01,
        "metrics":{"validation":{"sequence_accuracy":0.98,"native_token_accuracy":0.9,"loss":0.2}},
        "model_info":{"new_optimizer_updates":3,"new_gradient_batches":5,"new_local_row_updates":7},
        "metric_provenance":{"kind":"fresh_changed_model"}}]);
    let out = summary::combine(p, &r, &json!("pin"))?;
    assert_eq!(out["changes"].as_array().unwrap().len(), 2);
    assert_eq!(out["new_optimizer_updates"], 3);
    assert_eq!(out["new_gradient_batches"], 5);
    assert_eq!(out["new_local_row_updates"], 7);
    for arm in out["arms"].as_array().unwrap() {
        assert_eq!(arm["full35_pass"], false);
        assert_eq!(arm["by_mode"][0]["primary_passes"], 34);
    }
    assert!(
        out["changes"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["after_ce"] == 0.2 && c["after_primary_pass"] == false)
    );
    Ok(())
}

#[test]
fn incomplete_duplicate_failed_or_unselected_cases_cannot_inherit_success() {
    let (p, r) = snapshot();
    assert!(summary::combine(p.clone(), &r[..34], &json!("pin")).is_err());
    let mut bad = r.clone();
    bad[0] = bad[1].clone();
    assert!(summary::combine(p.clone(), &bad, &json!("pin")).is_err());
    let mut bad = r.clone();
    bad[0]["status"] = json!("failed");
    assert!(summary::combine(p.clone(), &bad, &json!("pin")).is_err());
    let mut bad = r;
    bad[0]["outcomes"] = json!([{"task":"task0","parent_reference":"missing"}]);
    assert!(summary::combine(p, &bad, &json!("pin")).is_err());
}

#[test]
fn original_success_threshold_and_raw_metric_are_distinct() -> Result<()> {
    let metrics = json!({"train":{"sequence_accuracy":1.},"validation":{"sequence_accuracy":0.99,"native_token_accuracy":0.4,"loss":8.}});
    assert_eq!(passes(&metrics, "validation")?, (true, true));
    assert!(passes(&metrics, "missing").is_err());
    assert!(name("../escape").is_err());
    assert!(name("x/y").is_err());
    Ok(())
}

#[test]
fn native_source_fingerprint_matches_reviewed_parent_and_fixed_protocol() -> Result<()> {
    let p = protocol()?;
    assert_eq!(p["parent_summary_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(p["parent_inventory_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(fingerprint().len(), 64);
    assert_eq!(p["case_scalar_work_limit"], CASE_WORK_CAP);
    assert_eq!(p["case_buffer_limit_bytes"], CASE_BUFFER_LIMIT);
    assert_eq!(p["per_component_scalar_work_limit"], head::WORK_CAP);
    Ok(())
}
