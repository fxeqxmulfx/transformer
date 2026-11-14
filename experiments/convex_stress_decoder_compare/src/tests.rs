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
fn original_record_binding_retains_repeats_and_ignores_heldout_targets() -> Result<()> {
    let source = tagged::Model::construct(
        5,
        vec![0, 0, 2, 3, 4],
        vec![0, 1, 1, 3, 5],
        vec![0, 1, 2, 1, 0, 2],
    )?;
    let example = |tokens, target| data::Example {
        task: "fixture".into(),
        tokens,
        targets: vec![-100, target],
        changes: vec![false, false],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    };
    let mut task = data::Task {
        vocab: 5,
        selection: "validation".into(),
        splits: BTreeMap::from([
            (
                "train".into(),
                vec![
                    example(vec![0, 0], 0),
                    example(vec![0, 0], 0),
                    example(vec![1, 1], 1),
                ],
            ),
            ("validation".into(), vec![example(vec![4, 4], 1)]),
        ]),
    };
    let entry = json!({"reference":"fixture:0","training_mode":"forward_only",
        "source_gradient_training_run":false,"source_model_sha256":head::digest(&source)?});
    let m = model::construct(source.clone(), &task, &entry)?;
    assert_eq!(m.records().len(), 3);
    assert_eq!(m.records()[0].cell, m.records()[1].cell);
    assert_ne!(m.records()[0].id, m.records()[1].id);
    assert_eq!(m.selection().selected_margin, 24.);
    assert!(model::forecast(&m, &task)? > 0);
    task.splits.get_mut("validation").unwrap()[0].targets[1] = 2;
    assert_eq!(model::construct(source.clone(), &task, &entry)?, m);
    task.splits.get_mut("train").unwrap()[0].targets[1] = 2;
    let changed = model::construct(source.clone(), &task, &entry)?;
    assert_ne!(changed.binding().records_sha256, m.binding().records_sha256);
    assert!(changed.selection().selected_margin < 24.);
    let mut bad = entry;
    bad["source_model_sha256"] = json!("wrong");
    assert!(model::construct(source, &task, &bad).is_err());
    Ok(())
}

#[test]
fn saved_family_import_is_independent_of_scores_and_preserves_all_other_choices() -> Result<()> {
    let (mut p, _) = snapshot();
    for arm in p["arms"].as_array_mut().unwrap() {
        arm["choices"][0]["saved_endpoint"] = json!({"family":"identified_modular",
            "id":"identified-0","model_sha256":"pin","gradient_training_run":false});
    }
    let before = import::eligible(&p, "task0")?;
    assert_eq!(before.len(), 1);
    for arm in p["arms"].as_array_mut().unwrap() {
        arm["choices"][0]["metrics"] = json!({"validation":{"loss":1000.,"token_accuracy":0.}});
        arm["choices"][1]["saved_endpoint"] = json!({"family":"earlier","model_sha256":"other"});
    }
    assert_eq!(import::eligible(&p, "task0")?, before);
    p["arms"][1]["choices"][0]["saved_endpoint"]["model_sha256"] = json!("different");
    assert!(import::eligible(&p, "task0").is_err());
    assert_eq!(raw_accuracy(&json!({"token_accuracy":0.8})), json!(0.8));
    assert_eq!(
        raw_accuracy(&json!({"native_token_accuracy":0.7,"token_accuracy":0.8})),
        json!(0.7)
    );
    Ok(())
}

#[test]
fn native_extraction_requires_exact_reviewed_tensors_and_safe_locations() -> Result<()> {
    let root = std::env::temp_dir().join(format!("decoder-import-{}", std::process::id()));
    fs::create_dir(&root)?;
    let source = tagged::Model::construct(3, vec![0, 1, 2], vec![2, 1, 0], vec![1, 0, 1])?;
    let path = root.join("proposal.json");
    write(&path, &json!({"candidates":[{"model":source}]}))?;
    let selector = ".candidates[0].model";
    assert_eq!(
        import::extract(&path, selector, &head::digest(&source)?)?,
        source
    );
    assert!(import::extract(&path, selector, "wrong").is_err());
    assert!(import::extract(&path, ".candidates[1].model", &head::digest(&source)?).is_err());
    let request = Request {
        producer_commit: "a".repeat(40),
        parent_root: root.clone(),
        completed_manifest: path.clone(),
        parent_review_exit: path.clone(),
        parent_review_commit: path.clone(),
        component_review_exit: path.clone(),
        component_review_commit: path.clone(),
        identified_root: root.clone(),
        consequences_root: root.clone(),
        fibers_root: root.clone(),
    };
    for (family, id, expected) in [
        ("identified_modular", "identified-0", ".candidates[0].model"),
        (
            "decoded_consequences",
            "consequence-2",
            ".candidates[2].value.model",
        ),
        ("fibers", "fibers-1", ".extension.candidates[1].model"),
    ] {
        let entry = json!({"family":family,"id":id});
        assert_eq!(import::location(&request, "task0", &entry)?.1, expected);
        assert!(import::location(&request, "../task0", &entry).is_err());
    }
    assert!(
        import::location(
            &request,
            "task0",
            &json!({"family":"identified_modular","id":"identified-0]|input"})
        )
        .is_err()
    );
    fs::remove_dir_all(root)?;
    Ok(())
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
                .all(|r| r["metric_provenance"]["kind"] == "inherited_reviewed_287")
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
        "metric_provenance":{"kind":"fresh_changed_model"}}]);
    let out = summary::combine(p, &r, &json!("pin"))?;
    assert_eq!(out["changes"].as_array().unwrap().len(), 2);
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
    Ok(())
}
