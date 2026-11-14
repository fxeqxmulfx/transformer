use super::*;

fn fixture() -> Value {
    let modes = ["full_batch", "minibatch", "forward_only"];
    let choices: Vec<_> = (0..35)
        .flat_map(|t| {
            modes.map(move |mode| {
                json!({"task":format!("task{t}"),"training_mode":mode,
            "reference":format!("local:{t}:{mode}"),"model_sha256":"a".repeat(64),
            "metrics":{"validation":{"loss":0.2,"token_accuracy":0.8}},
            "primary_split":"validation","primary_pass":true})
            })
        })
        .collect();
    let records: Vec<_> = (0..35).map(|t| json!({"task":format!("task{t}"),
        "status":"complete","outcomes":if t==0 {modes.map(|mode|
            json!({"task":"task0","training_mode":mode,"metrics":{"validation":{"loss":0.2}},
                "model_info":{"new_optimizer_updates":3,"new_gradient_batches":5,"new_local_row_updates":7}})).to_vec()} else {vec![]}})).collect();
    json!({"tasks":35,"choices_per_arm":105,"process_failures":0,"records":records,
        "source_fingerprint":"fixture","new_optimizer_updates":9,"new_gradient_batches":15,
        "new_local_row_updates":21,"changes":[{"before_ce":0.1,"after_ce":0.2}],
        "arms":[{"initializer":"minimum_ce","choices":choices},
            {"initializer":"first_small_ce","choices":choices}],
        "baselines":{"transformer_endpoints":(0..35).collect::<Vec<_>>(),
            "depth_outcomes":(0..336).collect::<Vec<_>>()}})
}

#[test]
fn negative_reference_preserves_every_choice_metric_and_fresh_cost() -> Result<()> {
    let dir =
        std::env::temp_dir().join(format!("count-reference-projection-{}", std::process::id()));
    fs::create_dir(&dir)?;
    let path = dir.join("summary.json");
    let original = fixture();
    write(&path, &original)?;
    let projected = reference::project(&path)?;
    assert_eq!(projected["records"], original["records"]);
    assert_eq!(projected["changes"], original["changes"]);
    assert_eq!(projected["new_local_row_updates"], 21);
    for (old, new) in original["arms"]
        .as_array()
        .unwrap()
        .iter()
        .zip(projected["arms"].as_array().unwrap())
    {
        for (old, new) in old["choices"]
            .as_array()
            .unwrap()
            .iter()
            .zip(new["choices"].as_array().unwrap())
        {
            for key in [
                "task",
                "training_mode",
                "reference",
                "model_sha256",
                "metrics",
            ] {
                assert_eq!(old[key], new[key]);
            }
        }
    }
    let mut bad = projected.clone();
    bad["records"][0]["status"] = json!("failed");
    assert!(reference::validate(&bad).is_err());
    let mut bad = projected.clone();
    bad["arms"][1] = bad["arms"][0].clone();
    assert!(reference::validate(&bad).is_err());
    let mut bad = projected;
    bad["arms"][0]["choices"][0] = bad["arms"][0]["choices"][1].clone();
    assert!(reference::validate(&bad).is_err());
    let mut missing = original;
    missing["records"].as_array_mut().unwrap().pop();
    fs::write(&path, serde_json::to_vec(&missing)?)?;
    assert!(reference::project(&path).is_err());
    fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn reference_requires_review_manifest_and_every_artifact_to_match() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("count-reference-pins-{}", std::process::id()));
    fs::create_dir(&dir)?;
    for (file, value) in [
        ("summary.json", fixture()),
        ("inventory.json", json!({"fixture":true})),
        ("result.json", json!({"negative_ce":true})),
    ] {
        write(&dir.join(file), &value)?;
    }
    let mut manifest = String::new();
    for file in ["summary.json", "inventory.json", "result.json"] {
        let path = dir.join(file);
        manifest.push_str(&format!("{}  {}\n", digest(&path)?, path.display()));
    }
    fs::write(dir.join("completed.sha256"), manifest)?;
    fs::write(dir.join("review.exit"), "0\n")?;
    fs::write(dir.join("review.commit"), "reviewed\n")?;
    let request = Request {
        producer_commit: "a".repeat(40),
        parent_root: PathBuf::new(),
        completed_manifest: PathBuf::new(),
        parent_review_exit: PathBuf::new(),
        parent_review_commit: PathBuf::new(),
        component_review_exit: PathBuf::new(),
        component_review_commit: PathBuf::new(),
        local_reference_root: dir.clone(),
        local_reference_manifest: dir.join("completed.sha256"),
        local_reference_review_exit: dir.join("review.exit"),
        local_reference_review_commit: dir.join("review.commit"),
    };
    let protocol = json!({"local_reference_review_commit":"reviewed",
        "local_reference_summary_sha256":digest(&dir.join("summary.json"))?,
        "local_reference_inventory_sha256":digest(&dir.join("inventory.json"))?,
        "local_reference_completed_manifest_sha256":digest(&request.local_reference_manifest)?,
        "local_reference_source_fingerprint":"fixture"});
    let prepared = reference::prepare(&request, &protocol)?;
    assert_eq!(prepared["changes"][0]["after_ce"], 0.2);
    for (file, bad) in [
        ("review.exit", "1\n"),
        ("review.commit", "other\n"),
        ("summary.json", "{}"),
        ("inventory.json", "{}"),
        ("result.json", "{}"),
        ("completed.sha256", ""),
    ] {
        let path = dir.join(file);
        let bytes = fs::read(&path)?;
        fs::write(&path, bad)?;
        assert!(
            reference::prepare(&request, &protocol).is_err(),
            "accepted {file}"
        );
        fs::write(&path, bytes)?;
    }
    fs::remove_dir_all(dir)?;
    Ok(())
}
