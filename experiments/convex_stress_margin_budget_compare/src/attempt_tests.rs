use super::*;

#[test]
fn refused_attempt_keeps_all_jobs_the_partial_model_and_missing_replay_reason() -> Result<()> {
    let root = std::env::temp_dir().join(format!("margin-prior-attempt-{}", std::process::id()));
    fs::create_dir(&root)?;
    let suite: Value = serde_json::from_str(SUITE)?;
    let tasks: Vec<_> = suite["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].clone())
        .collect();
    let mut paths = Vec::new();
    let mut cases = Vec::new();
    for (i, task) in tasks.iter().enumerate() {
        let task = task.as_str().unwrap();
        for (suffix, bytes) in [
            ("run.exit", if i == 0 { "1\n" } else { "0\n" }),
            (
                "log",
                if i == 0 {
                    "work forecast refused\n"
                } else {
                    ""
                },
            ),
        ] {
            let path = root.join(format!("{task}.{suffix}"));
            fs::write(&path, bytes)?;
            paths.push(path);
        }
        if i != 0 {
            let path = root.join(format!("{task}.replay.exit"));
            fs::write(&path, "0\n")?;
            paths.push(path);
        } else {
            fs::create_dir(root.join(task))?;
            let path = root.join(task).join("model-0.json");
            write(&path, &json!({"frozen":"checkpoint"}))?;
            paths.push(path);
        }
        cases.push(json!({"task":task,"eligible_models":if i==0 {vec![json!({"training_mode":"forward_only"})]}else{vec![]}}));
    }
    let inventory = root.join("inventory.json");
    write(
        &inventory,
        &json!({"request":{"producer_commit":"old"},"source_fingerprint":"source","tasks":tasks,"cases":cases}),
    )?;
    paths.push(inventory);
    let summary = root.join("summary.exit");
    fs::write(&summary, "1\n")?;
    paths.push(summary);
    let manifest = root.join("completed.sha256");
    let text = paths
        .iter()
        .map(|p| Ok(format!("{}  {}\n", digest(p)?, p.display())))
        .collect::<Result<Vec<_>>>()?
        .join("");
    fs::write(&manifest, text)?;
    let request = Request {
        producer_commit: String::new(),
        parent_root: PathBuf::new(),
        completed_manifest: PathBuf::new(),
        parent_review_exit: PathBuf::new(),
        parent_review_commit: PathBuf::new(),
        component_review_exit: PathBuf::new(),
        component_review_commit: PathBuf::new(),
        local_reference_root: PathBuf::new(),
        local_reference_manifest: PathBuf::new(),
        local_reference_review_exit: PathBuf::new(),
        local_reference_review_commit: PathBuf::new(),
        previous_attempt_root: root.clone(),
        previous_attempt_manifest: manifest.clone(),
    };
    let p = json!({"previous_attempt_manifest_sha256":digest(&manifest)?,"previous_attempt_producer":"old","previous_attempt_source_fingerprint":"source"});
    let actual = attempt::prepare(&request, &p)?;
    assert_eq!(actual["records"].as_array().unwrap().len(), 35);
    assert_eq!(actual["successful_runs"], 34);
    assert_eq!(actual["successful_replays"], 34);
    assert_eq!(actual["records"][0]["replay_exit"], Value::Null);
    assert_eq!(
        actual["records"][0]["replay_status"],
        "not_started_after_run_failure"
    );
    assert_eq!(actual["stored_models"].as_array().unwrap().len(), 1);
    assert_eq!(actual["new_heldout_metrics_available"], false);
    for path in [&paths[0], &paths[2], &manifest] {
        let old = fs::read(path)?;
        fs::write(path, b"changed")?;
        assert!(attempt::prepare(&request, &p).is_err());
        fs::write(path, old)?;
    }
    let mut bad = p;
    bad["previous_attempt_source_fingerprint"] = json!("other");
    assert!(attempt::prepare(&request, &bad).is_err());
    fs::remove_dir_all(root)?;
    Ok(())
}
