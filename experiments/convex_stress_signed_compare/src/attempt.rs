//! Preserve the complete failed v1 attempt, including its sealed partial model.
//! Historical failure metadata is never an input to coefficient selection.
use super::*;

fn exit_code(path: &Path, files: &BTreeMap<String, String>) -> Result<i32> {
    parent::pinned(path, files)?;
    Ok(fs::read_to_string(path)?.trim().parse()?)
}

pub(crate) fn prepare(request: &Request, protocol: &Value) -> Result<Value> {
    inspect(
        &request.previous_attempt_root,
        &request.previous_attempt_manifest,
        protocol,
        "previous_attempt",
        1,
    )
}

pub(crate) fn prepare_signed(request: &Request, protocol: &Value) -> Result<Value> {
    inspect(
        &request.signed_attempt_root,
        &request.signed_attempt_manifest,
        protocol,
        "signed_attempt",
        0,
    )
}

fn inspect(
    root: &Path,
    manifest: &Path,
    protocol: &Value,
    prefix: &str,
    models: usize,
) -> Result<Value> {
    if !root.is_absolute() {
        return Err("absolute previous-attempt root required".into());
    }
    let files = parent::read_manifest(manifest, &protocol[format!("{prefix}_manifest_sha256")])?;
    for path in files.keys() {
        parent::pinned(Path::new(path), &files)?;
    }
    let inventory = root.join("inventory.json");
    parent::pinned(&inventory, &files)?;
    let inv: Value = read(&inventory)?;
    if inv["request"]["producer_commit"] != protocol[format!("{prefix}_producer")]
        || inv["source_fingerprint"] != protocol[format!("{prefix}_source_fingerprint")]
    {
        return Err("previous attempt source identity differs".into());
    }
    let suite: Value = serde_json::from_str(SUITE)?;
    let tasks: Vec<_> = suite["tasks"]
        .as_array()
        .ok_or("missing suite")?
        .iter()
        .map(|t| t["name"].clone())
        .collect();
    if inv["tasks"] != json!(tasks) || tasks.len() != 35 {
        return Err("complete previous35task scope required".into());
    }
    let mut records = Vec::new();
    let mut stored_models = Vec::new();
    for task in &tasks {
        let task = name(task.as_str().ok_or("invalid old task")?)?;
        let run = exit_code(&root.join(format!("{task}.run.exit")), &files)?;
        let replay_path = root.join(format!("{task}.replay.exit"));
        let replay = if replay_path.exists() {
            Some(exit_code(&replay_path, &files)?)
        } else {
            None
        };
        if run == 0 && replay.is_none() {
            return Err("successful historical run missing replay".into());
        }
        let log = root.join(format!("{task}.log"));
        parent::pinned(&log, &files)?;
        records.push(json!({"task":task,"run_exit":run,"replay_exit":replay,
            "replay_status":if replay.is_some(){"attempted"}else{"not_started_after_run_failure"},
            "failure_log":if run!=0 {Some(fs::read_to_string(log)?)}else{None}}));
        let cases: Vec<_> = inv["cases"]
            .as_array()
            .ok_or("missing previous cases")?
            .iter()
            .filter(|c| c["task"] == task)
            .collect();
        if cases.len() != 1 {
            return Err("missing or duplicated previous task".into());
        }
        for (i, entry) in cases[0]["eligible_models"]
            .as_array()
            .ok_or("missing old eligibility")?
            .iter()
            .enumerate()
        {
            let path = root.join(task).join(format!("model-{i}.json"));
            if path.exists() {
                parent::pinned(&path, &files)?;
                stored_models.push(json!({"task":task,"training_mode":entry["training_mode"],"path":path,"sha256":digest(&path)?}));
            }
        }
    }
    let successful_runs = records.iter().filter(|r| r["run_exit"] == 0).count();
    let successful_replays = records.iter().filter(|r| r["replay_exit"] == 0).count();
    let summary_exit = exit_code(&root.join("summary.exit"), &files)?;
    if successful_runs != 34
        || successful_replays != 34
        || stored_models.len() != models
        || summary_exit == 0
    {
        return Err("previous refused-attempt accounting differs".into());
    }
    Ok(
        json!({"status":"failed_work_forecast_before_complete_selection","tasks":35,
        "successful_runs":successful_runs,"successful_replays":successful_replays,
        "summary_exit":summary_exit,"new_heldout_metrics_available":false,
        "producer_commit":protocol[format!("{prefix}_producer")],
        "source_fingerprint":protocol[format!("{prefix}_source_fingerprint")],
        "manifest":manifest,"manifest_sha256":protocol[format!("{prefix}_manifest_sha256")],
        "records":records,"stored_models":stored_models}),
    )
}
