//! Retain the reviewed block-local regression without using it for model selection.
//! Only the CM45 parent supplies eligibility, training records and actual models.
use super::*;
const PROJECT: &str = include_str!("../reference.jq");

pub(crate) fn validate(value: &Value) -> Result<()> {
    if value["tasks"] != 35
        || value["choices_per_arm"] != 105
        || value["process_failures"] != 0
        || value["conventional_transformer_endpoints"] != 35
        || value["depth_outcomes"] != 336
    {
        return Err("complete reviewed local-reference scope required".into());
    }
    let records = value["records"].as_array().ok_or("missing local records")?;
    let tasks: BTreeSet<_> = records.iter().map(|r| r["task"].as_str()).collect();
    if records.len() != 35
        || tasks.len() != 35
        || tasks.contains(&None)
        || records.iter().any(|r| r["status"] != "complete")
    {
        return Err("all35 distinct complete local-reference records required".into());
    }
    let arms = value["arms"].as_array().ok_or("missing local policies")?;
    if arms.len() != 2
        || arms
            .iter()
            .map(|a| a["initializer"].as_str())
            .collect::<BTreeSet<_>>()
            != BTreeSet::from([Some("minimum_ce"), Some("first_small_ce")])
    {
        return Err("both local-reference policies required".into());
    }
    for arm in arms {
        let choices = arm["choices"].as_array().ok_or("missing local choices")?;
        let mut seen = BTreeSet::new();
        for choice in choices {
            let task = choice["task"].as_str().ok_or("missing local task")?;
            let mode = choice["training_mode"]
                .as_str()
                .ok_or("missing local mode")?;
            import::mode(&choice["training_mode"])?;
            if !tasks.contains(&Some(task))
                || !seen.insert((task, mode))
                || !choice["metrics"].is_object()
                || !choice["reference"].is_string()
            {
                return Err("missing or duplicate local-reference choice/metrics".into());
            }
        }
        if choices.len() != 105 {
            return Err("all105 local-reference choices required".into());
        }
    }
    Ok(())
}

pub(crate) fn project(path: &Path) -> Result<Value> {
    let projected = std::process::Command::new("jq")
        .args(["-c", "-S", PROJECT])
        .arg(path)
        .output()?;
    if !projected.status.success() {
        return Err("reviewed local-reference projection failed".into());
    }
    let value = serde_json::from_slice(&projected.stdout)?;
    validate(&value)?;
    Ok(value)
}

pub(crate) fn prepare(request: &Request, protocol: &Value) -> Result<Value> {
    if !request.local_reference_root.is_absolute()
        || fs::read_to_string(&request.local_reference_review_exit)?.trim() != "0"
        || fs::read_to_string(&request.local_reference_review_commit)?.trim()
            != protocol["local_reference_review_commit"]
                .as_str()
                .ok_or("missing reference review pin")?
    {
        return Err("exact-commit local-reference review required".into());
    }
    for (file, key) in [
        ("summary.json", "local_reference_summary_sha256"),
        ("inventory.json", "local_reference_inventory_sha256"),
    ] {
        if digest(&request.local_reference_root.join(file))? != protocol[key] {
            return Err("frozen local-reference summary or inventory changed".into());
        }
    }
    let files = parent::read_manifest(
        &request.local_reference_manifest,
        &protocol["local_reference_completed_manifest_sha256"],
    )?;
    for path in files.keys() {
        parent::pinned(Path::new(path), &files)?;
    }
    let path = request.local_reference_root.join("summary.json");
    parent::pinned(&path, &files)?;
    parent::pinned(&request.local_reference_root.join("inventory.json"), &files)?;
    let mut value = project(&path)?;
    if value["source_fingerprint"] != protocol["local_reference_source_fingerprint"] {
        return Err("local-reference source fingerprint differs".into());
    }
    value["evidence"] = json!({"summary":path,
        "summary_sha256":protocol["local_reference_summary_sha256"],
        "completed_manifest":request.local_reference_manifest,
        "completed_manifest_sha256":protocol["local_reference_completed_manifest_sha256"],
        "review_commit":protocol["local_reference_review_commit"],
        "role":"retained negative comparison; never a model-selection input"});
    Ok(value)
}
