//! Preserve the complete reviewed319 run and freeze every choice before fitting.
use super::*;

pub(crate) fn entries(snapshot: &Value, task: &str) -> Result<Vec<Value>> {
    let p = protocol()?;
    let prefix = p["eligible_prefix"]
        .as_str()
        .ok_or("missing eligibility prefix")?;
    let mut found = BTreeMap::new();
    for arm in snapshot["arms"].as_array().ok_or("missing policies")? {
        for choice in arm["choices"].as_array().ok_or("missing choices")? {
            let reference = choice["reference"].as_str().ok_or("missing reference")?;
            if choice["task"] != task || !reference.starts_with(prefix) {
                continue;
            }
            policy(choice["training_mode"].as_str().ok_or("missing mode")?)?;
            if let Some(prior) = found.insert(reference.to_owned(), choice.clone())
                && prior != *choice
            {
                return Err("policies disagree on selected state or metrics".into());
            }
        }
    }
    Ok(found.into_values().collect())
}
pub(crate) fn dataset(task: &str) -> Result<data::Task> {
    let suite: Value = serde_json::from_str(SUITE)?;
    let rows: Vec<_> = suite["tasks"]
        .as_array()
        .ok_or("missing suite")?
        .iter()
        .filter(|r| r["name"] == task)
        .collect();
    if rows.len() != 1 {
        return Err("missing or duplicate original task".into());
    }
    data::build(rows[0])
}
pub(crate) fn inventory(root: &Path) -> Result<Value> {
    let inv: Value = read(&root.join("inventory.json"))?;
    if inv["source_fingerprint"] != fingerprint() || inv["protocol"] != protocol()? {
        return Err("frozen runner or protocol changed".into());
    }
    Ok(inv)
}
pub fn prepare(root: &Path, request: Request) -> Result<()> {
    if root.exists() || request.producer_commit.len() != 40 {
        return Err("fresh root and producer commit required".into());
    }
    let p = protocol()?;
    for artifact in p["previous_refused_preparation"]["artifacts"]
        .as_array()
        .ok_or("missing prior refusal")?
    {
        if digest(Path::new(
            artifact["file"]
                .as_str()
                .ok_or("missing refusal artifact")?,
        ))? != artifact["sha256"]
        {
            return Err("prior refused preparation evidence changed".into());
        }
    }
    if digest(&request.parent_manifest)? != p["parent_manifest_sha256"] {
        return Err("complete319 manifest changed".into());
    }
    let mut manifest = BTreeMap::new();
    for line in fs::read_to_string(&request.parent_manifest)?.lines() {
        let (sha, path) = line.split_once("  ").ok_or("invalid parent manifest")?;
        let path = PathBuf::from(path);
        if !path.starts_with(&request.parent_root)
            || digest(&path)? != sha
            || manifest.insert(path, sha.to_owned()).is_some()
        {
            return Err("changed or duplicate319 artifact".into());
        }
    }
    if manifest.len() != 331 {
        return Err("all331 reviewed319 artifacts required".into());
    }
    let baseline_path = request.parent_root.join("summary.json");
    if digest(&baseline_path)? != p["parent_summary_sha256"]
        || digest(&request.component_root.join("summary.json"))? != p["component_summary_sha256"]
        || digest(&request.diff_gpt_history)? != p["diff_gpt_history_sha256"]
    {
        return Err("reviewed baseline/component/history changed".into());
    }
    for (key, sha) in p["outcomes"].as_object().ok_or("missing component pins")? {
        if digest(&request.component_root.join(key).join("outcome.json"))? != *sha {
            return Err("reviewed component state changed".into());
        }
    }
    let snapshot: Value = read(&baseline_path)?;
    counts::summarize(&snapshot)?;
    if snapshot["producer_commit"] != p["parent_producer"]
        || snapshot["source_fingerprint"] != p["parent_source_fingerprint"]
        || snapshot["baselines"]["transformer_endpoints"]
            .as_array()
            .map(Vec::len)
            != Some(35)
        || snapshot["baselines"]["depth_outcomes"]
            .as_array()
            .map(Vec::len)
            != Some(336)
    {
        return Err("incomplete original baseline".into());
    }
    fs::create_dir(root)?;
    fs::create_dir(root.join("sources"))?;
    fs::copy(&baseline_path, root.join("baseline.json"))?;
    fs::copy(
        &request.parent_manifest,
        root.join("parent-manifest.sha256"),
    )?;
    fs::copy(
        &request.diff_gpt_history,
        root.join("diff-gpt-history.json"),
    )?;
    let mut cases = Vec::new();
    for record in snapshot["records"].as_array().ok_or("missing cases")? {
        let task = name(record["task"].as_str().ok_or("missing task")?)?;
        let choices = entries(&snapshot, task)?;
        let mut models = Vec::new();
        if !choices.is_empty() {
            let data = dataset(task)?;
            let seal_path = request.parent_root.join(task).join("selection-seal.json");
            if !manifest.contains_key(&seal_path) {
                return Err("source seal absent from manifest".into());
            }
            let seal: Value = read(&seal_path)?;
            for choice in choices {
                let refs: Vec<_> = seal["models"]
                    .as_array()
                    .ok_or("missing old seals")?
                    .iter()
                    .filter(|s| {
                        choice["reference"]
                            == format!("signed-kernel:{}", s["reference"].as_str().unwrap_or(""))
                    })
                    .collect();
                if refs.len() != 1 || refs[0]["model_sha256"] != choice["model_sha256"] {
                    return Err("actual selected319 tensor missing".into());
                }
                let source = request
                    .parent_root
                    .join(task)
                    .join(refs[0]["file"].as_str().ok_or("missing file")?);
                let sha = choice["model_sha256"]
                    .as_str()
                    .ok_or("missing state hash")?;
                if manifest.get(&source).map(String::as_str) != Some(sha) {
                    return Err("selected state absent from complete manifest".into());
                }
                let relative = format!("sources/{sha}.json");
                fs::copy(&source, root.join(&relative))?;
                let value: Value = read(&source)?;
                let mode = choice["training_mode"].as_str().ok_or("missing mode")?;
                let forecasts = costs::forecast(
                    &value,
                    sha,
                    &data,
                    mode,
                    fs::metadata(&source)?.len(),
                    fs::metadata(&baseline_path)?.len(),
                )?;
                models.push(json!({"reference":choice["reference"],"training_mode":mode,"source_model_sha256":sha,
                    "source_file":relative,"previous":choice,"forecast":forecasts}));
            }
        }
        let work: u64 = models
            .iter()
            .map(|m| m["forecast"]["total_kernel_work"].as_u64().unwrap())
            .sum();
        if work > WORK_LIMIT {
            return Err(format!("complete case forecast {work} exceeds {WORK_LIMIT}").into());
        }
        cases.push(json!({"task":task,"models":models,"total_kernel_work":work}));
    }
    write(
        &root.join("inventory.json"),
        &json!({"request":request,"protocol":p,"source_fingerprint":fingerprint(),
        "tasks":35,"cases":cases,"parent_artifacts_checked":manifest.len(),"all_costs_before_fitting_and_scoring":true}),
    )
}
