//! Preserve the complete reviewed319 run and freeze every choice before fitting.
use super::*;

fn verify_reference(root: &Path, manifest: &Path, pin: &Value) -> Result<()> {
    if digest(manifest)? != pin["manifest_sha256"] {
        return Err("reviewed comparison manifest changed".into());
    }
    let mut comparison_files = BTreeSet::new();
    for line in fs::read_to_string(manifest)?.lines() {
        let (sha, relative) = line.split_once("  ").ok_or("invalid comparison manifest")?;
        let relative = Path::new(relative);
        if relative
            .components()
            .any(|part| !matches!(part, std::path::Component::Normal(_)))
            || !comparison_files.insert(relative.to_owned())
            || digest(&root.join(relative))? != sha
        {
            return Err("changed or invalid comparison artifact".into());
        }
    }
    if Some(comparison_files.len() as u64) != pin["artifacts"].as_u64()
        || digest(&root.join("summary.json"))? != pin["summary_sha256"]
    {
        return Err("complete reviewed comparison required".into());
    }
    Ok(())
}
fn verify_comparison(request: &Request, p: &Value) -> Result<()> {
    verify_reference(
        &request.comparison_root,
        &request.comparison_manifest,
        &p["comparison332"],
    )?;
    // The complete332 manifest authenticates this inventory and its retained330
    // snapshot. Recheck every330 artifact as well, without loading its tensors.
    let inv: Value = read(&request.comparison_root.join("inventory.json"))?;
    let prior = &inv["request"];
    verify_reference(
        Path::new(prior["comparison_root"].as_str().ok_or("missing330 root")?),
        Path::new(
            prior["comparison_manifest"]
                .as_str()
                .ok_or("missing330 manifest")?,
        ),
        &p["comparison330"],
    )?;
    for (file, field) in [
        ("comparison330.json", "summary_sha256"),
        ("comparison330-manifest.sha256", "manifest_sha256"),
    ] {
        if digest(&request.comparison_root.join(file))? != p["comparison330"][field] {
            return Err("retained330 copy changed".into());
        }
    }
    Ok(())
}
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
pub(crate) fn retained_raw(
    root: &Path,
    task: &str,
    index: usize,
    split: &str,
    replay: bool,
) -> Result<PathBuf> {
    let inv = inventory(root)?;
    let request: Request = serde_json::from_value(inv["request"].clone())?;
    let relative = format!("{}/model-{index}-{}.jsonl", name(task)?, name(split)?);
    let manifest = root.join("comparison332-manifest.sha256");
    if digest(&manifest)? != protocol()?["comparison332"]["manifest_sha256"] {
        return Err("retained332 manifest changed".into());
    }
    let text = fs::read_to_string(&manifest)?;
    let entries: Vec<_> = text
        .lines()
        .filter_map(|line| line.split_once("  "))
        .filter(|(_, p)| *p == relative)
        .collect();
    if entries.len() != 1 {
        return Err("retained332 raw file absent or duplicated".into());
    }
    let original = request.comparison_root.join(&relative);
    let copy = root
        .join(task)
        .join(format!("reference332-model-{index}-{split}.jsonl"));
    if digest(&original)? != entries[0].0 {
        return Err("reviewed332 raw logits changed".into());
    }
    if !replay {
        fs::copy(&original, &copy)?;
    }
    if digest(&copy)? != entries[0].0 {
        return Err("copied332 raw logits changed".into());
    }
    Ok(copy)
}
pub fn prepare(root: &Path, request: Request) -> Result<()> {
    if root.exists() || request.producer_commit.len() != 40 {
        return Err("fresh root and producer commit required".into());
    }
    let p = protocol()?;
    verify_comparison(&request, &p)?;
    verify_reference(
        &request.component_root,
        &request.component_manifest,
        &p["component_reference"],
    )?;
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
    let summary_buffers = 32
        * (fs::metadata(&baseline_path)?.len()
            + fs::metadata(request.comparison_root.join("summary.json"))?
                .len()
                .max(fs::metadata(request.comparison_root.join("comparison330.json"))?.len()))
        + (2u64 << 30);
    if summary_buffers > BUFFER_LIMIT {
        return Err("combined summary parse allowance refused".into());
    }
    if digest(&baseline_path)? != p["parent_summary_sha256"]
        || digest(&request.component_root.join("summary.json"))? != p["component_summary_sha256"]
        || digest(&request.diff_gpt_history)? != p["diff_gpt_history_sha256"]
    {
        return Err("reviewed baseline/component/history changed".into());
    }
    for (key, sha) in p["outcomes"].as_object().ok_or("missing component pins")? {
        let path = request.component_root.join(key).join("outcome.json");
        if digest(&path)? != *sha
            || Some(fs::metadata(&path)?.len()) != p["outcome_bytes"][key].as_u64()
        {
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
        request.comparison_root.join("summary.json"),
        root.join("comparison332.json"),
    )?;
    fs::copy(
        &request.comparison_manifest,
        root.join("comparison332-manifest.sha256"),
    )?;
    for file in ["comparison330.json", "comparison330-manifest.sha256"] {
        fs::copy(request.comparison_root.join(file), root.join(file))?;
    }
    fs::copy(
        request.component_root.join("summary.json"),
        root.join("component335.json"),
    )?;
    fs::copy(
        &request.component_manifest,
        root.join("component335-manifest.sha256"),
    )?;
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
        "tasks":35,"cases":cases,"parent_artifacts_checked":manifest.len(),"comparison332_artifacts_checked":163,"comparison330_artifacts_checked":156,"component335_artifacts_checked":56,
        "separate_summary_phase_buffer_bytes":summary_buffers,
        "all_costs_before_fitting_and_scoring":true}),
    )
}
