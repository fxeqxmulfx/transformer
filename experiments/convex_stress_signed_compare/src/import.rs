//! Bind current selected context tensors, not nested stale saved metadata.
//! Eligibility depends only on representation and identity, never task scores.
use super::*;

pub(crate) fn eligible(snapshot: &Value, task: &str) -> Result<Vec<Value>> {
    let p = protocol()?;
    let prefix = p["eligible_prefix"]
        .as_str()
        .ok_or("missing representation prefix")?;
    let mut found = BTreeMap::new();
    for arm in snapshot["arms"].as_array().ok_or("missing policies")? {
        for choice in arm["choices"].as_array().ok_or("missing choices")? {
            let reference = choice["reference"]
                .as_str()
                .ok_or("missing selected reference")?;
            if choice["task"] != task || !reference.starts_with(prefix) {
                continue;
            }
            let sha = choice["model_sha256"]
                .as_str()
                .ok_or("missing actual selected model hash")?;
            if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("invalid actual context model hash".into());
            }
            mode(&choice["training_mode"])?;
            let item = json!({"reference":reference,"training_mode":choice["training_mode"],"source_model_sha256":sha});
            if let Some(old) = found.insert(reference.to_owned(), item.clone())
                && old != item
            {
                return Err("policies disagree on actual selected context identity".into());
            }
        }
    }
    Ok(found.into_values().collect())
}

pub(crate) fn location(request: &Request, task: &str, entry: &Value) -> Result<PathBuf> {
    name(task)?;
    if !request.parent_root.is_absolute() {
        return Err("absolute context root required".into());
    }
    let files = parent::manifest(request)?;
    let seal_path = request.parent_root.join(task).join("selection-seal.json");
    parent::pinned(&seal_path, &files)?;
    let seal: Value = read(&seal_path)?;
    let mut paths = Vec::new();
    for (i, item) in seal["models"]
        .as_array()
        .ok_or("missing context models")?
        .iter()
        .enumerate()
    {
        let reference = item["reference"]
            .as_str()
            .ok_or("missing context reference")?;
        if entry["reference"] != format!("crossing-exclusion:{reference}") {
            continue;
        }
        if item["model_sha256"] != entry["source_model_sha256"]
            || item["file"] != format!("model-{i}.json")
        {
            return Err("actual selected context hash or filename differs".into());
        }
        let path = request
            .parent_root
            .join(task)
            .join(format!("model-{i}.json"));
        parent::pinned(&path, &files)?;
        paths.push(path);
    }
    if paths.len() != 1 {
        return Err("context source reference absent or repeated".into());
    }
    Ok(paths.remove(0))
}

pub(crate) fn mode(value: &Value) -> Result<head::Mode> {
    match value.as_str() {
        Some("full_batch") => Ok(head::Mode::Full),
        Some("minibatch") => Ok(head::Mode::Mini),
        Some("forward_only") => Ok(head::Mode::Forward),
        _ => Err("unsupported original training mode".into()),
    }
}

pub(crate) fn previous<'a>(snapshot: &'a Value, task: &str, reference: &str) -> Result<&'a Value> {
    let mut result: Option<&Value> = None;
    for arm in snapshot["arms"].as_array().ok_or("missing policies")? {
        for choice in arm["choices"].as_array().ok_or("missing choices")? {
            if choice["task"] == task && choice["reference"] == reference {
                if let Some(old) = result
                    && (old["metrics"] != choice["metrics"]
                        || old["training_mode"] != choice["training_mode"])
                {
                    return Err("policies disagree on prior context metrics".into());
                }
                result = Some(choice);
            }
        }
    }
    result.ok_or_else(|| "selected context absent from parent".into())
}
