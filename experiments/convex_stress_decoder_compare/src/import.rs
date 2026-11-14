//! Import exact selected phase tensors. Eligibility uses saved representation
//! and identity, never a held-out metric or an arithmetic operation name.
use super::*;

pub(crate) fn eligible(snapshot: &Value, task: &str) -> Result<Vec<Value>> {
    let families = protocol()?["eligible_families"]
        .as_array()
        .ok_or("missing eligible families")?
        .clone();
    let mut found = BTreeMap::new();
    for arm in snapshot["arms"].as_array().ok_or("missing policies")? {
        for choice in arm["choices"].as_array().ok_or("missing choices")? {
            if choice["task"] != task || !families.contains(&choice["saved_endpoint"]["family"]) {
                continue;
            }
            let saved = &choice["saved_endpoint"];
            let reference = choice["reference"]
                .as_str()
                .ok_or("missing selected reference")?;
            let entry = json!({"reference":reference,"training_mode":choice["training_mode"],
                "family":saved["family"],"id":saved["id"],"source_model_sha256":saved["model_sha256"],
                "source_gradient_training_run":saved["gradient_training_run"]});
            if let Some(previous) = found.insert(reference.to_owned(), entry.clone())
                && previous != entry
            {
                return Err("policies disagree on an inherited phase-model identity".into());
            }
        }
    }
    Ok(found.into_values().collect())
}

pub(crate) fn location(request: &Request, task: &str, entry: &Value) -> Result<(PathBuf, String)> {
    name(task)?;
    let (root, prefix, filename, before, after) = match entry["family"].as_str() {
        Some("identified_modular") => (
            &request.identified_root,
            "identified-",
            "proposal.json",
            ".candidates[",
            "].model",
        ),
        Some("decoded_consequences") => (
            &request.consequences_root,
            "consequence-",
            "bank.json",
            ".candidates[",
            "].value.model",
        ),
        Some("fibers") => (
            &request.fibers_root,
            "fibers-",
            "report.json",
            ".extension.candidates[",
            "].model",
        ),
        _ => return Err("unsupported saved source representation".into()),
    };
    let index = entry["id"]
        .as_str()
        .and_then(|s| s.strip_prefix(prefix))
        .ok_or("invalid phase candidate ID")?
        .parse::<usize>()?;
    if !root.is_absolute() {
        return Err("absolute phase source root required".into());
    }
    Ok((
        root.join(task).join(filename),
        format!("{before}{index}{after}"),
    ))
}

pub(crate) fn extract(path: &Path, selector: &str, expected: &str) -> Result<tagged::Model> {
    let output = std::process::Command::new("jq")
        .args(["-c", selector])
        .arg(path)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "source extraction failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let source: tagged::Model = serde_json::from_slice(&output.stdout)?;
    source.validate()?;
    if head::digest(&source)? != expected {
        return Err("actual source model differs from reviewed selected model hash".into());
    }
    Ok(source)
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
    let mut result = None;
    for arm in snapshot["arms"].as_array().ok_or("missing policies")? {
        for choice in arm["choices"].as_array().ok_or("missing choices")? {
            if choice["task"] == task && choice["reference"] == reference {
                if let Some(old) = result {
                    let old: &Value = old;
                    if old["metrics"] != choice["metrics"]
                        || old["training_mode"] != choice["training_mode"]
                    {
                        return Err("inherited policies disagree on source scores or mode".into());
                    }
                }
                result = Some(choice);
            }
        }
    }
    result.ok_or_else(|| "selected source reference missing from parent".into())
}
