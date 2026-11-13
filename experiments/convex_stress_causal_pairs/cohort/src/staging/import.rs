//! Import is gated by the original reviewed parent. Pins bind complete stage
//! costs; independent numerical replay precedes every imported selection.
use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub root: PathBuf,
    pub summary_sha256: String,
    pub inventory_sha256: String,
}
fn pinned(pin: &Reference) -> Result<(Value, Value)> {
    if !pin.root.is_absolute()
        || digest(&pin.root.join("summary.json"))? != pin.summary_sha256
        || digest(&pin.root.join("inventory.json"))? != pin.inventory_sha256
    {
        return Err("staged summary or inventory pin differs".into());
    }
    let inv: Value = read(&pin.root.join("inventory.json"))?;
    let summary: Value = read(&pin.root.join("summary.json"))?;
    if inv["source_fingerprint"] != fingerprint()
        || inv["binary_sha256"] != digest(&std::env::current_exe()?)?
        || summary["source_fingerprint"] != fingerprint()
        || summary["inventory_sha256"] != pin.inventory_sha256
        || summary["producer_commit"] != inv["producer_commit"]
    {
        return Err("staged producer or binary differs".into());
    }
    Ok((inv, summary))
}
pub(crate) fn attach(
    pin: &Reference,
    suite: &Value,
    inputs: &[Value],
    producer: &str,
) -> Result<Value> {
    let inv = inventory_with(&pin.root, suite)?;
    let (pinned_inv, summary) = pinned(pin)?;
    if inv != pinned_inv
        || inv["inputs"] != json!(inputs)
        || inv["producer_commit"] != producer
        || summary != summarize_with(&pin.root, &inv)?
    {
        return Err("complete original staging scope or inputs differ".into());
    }
    let mut artifacts = vec![];
    parent::tree(&pin.root, &mut artifacts)?;
    Ok(json!({"reference":pin,"inventory":inv,"summary":summary,"artifacts":artifacts}))
}
pub(crate) fn check_attachment(
    pin: &Reference,
    saved: &Value,
    suite: &Value,
    deep: bool,
) -> Result<()> {
    let (inv, summary) = pinned(pin)?;
    if saved["reference"] != json!(pin)
        || saved["inventory"] != inv
        || saved["summary"] != summary
        || inv != inventory_with(&pin.root, suite)?
    {
        return Err("retained staging attachment differs".into());
    }
    if deep {
        let mut artifacts = vec![];
        parent::tree(&pin.root, &mut artifacts)?;
        if saved["artifacts"] != json!(artifacts) || summary != summarize_with(&pin.root, &inv)? {
            return Err("complete staging artifacts or process evidence changed".into());
        }
    }
    Ok(())
}
fn source(inv: &Value, task: &str) -> Result<Option<(Reference, Value, Value)>> {
    if inv["staging"].is_null() {
        return Ok(None);
    }
    let pin: Reference = serde_json::from_value(inv["staging"]["reference"].clone())?;
    let (stage, summary) = pinned(&pin)?;
    if inv["request"]["staging"] != json!(pin)
        || inv["staging"]["inventory"] != stage
        || inv["staging"]["summary"] != summary
    {
        return Err("comparison staging reference changed".into());
    }
    let record = case_record(&pin.root, task, &stage)?;
    let expected = summary["records"]
        .as_array()
        .ok_or("missing staged outcomes")?
        .iter()
        .filter(|r| r["task"] == task)
        .collect::<Vec<_>>();
    if expected.len() != 1 || *expected[0] != record {
        return Err("staged case state, time, resource or replay evidence changed".into());
    }
    Ok(Some((pin, stage, record)))
}
pub(crate) fn imported(
    dir: &Path,
    task: &str,
    inv: &Value,
    x: &inputs::Inputs,
) -> Result<Option<execute::Attempt>> {
    let Some((pin, stage, record)) = source(inv, task)? else {
        return Ok(None);
    };
    inputs::check(&stage, task, x)?;
    write(
        &dir.join("import.json"),
        &json!({"reference":pin,"record":record}),
    )?;
    if record["status"] != "complete" {
        return Err("staged process or replay failed; retain parent and all staged costs".into());
    }
    let attempt: execute::Attempt = read(&pin.root.join(task).join("attempt.json"))?;
    if read::<Option<binding::Request>>(&pin.root.join(task).join("configuration.json"))?
        != x.configuration
    {
        return Err("staged actual training configuration changed".into());
    }
    execute::check_attempt(x, &attempt)?;
    Ok(Some(attempt))
}
pub(crate) fn retained_import(dir: &Path, task: &str, inv: &Value) -> Result<bool> {
    let Some((pin, _, record)) = source(inv, task)? else {
        if dir.join("import.json").exists() {
            return Err("unexpected staged import".into());
        }
        return Ok(false);
    };
    if record["status"] != "complete"
        || read::<Value>(&dir.join("import.json"))? != json!({"reference":pin,"record":record})
        || digest(&dir.join("attempt.json"))?
            != record["result"]["attempt_sha256"]
                .as_str()
                .ok_or("missing staged report hash")?
        || digest(&dir.join("configuration.json"))?
            != record["result"]["configuration_sha256"]
                .as_str()
                .ok_or("missing staged configuration hash")?
    {
        return Err("imported state is not the exact sealed staged state".into());
    }
    Ok(true)
}
