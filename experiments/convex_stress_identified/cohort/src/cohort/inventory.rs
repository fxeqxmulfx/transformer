use super::*;
pub(super) fn inventory(root: &Path) -> Result<(Value, Value)> {
    let inv: Value = read(&root.join("inventory.json"))?;
    if inv["source_fingerprint"] != fingerprint()
        || inv["binary_sha256"] != binary_hash()?
        || inv["protocol"] != protocol()?
        || inv["tasks"] != serde_json::to_value(tasks()?)?
    {
        return Err("identification inventory source/binary/protocol differs".into());
    }
    let raw = bytes(&root.join("parent.json"))?;
    if inv["parent_sha256"] != hash(&raw)
        || inv["parent_sha256"] != protocol()?["parent_summary_sha256"]
    {
        return Err("complete parent binding changed".into());
    }
    let parent: Value = serde_json::from_slice(&raw)?;
    parent::original(&parent)?;
    Ok((inv, parent))
}
pub fn prepare(root: &Path, parent_root: &Path) -> Result<()> {
    if root.exists() {
        return Err("identification destination already exists".into());
    }
    let producer = std::env::var("IDENTIFIED_PRODUCER_COMMIT")?;
    if producer.len() != 40 || !producer.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid committed identification producer".into());
    }
    let p = protocol()?;
    policy()?;
    if p["shared_fingerprint"] != convex_stress_relabel::cohort::fingerprint() {
        return Err("original data/metric source changed".into());
    }
    let raw = bytes(&parent_root.join("summary.json"))?;
    if p["parent_summary_sha256"] != hash(&raw) {
        return Err("complete full105 parent differs from freeze".into());
    }
    let parent: Value = serde_json::from_slice(&raw)?;
    parent::original(&parent)?;
    let inv: Value = read(&parent_root.join("inventory.json"))?;
    if inv["source_fingerprint"] != p["parent_source_fingerprint"]
        || inv["producer_commit"] != p["parent_producer_commit"]
        || inv["tasks"] != serde_json::to_value(tasks()?)?
    {
        return Err("original modular inventory differs".into());
    }
    let mut records = vec![];
    for name in tasks()? {
        let x = inputs(&name)?;
        parent::choices(&parent, &name, &x.record)?;
        let row = parent::record(&parent, &name)?;
        let dir = parent_root.join(&name);
        let old_result = bytes(&dir.join("result.json"))?;
        if row["seal"]["result_sha256"] != hash(&old_result)
            || serde_json::from_slice::<Value>(&old_result)? != row["result"]
            || read::<Value>(&parent_root.join(format!("{name}.replay.json")))? != row["seal"]
            || fs::read_to_string(parent_root.join(format!("{name}.run.exit")))?.trim() != "0"
            || fs::read_to_string(parent_root.join(format!("{name}.replay.exit")))?.trim() != "0"
            || hash(&bytes(&dir.join("input.json"))?) != x.record["training_input_sha256"]
        {
            return Err("original modular artifacts/replay/input changed".into());
        }
        parent::proposal(&bytes(&dir.join("report.json"))?, &parent, &name, &x)?;
        records.push(x.record);
    }
    fs::create_dir(root)?;
    write_raw(&root.join("parent.json"), &raw)?;
    write(
        &root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),"binary_sha256":binary_hash()?,
        "producer_commit":producer,"protocol":p,"tasks":tasks()?,"inputs":records,"parent_sha256":hash(&raw),
        "modular_parent_root":fs::canonicalize(parent_root)?}),
    )
}
