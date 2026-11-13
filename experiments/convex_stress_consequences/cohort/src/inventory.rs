use super::*;
pub(super) fn inventory(root: &Path) -> Result<(Value, Value)> {
    let inv: Value = read(&root.join("inventory.json"))?;
    let p = protocol()?;
    if inv["source_fingerprint"] != fingerprint()
        || inv["binary_sha256"] != digest(&std::env::current_exe()?)?
        || inv["protocol"] != p
        || inv["tasks"] != serde_json::to_value(tasks()?)?
        || inv["parent_sha256"] != p["parent_summary_sha256"]
        || inv["parent_sha256"] != digest(&root.join("parent.json"))?
    {
        return Err("source/binary/protocol or complete parent binding changed".into());
    }
    let parent = read(&root.join("parent.json"))?;
    parent::original(&parent)?;
    Ok((inv, parent))
}
pub fn prepare(root: &Path, parent_root: &Path) -> Result<()> {
    if root.exists() {
        return Err("consequences destination already exists".into());
    }
    let producer = std::env::var("CONSEQUENCES_PRODUCER_COMMIT")?;
    if producer.len() != 40 || !producer.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid committed consequence producer".into());
    }
    let p = protocol()?;
    if p["shared_fingerprint"] != convex_stress_relabel::cohort::fingerprint()
        || p["parent_summary_sha256"] != digest(&parent_root.join("summary.json"))?
    {
        return Err("original source or complete task223 parent differs from freeze".into());
    }
    let parent: Value = read(&parent_root.join("summary.json"))?;
    parent::original(&parent)?;
    let old: Value = read(&parent_root.join("inventory.json"))?;
    if old["source_fingerprint"] != p["parent_source_fingerprint"]
        || old["producer_commit"] != p["parent_producer_commit"]
        || old["tasks"] != serde_json::to_value(tasks()?)?
    {
        return Err("original task223 inventory differs".into());
    }
    let mut records = vec![];
    for name in tasks()? {
        let x = inputs(&name)?;
        parent::choices(&parent, &name, &x.record)?;
        let r = parent::record(&parent, &name)?;
        let dir = parent_root.join(&name);
        let old_input: Value = read(&dir.join("input.json"))?;
        if r["seal"]["result_sha256"] != digest(&dir.join("result.json"))?
            || read::<Value>(&dir.join("result.json"))? != r["result"]
            || read::<Value>(&parent_root.join(format!("{name}.replay.json")))? != r["seal"]
            || fs::read_to_string(parent_root.join(format!("{name}.run.exit")))?.trim() != "0"
            || fs::read_to_string(parent_root.join(format!("{name}.replay.exit")))?.trim() != "0"
            || r["result"]["selection"]["input_sha256"] != digest(&dir.join("input.json"))?
            || old_input["categorical"]["n"] != x.input.categorical.n
            || old_input["categorical"]["vocab"] != x.input.categorical.vocabulary
            || old_input["categorical"]["observations"]
                != serde_json::to_value(&x.input.categorical.observations)?
        {
            return Err("original parent artifact/replay/observation binding differs".into());
        }
        records.push(x.record);
    }
    fs::create_dir(root)?;
    fs::copy(parent_root.join("summary.json"), root.join("parent.json"))?;
    if digest(&root.join("parent.json"))? != p["parent_summary_sha256"] {
        return Err("parent changed while copying".into());
    }
    write(
        &root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),
        "binary_sha256":digest(&std::env::current_exe()?)?,"producer_commit":producer,"protocol":p,
        "tasks":tasks()?,"inputs":records,"parent_sha256":p["parent_summary_sha256"],
        "identified_parent_root":fs::canonicalize(parent_root)?}),
    )
}
