use super::*;
fn check_records(suite: &Value, planned: &[String], records: &[Value]) -> Result<()> {
    let all = tasks(suite)?;
    if planned.is_empty()
        || planned.iter().collect::<BTreeSet<_>>().len() != planned.len()
        || records.len() != planned.len()
    {
        return Err("missing or repeated original input scope".into());
    }
    for task in planned {
        name(task)?;
        if !all.contains(task) || records.iter().filter(|r| r["task"] == *task).count() != 1 {
            return Err("missing original input record".into());
        }
    }
    Ok(())
}
pub fn prepare(root: &Path, request: Request) -> Result<()> {
    if request.suite_sha256 != hash(SUITE.as_bytes()) {
        return Err("original suite bytes differ".into());
    }
    let suite = suite()?;
    let planned = planned()?;
    let records = planned
        .iter()
        .map(|name| Ok(inputs::build(&suite, name)?.record))
        .collect::<Result<_>>()?;
    prepare_with(root, request, &suite, &planned, records)
}
pub(super) fn prepare_with(
    root: &Path,
    request: Request,
    suite: &Value,
    planned: &[String],
    records: Vec<Value>,
) -> Result<()> {
    if root.exists() || !commit(&request.producer_commit) {
        return Err("fresh root and committed producer required".into());
    }
    check_records(suite, planned, &records)?;
    let parent = parent::load(&request.parent, suite, true)?;
    let all = tasks(suite)?;
    for record in &records {
        for arm in parent["arms"].as_array().ok_or("missing parent arms")? {
            for c in arm["choices"]
                .as_array()
                .ok_or("missing parent choices")?
                .iter()
                .filter(|c| c["task"] == record["task"])
            {
                if c["status"] == "selected" && c["primary_split"] != record["primary_split"] {
                    return Err("original primary split differs from reviewed parent".into());
                }
            }
        }
    }
    fs::create_dir(root)?;
    let snapshots = parent::snapshot(&request.parent, &root.join("parent"))?;
    write(&root.join("request.json"), &request)?;
    write(
        &root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),
        "binary_sha256":digest(&std::env::current_exe()?)?,"producer_commit":request.producer_commit,
        "request_sha256":digest(&root.join("request.json"))?,"request":request,"protocol":protocol()?,
        "suite_value_sha256":hash(&serde_json::to_vec(suite)?),"tasks":all,"planned":planned,
        "inputs":records,"parent_snapshots":snapshots}),
    )
}
pub(super) fn inventory(root: &Path, deep: bool) -> Result<(Value, Value)> {
    let request: Request = read(&root.join("request.json"))?;
    if request.suite_sha256 != hash(SUITE.as_bytes()) {
        return Err("original suite pin changed".into());
    }
    inventory_with(root, &suite()?, &planned()?, deep)
}
pub(super) fn inventory_with(
    root: &Path,
    suite: &Value,
    planned: &[String],
    deep: bool,
) -> Result<(Value, Value)> {
    let inv: Value = read(&root.join("inventory.json"))?;
    let request: Request = read(&root.join("request.json"))?;
    if inv["source_fingerprint"] != fingerprint()
        || inv["binary_sha256"] != digest(&std::env::current_exe()?)?
        || inv["producer_commit"] != request.producer_commit
        || !commit(&request.producer_commit)
        || inv["request_sha256"] != digest(&root.join("request.json"))?
        || inv["request"] != serde_json::to_value(&request)?
        || inv["protocol"] != protocol()?
        || inv["tasks"] != serde_json::to_value(tasks(suite)?)?
        || inv["planned"] != serde_json::to_value(planned)?
        || inv["suite_value_sha256"] != hash(&serde_json::to_vec(suite)?)
    {
        return Err("original inventory/source/binary/request scope changed".into());
    }
    let parent = parent::load(&request.parent, suite, deep)?;
    check_records(
        suite,
        planned,
        inv["inputs"].as_array().ok_or("missing original inputs")?,
    )?;
    parent::check_snapshot(&root.join("parent"), &inv["parent_snapshots"])?;
    if read::<Value>(&root.join("parent/summary.json"))? != parent {
        return Err("complete saved parent differs".into());
    }
    Ok((inv, parent))
}
