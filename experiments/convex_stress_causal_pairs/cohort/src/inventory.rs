use super::*;
pub(crate) fn records(suite: &Value, inputs: &[Value]) -> Result<()> {
    let all = tasks(suite)?;
    if inputs.len() != all.len()
        || all
            .iter()
            .any(|n| inputs.iter().filter(|r| r["task"] == *n).count() != 1)
    {
        return Err("all35 original input records required".into());
    }
    Ok(())
}
pub fn prepare(root: &Path, request: Request) -> Result<()> {
    if request.suite_sha256 != hash(SUITE.as_bytes()) {
        return Err("original suite bytes differ".into());
    }
    let suite = suite()?;
    let inputs = tasks(&suite)?
        .iter()
        .map(|n| Ok(inputs::build(&suite, n)?.record))
        .collect::<Result<_>>()?;
    prepare_with(root, request, &suite, inputs)
}
pub(crate) fn prepare_with(
    root: &Path,
    request: Request,
    suite: &Value,
    inputs: Vec<Value>,
) -> Result<()> {
    if root.exists() || !commit(&request.producer_commit) {
        return Err("fresh root and committed producer required".into());
    }
    records(suite, &inputs)?;
    let parent = parent::load(&request.parent, suite, true)?;
    for record in &inputs {
        for arm in parent["arms"].as_array().unwrap() {
            for c in arm["choices"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|c| c["task"] == record["task"])
            {
                if c["status"] == "selected" && c["primary_split"] != record["primary_split"] {
                    return Err("original primary split differs from reviewed parent".into());
                }
            }
        }
    }
    let mut artifacts = vec![];
    parent::tree(&request.parent.root, &mut artifacts)?;
    let staging = request
        .staging
        .as_ref()
        .map(|pin| staging::attach(pin, suite, &inputs, &request.producer_commit))
        .transpose()?;
    fs::create_dir(root)?;
    let snapshots = parent::snapshot(&request.parent, &root.join("parent"))?;
    write(&root.join("request.json"), &request)?;
    write(
        &root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),
        "binary_sha256":digest(&std::env::current_exe()?)?,"producer_commit":request.producer_commit,
        "request_sha256":digest(&root.join("request.json"))?,"request":request,"protocol":protocol()?,
        "suite_value_sha256":hash(&serde_json::to_vec(suite)?),"tasks":tasks(suite)?,"inputs":inputs,
        "parent_snapshots":snapshots,"parent_artifacts":artifacts,"staging":staging}),
    )
}
pub(crate) fn inventory(root: &Path, deep: bool) -> Result<(Value, Value)> {
    let request: Request = read(&root.join("request.json"))?;
    if request.suite_sha256 != hash(SUITE.as_bytes()) {
        return Err("original suite pin changed".into());
    }
    inventory_with(root, &suite()?, deep)
}
pub(crate) fn inventory_with(root: &Path, suite: &Value, deep: bool) -> Result<(Value, Value)> {
    let request: Request = read(&root.join("request.json"))?;
    let inv: Value = read(&root.join("inventory.json"))?;
    if inv["source_fingerprint"] != fingerprint()
        || inv["binary_sha256"] != digest(&std::env::current_exe()?)?
        || inv["producer_commit"] != request.producer_commit
        || !commit(&request.producer_commit)
        || inv["request_sha256"] != digest(&root.join("request.json"))?
        || inv["request"] != json!(request)
        || inv["protocol"] != protocol()?
        || inv["tasks"] != json!(tasks(suite)?)
        || inv["suite_value_sha256"] != hash(&serde_json::to_vec(suite)?)
    {
        return Err("original inventory/source/binary/protocol changed".into());
    }
    records(
        suite,
        inv["inputs"].as_array().ok_or("missing input inventory")?,
    )?;
    match &request.staging {
        Some(pin) => {
            staging::check_attachment(pin, &inv["staging"], suite, deep)?;
            if inv["staging"]["inventory"]["inputs"] != inv["inputs"]
                || inv["staging"]["inventory"]["producer_commit"] != request.producer_commit
            {
                return Err("comparison and staging inputs or producer differ".into());
            }
        }
        None if !inv["staging"].is_null() => return Err("unexpected staging attachment".into()),
        None => {}
    }
    let parent = parent::load(&request.parent, suite, deep)?;
    parent::check_snapshot(&root.join("parent"), &inv["parent_snapshots"])?;
    if read::<Value>(&root.join("parent/summary.json"))? != parent {
        return Err("complete retained parent differs".into());
    }
    if deep {
        let mut artifacts = vec![];
        parent::tree(&request.parent.root, &mut artifacts)?;
        if inv["parent_artifacts"] != json!(artifacts) {
            return Err("parent artifact inventory changed".into());
        }
    }
    Ok((inv, parent))
}
