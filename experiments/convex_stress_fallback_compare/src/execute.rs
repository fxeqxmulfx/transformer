use super::*;
pub fn prepare(root: &Path, request: Request) -> Result<()> {
    let suite = suite()?;
    let all = tasks(&suite)?;
    let (pi, parent) = parents::load(&request.parent, false, &all, true)?;
    let (si, stage) = parents::load(&request.staging, true, &all, true)?;
    let records = all
        .iter()
        .map(|t| Ok(inputs::build(&suite, t)?.record))
        .collect::<Result<Vec<_>>>()?;
    prepare_with(root, request, &suite, records, &pi, &parent, &si, &stage)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_with(
    root: &Path,
    request: Request,
    suite: &Value,
    records: Vec<Value>,
    pi: &Value,
    parent: &Value,
    si: &Value,
    stage: &Value,
) -> Result<()> {
    if root.exists()
        || !commit(&request.producer_commit)
        || request.suite_sha256 != hash(SUITE.as_bytes())
    {
        return Err("fresh destination, committed producer and unchanged suite required".into());
    }
    let all = tasks(suite)?;
    comparison::validate(parent, &all)?;
    parents::attachment(parent, stage, &request.staging)?;
    if records.len() != 35
        || records.iter().zip(&all).any(|(r, t)| r["task"] != *t)
        || si["inputs"] != json!(records)
        || pi["request"]["suite_sha256"] != request.suite_sha256
        || si["request"]["suite_sha256"] != request.suite_sha256
    {
        return Err("original inputs, stage or suite differ".into());
    }
    for r in &records {
        for policy in POLICIES {
            for m in MODES {
                let old = comparison::cell(parent, policy, r["task"].as_str().unwrap(), m)?;
                if old["status"] == "selected" && old["primary_split"] != r["primary_split"] {
                    return Err("original primary success split changed".into());
                }
            }
        }
    }
    let checkpoints = parents::checkpoints(&request.checkpoint_root, parent, &all)?;
    fs::create_dir(root)?;
    let parent_snapshots = parents::snapshot(&request.parent, &root.join("parent"))?;
    let staging_snapshots = parents::snapshot(&request.staging, &root.join("staging"))?;
    let (mut parent_artifacts, mut staging_artifacts) = (vec![], vec![]);
    parents::tree(&request.parent.root, &mut parent_artifacts)?;
    parents::tree(&request.staging.root, &mut staging_artifacts)?;
    write(&root.join("request.json"), &request)?;
    write(
        &root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),"binary_sha256":digest(&std::env::current_exe()?)?,
        "producer_commit":request.producer_commit,"request":request,"request_sha256":digest(&root.join("request.json"))?,
        "protocol":protocol()?,"suite_value_sha256":hash(&serde_json::to_vec(suite)?),"tasks":all,"inputs":records,
        "parent_snapshots":parent_snapshots,"staging_snapshots":staging_snapshots,"parent_artifacts":parent_artifacts,"staging_artifacts":staging_artifacts,
        "predecessor_checkpoints":checkpoints}),
    )
}
pub(crate) fn inventory(root: &Path, deep: bool) -> Result<(Value, Value, Value)> {
    let request: Request = read(&root.join("request.json"))?;
    let suite = suite()?;
    let all = tasks(&suite)?;
    let (pi, parent) = parents::load(&request.parent, false, &all, deep)?;
    let (si, stage) = parents::load(&request.staging, true, &all, deep)?;
    let inv = inventory_with(root, &suite, &pi, &parent, &si, &stage, deep)?;
    Ok((inv, parent, stage))
}
pub(crate) fn inventory_with(
    root: &Path,
    suite: &Value,
    pi: &Value,
    parent: &Value,
    si: &Value,
    stage: &Value,
    deep: bool,
) -> Result<Value> {
    let inv: Value = read(&root.join("inventory.json"))?;
    let request: Request = read(&root.join("request.json"))?;
    let all = tasks(suite)?;
    if inv["source_fingerprint"] != fingerprint()
        || inv["binary_sha256"] != digest(&std::env::current_exe()?)?
        || inv["producer_commit"] != request.producer_commit
        || !commit(&request.producer_commit)
        || request.suite_sha256 != hash(SUITE.as_bytes())
        || inv["request"] != json!(request)
        || inv["request_sha256"] != digest(&root.join("request.json"))?
        || inv["protocol"] != protocol()?
        || inv["suite_value_sha256"] != hash(&serde_json::to_vec(suite)?)
        || inv["tasks"] != json!(all)
        || si["inputs"] != inv["inputs"]
        || pi["request"]["suite_sha256"] != request.suite_sha256
        || si["request"]["suite_sha256"] != request.suite_sha256
        || inv["predecessor_checkpoints"]
            != parents::checkpoints(&request.checkpoint_root, parent, &all)?
    {
        return Err("comparison inventory, source, checkpoint or scope changed".into());
    }
    comparison::validate(parent, &all)?;
    parents::attachment(parent, stage, &request.staging)?;
    for (folder, pin, summary, key) in [
        ("parent", &request.parent, parent, "parent_snapshots"),
        ("staging", &request.staging, stage, "staging_snapshots"),
    ] {
        parents::check_snapshot(&root.join(folder), &inv[key])?;
        if read::<Value>(&root.join(folder).join("summary.json"))? != *summary
            || digest(&pin.root.join("summary.json"))? != pin.summary_sha256
        {
            return Err("complete foreign summary snapshot differs".into());
        }
        if deep {
            let mut files = vec![];
            parents::tree(&pin.root, &mut files)?;
            if inv[format!("{folder}_artifacts")] != json!(files) {
                return Err("foreign artifact inventory changed".into());
            }
        }
    }
    Ok(inv)
}
fn metadata(inv: &Value, x: &inputs::Inputs) -> Value {
    json!({"source_fingerprint":fingerprint(),"binary_sha256":inv["binary_sha256"],"producer_commit":inv["producer_commit"],"protocol":inv["protocol"],"input":x.record,
        "parent_summary_sha256":inv["request"]["parent"]["summary_sha256"],"staging_summary_sha256":inv["request"]["staging"]["summary_sha256"]})
}
fn selection_seal(dir: &Path, selection: &Value) -> Result<Value> {
    Ok(
        json!({"imports_sha256":digest(&dir.join("imports.json"))?,"metadata_sha256":digest(&dir.join("metadata.json"))?,"selection":selection,"heldout_inputs":false}),
    )
}
fn bundle(
    inv: &Value,
    parent: &Value,
    stage: &Value,
    task: &str,
    x: &inputs::Inputs,
) -> Result<imported::Bundle> {
    inputs::check(inv, task, x)?;
    let request: Request = serde_json::from_value(inv["request"].clone())?;
    imported::build(
        &request.checkpoint_root,
        &request.staging,
        stage,
        parent,
        task,
        x,
    )
}
pub fn run_case(root: &Path, task: &str) -> Result<()> {
    let (inv, parent, stage) = inventory(root, false)?;
    run_with(
        root,
        task,
        &inv,
        &parent,
        &stage,
        &inputs::build(&suite()?, task)?,
    )
}
pub(crate) fn run_with(
    root: &Path,
    task: &str,
    inv: &Value,
    parent: &Value,
    stage: &Value,
    x: &inputs::Inputs,
) -> Result<()> {
    let dir = root.join(name(task)?);
    fs::create_dir(&dir)?;
    write(&dir.join("metadata.json"), &metadata(inv, x))?;
    let b = bundle(inv, parent, stage, task, x)?;
    write(&dir.join("imports.json"), &b.rows)?;
    for row in &b.rows {
        if row["status"] == "wrapped" {
            let key = row["reference"].as_str().unwrap();
            write(
                &dir.join(row["model_file"].as_str().unwrap()),
                b.models.get(key).ok_or("missing wrapper")?,
            )?;
        }
    }
    let selection = comparison::transform(parent, task, &b.rows)?;
    let seal = selection_seal(&dir, &selection)?;
    write(&dir.join("selection.json"), &seal)?;
    // Every structural reference and actual model digest is sealed first.
    let evaluation = outcomes::evaluate(x, &b)?;
    write(
        &dir.join("result.json"),
        &json!({"metadata":metadata(inv,x),"selection":seal,"selection_sha256":digest(&dir.join("selection.json"))?,
        "evaluation":evaluation,"new_discovery_work":0,"new_gradient_updates":0,"unseen_identification":false}),
    )
}
pub(crate) fn retained(
    root: &Path,
    task: &str,
    inv: &Value,
    parent: &Value,
    stage: &Value,
    x: &inputs::Inputs,
) -> Result<Value> {
    let b = bundle(inv, parent, stage, task, x)?;
    let dir = root.join(name(task)?);
    if read::<Vec<Value>>(&dir.join("imports.json"))? != b.rows {
        return Err("imported actual heads, support, raw CE or refusal changed".into());
    }
    for row in &b.rows {
        if row["status"] == "wrapped" {
            let path = dir.join(row["model_file"].as_str().unwrap());
            if digest(&path)? != row["model_sha256"].as_str().ok_or("missing wrapper hash")? {
                return Err("saved wrapper bytes changed".into());
            }
            let input = &x
                .configuration
                .as_ref()
                .ok_or("missing original input")?
                .discovery
                .input;
            let expected = b
                .models
                .get(row["reference"].as_str().unwrap())
                .ok_or("missing actual wrapper")?;
            let model = fallback::Model::load(read(&path)?, input, expected.binding())?;
            if serde_json::to_value(model)? != serde_json::to_value(expected)? {
                return Err("saved wrapper is not the replayed actual endpoint".into());
            }
        }
    }
    let result: Value = read(&dir.join("result.json"))?;
    let selection = comparison::transform(parent, task, &b.rows)?;
    if read::<Value>(&dir.join("metadata.json"))? != metadata(inv, x)
        || result["metadata"] != metadata(inv, x)
        || result["selection"] != selection_seal(&dir, &selection)?
        || read::<Value>(&dir.join("selection.json"))? != result["selection"]
        || result["selection_sha256"] != digest(&dir.join("selection.json"))?
        || result["new_discovery_work"] != 0
        || result["new_gradient_updates"] != 0
        || result["unseen_identification"] != false
        || result["evaluation"] != json!(outcomes::evaluate(x, &b)?)
    {
        return Err("structural selection, actual original metrics or masks changed".into());
    }
    Ok(result)
}
pub(crate) fn seal(root: &Path, task: &str, inv: &Value) -> Result<Value> {
    let dir = root.join(name(task)?);
    Ok(
        json!({"task":task,"replayed":true,"source_fingerprint":fingerprint(),"parent_summary_sha256":inv["request"]["parent"]["summary_sha256"],
        "result_sha256":digest(&dir.join("result.json"))?,"imports_sha256":digest(&dir.join("imports.json"))?,"selection_sha256":digest(&dir.join("selection.json"))?,
        "actual_states_and_histories_replayed":true,"all_original_metrics_and_masks_replayed":true,"unseen_identification":false}),
    )
}
pub fn replay_case(root: &Path, task: &str) -> Result<Value> {
    let (inv, parent, stage) = inventory(root, false)?;
    retained(
        root,
        task,
        &inv,
        &parent,
        &stage,
        &inputs::build(&suite()?, task)?,
    )?;
    seal(root, task, &inv)
}
