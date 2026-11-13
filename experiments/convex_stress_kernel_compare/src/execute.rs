use super::*;
fn planned() -> Result<Vec<String>> {
    Ok(serde_json::from_value(protocol()?["planned"].clone())?)
}
pub fn prepare(root: &Path, request: Request) -> Result<()> {
    if root.exists()
        || !commit(&request.producer_commit)
        || request.suite_sha256 != hash(SUITE.as_bytes())
    {
        return Err("fresh destination, committed producer and unchanged suite required".into());
    }
    let suite = suite()?;
    let all = tasks(&suite)?;
    let planned = planned()?;
    let (parent_inv, parent) =
        parents::load(&request.parent, parents::Kind::Parent, &all, &planned, true)?;
    let (stage_inv, stage) = parents::load(
        &request.staging,
        parents::Kind::Kernel,
        &all,
        &planned,
        true,
    )?;
    let records = all
        .iter()
        .map(|t| Ok(inputs::build(&suite, t, &planned)?.record))
        .collect::<Result<Vec<_>>>()?;
    prepare_with(
        root, request, &suite, &planned, records, parent_inv, parent, stage_inv, stage,
    )
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_with(
    root: &Path,
    request: Request,
    suite: &Value,
    planned: &[String],
    records: Vec<Value>,
    parent_inv: Value,
    parent: Value,
    stage_inv: Value,
    stage: Value,
) -> Result<()> {
    if root.exists()
        || !commit(&request.producer_commit)
        || request.suite_sha256 != hash(SUITE.as_bytes())
    {
        return Err("fresh original comparison required".into());
    }
    let all = tasks(suite)?;
    comparison::validate(&parent, &all)?;
    if records.len() != all.len()
        || records.iter().zip(&all).any(|(r, t)| r["task"] != *t)
        || stage["planned"] != json!(planned)
        || stage_inv["request"]["suite_sha256"] != request.suite_sha256
        || parent_inv["request"]["suite_sha256"] != request.suite_sha256
    {
        return Err("original inputs or shared discovery ancestry differ".into());
    }
    for r in &records {
        for a in parent["arms"].as_array().unwrap() {
            for old in a["choices"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|c| c["task"] == r["task"])
            {
                if old["status"] == "selected" && old["primary_split"] != r["primary_split"] {
                    return Err("original primary success split changed".into());
                }
            }
        }
    }
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
        "protocol":protocol()?,"suite_value_sha256":hash(&serde_json::to_vec(suite)?),"tasks":all,"planned":planned,"inputs":records,
        "parent_snapshots":parent_snapshots,"staging_snapshots":staging_snapshots,"parent_artifacts":parent_artifacts,"staging_artifacts":staging_artifacts}),
    )
}
pub(crate) fn inventory(root: &Path, deep: bool) -> Result<(Value, Value, Value)> {
    let request: Request = read(&root.join("request.json"))?;
    let suite = suite()?;
    let all = tasks(&suite)?;
    let planned = planned()?;
    let (parent_inv, parent) =
        parents::load(&request.parent, parents::Kind::Parent, &all, &planned, deep)?;
    let (stage_inv, stage) = parents::load(
        &request.staging,
        parents::Kind::Kernel,
        &all,
        &planned,
        deep,
    )?;
    inventory_with(
        root,
        &suite,
        &planned,
        &parent_inv,
        parent,
        &stage_inv,
        stage,
        deep,
    )
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn inventory_with(
    root: &Path,
    suite: &Value,
    planned: &[String],
    parent_inv: &Value,
    parent: Value,
    stage_inv: &Value,
    stage: Value,
    deep: bool,
) -> Result<(Value, Value, Value)> {
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
        || inv["planned"] != json!(planned)
        || stage_inv["request"]["suite_sha256"] != request.suite_sha256
        || parent_inv["request"]["suite_sha256"] != request.suite_sha256
    {
        return Err("comparison inventory, source or scope changed".into());
    }
    comparison::validate(&parent, &all)?;
    for (folder, pin, summary, key) in [
        ("parent", &request.parent, &parent, "parent_snapshots"),
        ("staging", &request.staging, &stage, "staging_snapshots"),
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
                return Err("complete foreign artifact inventory changed".into());
            }
        }
    }
    Ok((inv, parent, stage))
}
fn metadata(inv: &Value, x: &inputs::Inputs) -> Value {
    json!({"source_fingerprint":fingerprint(),"binary_sha256":inv["binary_sha256"],"producer_commit":inv["producer_commit"],"protocol":inv["protocol"],"input":x.record,"parent_summary_sha256":inv["request"]["parent"]["summary_sha256"],"staging_summary_sha256":inv["request"]["staging"]["summary_sha256"]})
}
fn candidates(
    pin: &Reference,
    task: &str,
    record: &Value,
    x: &inputs::Inputs,
    deep: bool,
) -> Result<Vec<comparison::Offer>> {
    let mut offers = vec![];
    if record["status"] == "complete" {
        for saved in record["result"]["stages"]
            .as_array()
            .ok_or("missing staged catalogue")?
        {
            let s = imported::load(pin, task, record, saved, x, deep)?;
            if let Some(r) = s.attempt.head {
                offers.extend(outcomes::offers(task, s.prime, &s.sha, &r, s.eligible)?);
            }
        }
    }
    Ok(offers)
}
fn evaluate(pin: &Reference, task: &str, record: &Value, x: &inputs::Inputs) -> Result<Vec<Value>> {
    let mut out = vec![];
    if record["status"] == "complete" {
        for saved in record["result"]["stages"]
            .as_array()
            .ok_or("missing staged catalogue")?
        {
            let s = imported::load(pin, task, record, saved, x, false)?;
            if let Some(r) = s.attempt.head {
                out.extend(outcomes::evaluate(
                    task, s.prime, &s.sha, x, &r, s.eligible,
                )?);
            }
        }
    }
    Ok(out)
}
fn selection_seal(dir: &Path, selection: &Value) -> Result<Value> {
    Ok(
        json!({"offers_sha256":digest(&dir.join("offers.json"))?,"import_sha256":digest(&dir.join("import.json"))?,"selection":selection,"heldout_inputs":false}),
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
        inputs::build(&suite()?, task, &planned()?)?,
    )
}
pub(crate) fn run_with(
    root: &Path,
    task: &str,
    inv: &Value,
    parent: &Value,
    stage: &Value,
    x: inputs::Inputs,
) -> Result<()> {
    inputs::check(inv, task, &x)?;
    let pin: Reference = serde_json::from_value(inv["request"]["staging"].clone())?;
    let record = imported::case(&pin, stage, task, &x)?;
    let dir = root.join(name(task)?);
    fs::create_dir(&dir)?;
    write(&dir.join("metadata.json"), &metadata(inv, &x))?;
    write(&dir.join("import.json"), &record)?;
    let offers = candidates(&pin, task, &record, &x, true)?;
    write(&dir.join("offers.json"), &offers)?;
    let selection = comparison::select(
        parent,
        &offers,
        &serde_json::from_value::<Vec<String>>(inv["tasks"].clone())?,
    )?;
    let seal = selection_seal(&dir, &selection)?;
    write(&dir.join("selection.json"), &seal)?;
    // Every selected reference is sealed before the first held-out score.
    let evaluation = evaluate(&pin, task, &record, &x)?;
    write(
        &dir.join("result.json"),
        &json!({"metadata":metadata(inv,&x),"selection_sha256":digest(&dir.join("selection.json"))?,"selection":seal,
        "evaluation":evaluation,"eligible_offers":offers.len(),"new_construction_work":0,"new_discovery_work":0,"unseen_identification":false}),
    )
}
pub(crate) fn retained(
    root: &Path,
    task: &str,
    inv: &Value,
    parent: &Value,
    stage: &Value,
    x: &inputs::Inputs,
    deep: bool,
) -> Result<(Value, Vec<comparison::Offer>)> {
    inputs::check(inv, task, x)?;
    let pin: Reference = serde_json::from_value(inv["request"]["staging"].clone())?;
    let record = imported::case(&pin, stage, task, x)?;
    let dir = root.join(name(task)?);
    let result: Value = read(&dir.join("result.json"))?;
    let offers = candidates(&pin, task, &record, x, deep)?;
    let selected = comparison::select(
        parent,
        &offers,
        &serde_json::from_value::<Vec<String>>(inv["tasks"].clone())?,
    )?;
    if read::<Value>(&dir.join("metadata.json"))? != metadata(inv, x)
        || result["metadata"] != metadata(inv, x)
        || read::<Value>(&dir.join("import.json"))? != record
        || read::<Vec<comparison::Offer>>(&dir.join("offers.json"))? != offers
        || result["selection"] != selection_seal(&dir, &selected)?
        || read::<Value>(&dir.join("selection.json"))? != result["selection"]
        || result["selection_sha256"] != digest(&dir.join("selection.json"))?
        || result["eligible_offers"] != offers.len()
        || result["new_construction_work"] != 0
        || result["new_discovery_work"] != 0
        || result["unseen_identification"] != false
    {
        return Err("actual imported state, ordinary-CE selection or scope changed".into());
    }
    let rows = result["evaluation"]
        .as_array()
        .ok_or("missing original evaluation outcomes")?;
    let mut count = 0;
    if record["status"] == "complete" {
        for saved in record["result"]["stages"]
            .as_array()
            .ok_or("missing stages")?
        {
            let s = imported::load(&pin, task, &record, saved, x, false)?;
            if let Some(r) = s.attempt.head {
                let matching: Vec<_> = rows
                    .iter()
                    .filter(|row| row["prime"] == s.prime)
                    .cloned()
                    .collect();
                outcomes::check(task, s.prime, &s.sha, &x.record, &r, s.eligible, &matching)?;
                count += matching.len();
            }
        }
    }
    if count != rows.len()
        || (deep && json!(evaluate(&pin, task, &record, x)?) != result["evaluation"])
    {
        return Err("original metrics omitted, altered or extra".into());
    }
    Ok((result, offers))
}
pub(crate) fn seal(root: &Path, task: &str, inv: &Value) -> Result<Value> {
    let dir = root.join(name(task)?);
    Ok(
        json!({"task":task,"replayed":true,"source_fingerprint":fingerprint(),"parent_summary_sha256":inv["request"]["parent"]["summary_sha256"],"staging_summary_sha256":inv["request"]["staging"]["summary_sha256"],"result_sha256":digest(&dir.join("result.json"))?,"selection_sha256":digest(&dir.join("selection.json"))?,"actual_complex_states_replayed":true,"all_original_metrics_replayed":true,"new_training":false}),
    )
}
pub fn replay_case(root: &Path, task: &str) -> Result<Value> {
    let (inv, parent, stage) = inventory(root, false)?;
    replay_with(
        root,
        task,
        &inv,
        &parent,
        &stage,
        inputs::build(&suite()?, task, &planned()?)?,
    )
}
pub(crate) fn replay_with(
    root: &Path,
    task: &str,
    inv: &Value,
    parent: &Value,
    stage: &Value,
    x: inputs::Inputs,
) -> Result<Value> {
    retained(root, task, inv, parent, stage, &x, true)?;
    seal(root, task, inv)
}
