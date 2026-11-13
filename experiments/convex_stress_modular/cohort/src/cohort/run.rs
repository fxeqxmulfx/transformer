use super::*;
fn metadata(inv: &Value, x: &Inputs) -> Value {
    json!({"source_fingerprint":fingerprint(),"binary_sha256":inv["binary_sha256"],
    "producer_commit":inv["producer_commit"],"protocol":protocol().expect("embedded protocol"),"input":x.record,
    "parent_sha256":inv["parent_sha256"],"training_gradient_evaluations":0,"standalone_forward_construction":true})
}
fn evaluate(
    report: &search::Report,
    x: &Inputs,
    parents: &[Value],
    references: &Value,
) -> Result<Value> {
    let mut candidates = vec![];
    for (index, c) in report.candidates.iter().enumerate() {
        let mut metrics = serde_json::Map::new();
        for (name, examples) in &x.task.splits {
            metrics.insert(name.clone(), metrics::evaluate(&c.model, examples)?);
        }
        let ce = c.audit["train_ce"].as_f64().ok_or("missing raw CE")?;
        if (metrics["train"]["loss"]
            .as_f64()
            .ok_or("missing train CE")?
            - ce)
            .abs()
            > 1e-10
        {
            return Err("ordinary original metric disagrees with raw CE".into());
        }
        let primary = metrics[&x.task.selection]["sequence_accuracy"]
            .as_f64()
            .ok_or("missing primary accuracy")?;
        let heldout: Vec<_> = metrics
            .iter()
            .filter(|(name, _)| name.as_str() != "train")
            .map(|(_, m)| {
                m["sequence_accuracy"]
                    .as_f64()
                    .ok_or("missing split accuracy")
            })
            .collect::<std::result::Result<_, _>>()?;
        if heldout.is_empty() {
            return Err("no held-out splits".into());
        }
        candidates.push(json!({"id":format!("modular-{index}"),"family":"modular","vocab":x.task.vocab,"train_ce":ce,
            "model_sha256":hash(&serde_json::to_vec(&c.model)?),"audit":c.audit,"metrics":metrics,
            "primary_pass":primary>=0.99,"all_splits_pass":heldout.iter().all(|&a|a>=0.99),
            "constructor_reference":true,"gradient_training_run":false}));
    }
    let mut choices = vec![];
    for old in parents {
        let r = references
            .as_array()
            .ok_or("missing references")?
            .iter()
            .find(|r| r["mode"] == old["training_mode"])
            .ok_or("missing sealed mode reference")?;
        let selected = if r["selected"] == "parent" {
            old["selected"].clone()
        } else if r["selected"].is_null() {
            Value::Null
        } else {
            candidates
                .iter()
                .find(|c| c["id"] == r["selected"])
                .ok_or("selected modular candidate absent")?
                .clone()
        };
        choices.push(json!({"task":x.record["task"],"training_mode":old["training_mode"],"tier":old["tier"],"selection":x.task.selection,
            "status":if selected.is_null(){"no_model"}else{"selected"},"primary_pass":selected["primary_pass"]==true,
            "all_splits_pass":selected["all_splits_pass"]==true,"selected":selected,"baseline":old,
            "modular_candidates":candidates,"sealed_reference":r}));
    }
    Ok(json!({"candidates":candidates,"choices":choices}))
}
pub fn run(root: &Path, name: &str) -> Result<()> {
    let (inv, parent) = inventory(root)?;
    execute(root, name, &inv, &parent, inputs(name)?)
}
pub(super) fn execute(
    root: &Path,
    name: &str,
    inv: &Value,
    parent: &Value,
    x: Inputs,
) -> Result<()> {
    check_input(inv, name, &x)?;
    let parents = parent::choices(parent, name, &x.record)?;
    let dir = root.join(name);
    fs::create_dir(&dir)?;
    let meta = metadata(inv, &x);
    write(&dir.join("metadata.json"), &meta)?;
    write(&dir.join("input.json"), &x.input)?;
    write(&dir.join("parents.json"), &parents)?;
    let started = std::time::Instant::now();
    let report = search::run(x.input.clone())?;
    let checked = search::check(&x.input, &report)?;
    let seconds = started.elapsed().as_secs_f64();
    write(&dir.join("report.json"), &report)?;
    let references = parent::select(&report, &parents)?;
    let selection = json!({"references":references,"input_sha256":hash(&bytes(&dir.join("input.json"))?),
        "parents_sha256":hash(&bytes(&dir.join("parents.json"))?),"report_sha256":hash(&bytes(&dir.join("report.json"))?)});
    write(&dir.join("selection.json"), &selection)?;
    // Original held-out metrics are first read after this observed-only seal.
    let evaluated = evaluate(&report, &x, &parents, &references)?;
    write(
        &dir.join("result.json"),
        &json!({"metadata":meta,"selection":selection,"selection_sha256":hash(&bytes(&dir.join("selection.json"))?),
        "construction_seconds":seconds,"certificate_audit":checked,"evaluation":evaluated}),
    )
}
pub fn replay(root: &Path, name: &str) -> Result<Value> {
    let (inv, parent) = inventory(root)?;
    replay_with(root, name, &inv, &parent, inputs(name)?)
}
pub(super) fn replay_with(
    root: &Path,
    name: &str,
    inv: &Value,
    parent: &Value,
    x: Inputs,
) -> Result<Value> {
    check_input(inv, name, &x)?;
    let dir = root.join(name);
    let parents = parent::choices(parent, name, &x.record)?;
    let result: Value = read(&dir.join("result.json"))?;
    let report: search::Report = read(&dir.join("report.json"))?;
    if read::<search::Input>(&dir.join("input.json"))? != x.input
        || read::<Value>(&dir.join("metadata.json"))? != metadata(inv, &x)
        || result["metadata"] != metadata(inv, &x)
        || read::<Vec<Value>>(&dir.join("parents.json"))? != parents
    {
        return Err("modular original input/parent/metadata changed".into());
    }
    if search::check(&x.input, &report)? != result["certificate_audit"] {
        return Err("saved modular certificate audit changed".into());
    }
    search::replay(&x.input, &report)?;
    let references = parent::select(&report, &parents)?;
    let selection = json!({"references":references,"input_sha256":hash(&bytes(&dir.join("input.json"))?),
        "parents_sha256":hash(&bytes(&dir.join("parents.json"))?),"report_sha256":hash(&bytes(&dir.join("report.json"))?)});
    if read::<Value>(&dir.join("selection.json"))? != selection
        || result["selection"] != selection
        || result["selection_sha256"] != hash(&bytes(&dir.join("selection.json"))?)
        || result["evaluation"] != evaluate(&report, &x, &parents, &references)?
    {
        return Err(
            "modular sealed observed CE, full state or original split metrics changed".into(),
        );
    }
    Ok(
        json!({"task":name,"replayed":true,"source_fingerprint":fingerprint(),"parent_sha256":inv["parent_sha256"],
        "result_sha256":hash(&bytes(&dir.join("result.json"))?),"selection_sha256":result["selection_sha256"],"references":references,
        "full_integer_strip_state_replay":true,"raw_latent_ce_metrics_audit":true}),
    )
}
