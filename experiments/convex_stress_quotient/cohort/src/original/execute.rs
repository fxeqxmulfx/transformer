use super::*;
pub(super) fn metadata(inv: &Value, x: &inputs::Inputs) -> Value {
    json!({"source_fingerprint":fingerprint(),"binary_sha256":inv["binary_sha256"],
        "producer_commit":inv["producer_commit"],"protocol":inv["protocol"],"input":x.record,
        "parent_summary_sha256":inv["request"]["parent"]["summary_sha256"]})
}
pub(super) fn seal(dir: &Path, selection: &Value) -> Result<Value> {
    Ok(
        json!({"configuration_sha256":digest(&dir.join("catalogue/configuration.json"))?,
        "ledger_sha256":digest(&dir.join("catalogue/ledger.json"))?,"offers_sha256":digest(&dir.join("offers.json"))?,
        "selection":selection,"heldout_inputs":false}),
    )
}
pub fn run_case(root: &Path, task: &str) -> Result<()> {
    let (inv, parent) = inventory::inventory(root, false)?;
    if !planned()?.iter().any(|t| t == task) {
        return Err("unplanned quotient case".into());
    }
    let started = std::time::Instant::now();
    run_with(
        root,
        task,
        &inv,
        &parent,
        inputs::build(&suite()?, task)?,
        started,
    )
}
pub(super) fn run_with(
    root: &Path,
    task: &str,
    inv: &Value,
    parent: &Value,
    x: inputs::Inputs,
    started: std::time::Instant,
) -> Result<()> {
    inputs::check(inv, task, &x)?;
    let dir = root.join(name(task)?);
    fs::create_dir(&dir)?;
    let meta = metadata(inv, &x);
    write(&dir.join("metadata.json"), &meta)?;
    crate::run(
        x.config.clone(),
        &dir.join("catalogue"),
        inv["producer_commit"].as_str().ok_or("missing producer")?,
    )?;
    let (config, ledger) = crate::execute::retained(&dir.join("catalogue"))?;
    if config != x.config {
        return Err("original stage configuration changed".into());
    }
    let offers = candidates::offers(task, &ledger)?;
    write(&dir.join("offers.json"), &offers)?;
    let all: Vec<String> = serde_json::from_value(inv["tasks"].clone())?;
    let selected = comparison::select(parent, &offers, &all)?;
    let seal = seal(&dir, &selected)?;
    write(&dir.join("selection.json"), &seal)?;
    // The immutable original-CE reference seal precedes every held-out score.
    let evaluation = candidates::evaluate(&dir.join("catalogue"), task, &x, &ledger)?;
    write(
        &dir.join("result.json"),
        &json!({"metadata":meta,"selection_sha256":digest(&dir.join("selection.json"))?,
        "selection":seal,"evaluation":evaluation,"stages":ledger.stages,
        "construction_seconds":ledger.seconds_per_stage.iter().sum::<f64>(),"total_seconds":started.elapsed().as_secs_f64(),
        "eligible_offers":offers.len(),"all55_stages_retained":true,"raw_label_identification":false}),
    )
}
pub fn replay_case(root: &Path, task: &str) -> Result<Value> {
    let (inv, parent) = inventory::inventory(root, false)?;
    if !planned()?.iter().any(|t| t == task) {
        return Err("unplanned quotient replay".into());
    }
    replay_with(root, task, &inv, &parent, inputs::build(&suite()?, task)?)
}
pub(super) fn replay_with(
    root: &Path,
    task: &str,
    inv: &Value,
    parent: &Value,
    x: inputs::Inputs,
) -> Result<Value> {
    inputs::check(inv, task, &x)?;
    let dir = root.join(name(task)?);
    let result: Value = read(&dir.join("result.json"))?;
    if read::<Value>(&dir.join("metadata.json"))? != metadata(inv, &x)
        || result["metadata"] != metadata(inv, &x)
    {
        return Err("original case metadata changed".into());
    }
    let catalogue = crate::replay(&dir.join("catalogue"))?;
    let (config, ledger) = crate::execute::retained(&dir.join("catalogue"))?;
    if config != x.config {
        return Err("replayed original configuration differs".into());
    }
    let offers = candidates::offers(task, &ledger)?;
    let selected = comparison::select(
        parent,
        &offers,
        &serde_json::from_value::<Vec<String>>(inv["tasks"].clone())?,
    )?;
    let expected = seal(&dir, &selected)?;
    if read::<Vec<comparison::Offer>>(&dir.join("offers.json"))? != offers
        || read::<Value>(&dir.join("selection.json"))? != expected
        || result["selection"] != expected
        || result["selection_sha256"] != digest(&dir.join("selection.json"))?
        || result["stages"] != serde_json::to_value(&ledger.stages)?
        || result["eligible_offers"] != offers.len()
        || result["construction_seconds"] != ledger.seconds_per_stage.iter().sum::<f64>()
        || result["total_seconds"]
            .as_f64()
            .is_none_or(|s| !s.is_finite() || s < 0.)
        || result["all55_stages_retained"] != true
        || result["raw_label_identification"] != false
        || result["evaluation"] != candidates::evaluate(&dir.join("catalogue"), task, &x, &ledger)?
    {
        return Err("actual saved model/CE/selection/metric replay differs".into());
    }
    Ok(
        json!({"task":task,"replayed":true,"source_fingerprint":fingerprint(),
        "parent_summary_sha256":inv["request"]["parent"]["summary_sha256"],
        "result_sha256":digest(&dir.join("result.json"))?,"selection_sha256":digest(&dir.join("selection.json"))?,
        "catalogue":catalogue,"all_actual_models_and_metrics_replayed":true,"all_ce_steps_replayed":true,
        "raw_label_identification":false,"target_readout_compatibility_proved":false}),
    )
}
