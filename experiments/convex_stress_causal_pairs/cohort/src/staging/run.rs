use super::*;
fn metadata(inv: &Value, input: &Value) -> Value {
    json!({"phase":"training_only","source_fingerprint":inv["source_fingerprint"],
        "binary_sha256":inv["binary_sha256"],"producer_commit":inv["producer_commit"],
        "model_protocol":inv["model_protocol"],"staging_protocol":inv["staging_protocol"],
        "input":input,"parent_comparison":false,"heldout_evaluation":false})
}
pub fn run_case(root: &Path, task: &str) -> Result<()> {
    run_with(
        root,
        task,
        &inventory(root)?,
        inputs::build(&suite()?, task)?,
    )
}
pub(crate) fn run_with(root: &Path, task: &str, inv: &Value, x: inputs::Inputs) -> Result<()> {
    inputs::check(inv, task, &x)?;
    let dir = root.join(name(task)?);
    fs::create_dir(&dir)?;
    write(&dir.join("metadata.json"), &metadata(inv, &x.record))?;
    write(&dir.join("configuration.json"), &x.configuration)?;
    write(
        &dir.join("attempt.json"),
        &execute::construct(&x.configuration, &x.refusal),
    )?;
    write(&dir.join("result.json"), &record(&dir, inv, &x.record)?)
}
pub(crate) fn record(dir: &Path, inv: &Value, input: &Value) -> Result<Value> {
    let a: execute::Attempt = read(&dir.join("attempt.json"))?;
    let config: Option<binding::Request> = read(&dir.join("configuration.json"))?;
    if read::<Value>(&dir.join("metadata.json"))? != metadata(inv, input)
        || input["configuration_sha256"]
            != json!(
                config
                    .as_ref()
                    .map(|c| serde_json::to_vec(c).map(|b| hash(&b)))
                    .transpose()?
            )
        || a.input_refusal
            != serde_json::from_value::<Option<String>>(input["input_refusal"].clone())?
        || !a.construction_seconds.is_finite()
        || a.construction_seconds < 0.
    {
        return Err("staged configuration, metadata, refusal or timing differs".into());
    }
    let task = input["task"].as_str().ok_or("missing staged task")?;
    let offers = outcomes::offers(task, &a, &digest(&dir.join("attempt.json"))?)?;
    let gradients: usize = a.report.as_ref().map_or(0, |r| {
        r.endpoints
            .iter()
            .filter_map(|e| e.training.as_ref())
            .map(|t| t.work.gradient_batches)
            .sum()
    });
    Ok(
        json!({"metadata":metadata(inv, input),"configuration_sha256":digest(&dir.join("configuration.json"))?,
        "attempt_sha256":digest(&dir.join("attempt.json"))?,"construction_seconds":a.construction_seconds,
        "gradient_batches":gradients,"audited_training_offers":offers,
        "heldout_evaluation":false,"parent_comparison":false,"unseen_identification":false}),
    )
}
pub(crate) fn retained(root: &Path, task: &str, inv: &Value) -> Result<Value> {
    let dir = root.join(name(task)?);
    let input = inv["inputs"]
        .as_array()
        .ok_or("missing staging inputs")?
        .iter()
        .find(|x| x["task"] == task)
        .ok_or("unknown staged task")?;
    let result: Value = read(&dir.join("result.json"))?;
    if result != record(&dir, inv, input)? {
        return Err("staged result, costs or observed-only flags differ".into());
    }
    Ok(result)
}
pub(crate) fn seal(root: &Path, task: &str) -> Result<Value> {
    let dir = root.join(name(task)?);
    Ok(json!({"phase":"training_only","task":task,"replayed":true,
        "source_fingerprint":fingerprint(),"result_sha256":digest(&dir.join("result.json"))?,
        "attempt_sha256":digest(&dir.join("attempt.json"))?,
        "actual_complex_states_replayed":true,"all_ce_steps_replayed":true,
        "heldout_evaluation":false,"parent_comparison":false,"unseen_identification":false}))
}
pub fn replay_case(root: &Path, task: &str) -> Result<Value> {
    replay_with(
        root,
        task,
        &inventory(root)?,
        inputs::build(&suite()?, task)?,
    )
}
pub(crate) fn replay_with(
    root: &Path,
    task: &str,
    inv: &Value,
    x: inputs::Inputs,
) -> Result<Value> {
    inputs::check(inv, task, &x)?;
    retained(root, task, inv)?;
    let dir = root.join(name(task)?);
    if read::<Option<binding::Request>>(&dir.join("configuration.json"))? != x.configuration {
        return Err("staged original configuration changed".into());
    }
    execute::check_attempt(&x, &read(&dir.join("attempt.json"))?)?;
    seal(root, task)
}
