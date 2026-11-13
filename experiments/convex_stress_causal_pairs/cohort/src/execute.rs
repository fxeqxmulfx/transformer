use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Attempt {
    pub input_refusal: Option<String>,
    pub report: Option<binding::Report>,
    pub error: Option<String>,
    pub construction_seconds: f64,
}
pub(crate) fn metadata(inv: &Value, input: &Value) -> Value {
    json!({"source_fingerprint":fingerprint(),"binary_sha256":inv["binary_sha256"],
        "producer_commit":inv["producer_commit"],"protocol":inv["protocol"],"input":input,
        "parent_summary_sha256":inv["request"]["parent"]["summary_sha256"]})
}
pub(crate) fn seal(dir: &Path, selection: &Value) -> Result<Value> {
    Ok(
        json!({"configuration_sha256":digest(&dir.join("configuration.json"))?,
        "report_sha256":digest(&dir.join("attempt.json"))?,"offers_sha256":digest(&dir.join("offers.json"))?,
        "selection":selection,"heldout_inputs":false}),
    )
}
pub fn run_case(root: &Path, task: &str) -> Result<()> {
    let (inv, parent) = inventory::inventory(root, false)?;
    let input = inputs::build(&suite()?, task)?;
    run_with(root, task, &inv, &parent, input)
}
pub(crate) fn run_with(
    root: &Path,
    task: &str,
    inv: &Value,
    parent: &Value,
    x: inputs::Inputs,
) -> Result<()> {
    inputs::check(inv, task, &x)?;
    let dir = root.join(name(task)?);
    fs::create_dir(&dir)?;
    let meta = metadata(inv, &x.record);
    write(&dir.join("metadata.json"), &meta)?;
    write(&dir.join("configuration.json"), &x.configuration)?;
    let started = std::time::Instant::now();
    let mut attempt = Attempt {
        input_refusal: x.refusal.clone(),
        report: None,
        error: None,
        construction_seconds: 0.,
    };
    if let Some(config) = &x.configuration {
        match binding::run(config.clone()) {
            Ok(report) => attempt.report = Some(report),
            Err(error) => attempt.error = Some(error.to_string()),
        }
    }
    attempt.construction_seconds = started.elapsed().as_secs_f64();
    write(&dir.join("attempt.json"), &attempt)?;
    let offers = outcomes::offers(task, &attempt, &digest(&dir.join("attempt.json"))?)?;
    write(&dir.join("offers.json"), &offers)?;
    let selected = comparison::select(
        parent,
        &offers,
        &serde_json::from_value::<Vec<String>>(inv["tasks"].clone())?,
    )?;
    let seal = seal(&dir, &selected)?;
    write(&dir.join("selection.json"), &seal)?;
    // No held-out score is computed until actual references are immutable.
    let evaluation = outcomes::evaluate(task, &x, &attempt, &digest(&dir.join("attempt.json"))?)?;
    write(
        &dir.join("result.json"),
        &json!({"metadata":meta,"selection":seal,
        "selection_sha256":digest(&dir.join("selection.json"))?,"evaluation":evaluation,
        "construction_seconds":attempt.construction_seconds,"eligible_offers":offers.len(),
        "total_seconds":started.elapsed().as_secs_f64(),"unseen_identification":false}),
    )
}
fn check_attempt(x: &inputs::Inputs, attempt: &Attempt) -> Result<()> {
    if attempt.input_refusal != x.refusal
        || !attempt.construction_seconds.is_finite()
        || attempt.construction_seconds < 0.
    {
        return Err("input refusal or construction timing changed".into());
    }
    match (&x.configuration, &attempt.report, &attempt.error) {
        (None, None, None) => Ok(()),
        (Some(config), Some(report), None) => binding::replay(config, report),
        (Some(config), None, Some(error)) => match binding::run(config.clone()) {
            Err(actual) if actual.to_string() == *error => Ok(()),
            _ => Err("recorded constructor error does not replay".into()),
        },
        _ => Err("constructor outcome is missing or contradictory".into()),
    }
}
pub fn replay_case(root: &Path, task: &str) -> Result<Value> {
    let (inv, parent) = inventory::inventory(root, false)?;
    replay_with(root, task, &inv, &parent, inputs::build(&suite()?, task)?)
}
pub(crate) fn replay_with(
    root: &Path,
    task: &str,
    inv: &Value,
    parent: &Value,
    x: inputs::Inputs,
) -> Result<Value> {
    inputs::check(inv, task, &x)?;
    let dir = root.join(name(task)?);
    let attempt: Attempt = read(&dir.join("attempt.json"))?;
    check_attempt(&x, &attempt)?;
    let result = retained(root, task, inv, parent)?;
    if read::<Option<binding::Request>>(&dir.join("configuration.json"))? != x.configuration
        || result["evaluation"]
            != outcomes::evaluate(task, &x, &attempt, &digest(&dir.join("attempt.json"))?)?
    {
        return Err("actual original data/model/metric replay differs".into());
    }
    Ok(
        json!({"task":task,"replayed":true,"source_fingerprint":fingerprint(),
        "parent_summary_sha256":inv["request"]["parent"]["summary_sha256"],
        "result_sha256":digest(&dir.join("result.json"))?,"report_sha256":digest(&dir.join("attempt.json"))?,
        "selection_sha256":digest(&dir.join("selection.json"))?,
        "actual_complex_states_replayed":true,"all_ce_steps_replayed":true,
        "all_original_metrics_and_masks_replayed":true,"unseen_identification":false}),
    )
}
pub(crate) fn retained(root: &Path, task: &str, inv: &Value, parent: &Value) -> Result<Value> {
    let dir = root.join(name(task)?);
    let attempt: Attempt = read(&dir.join("attempt.json"))?;
    let config: Option<binding::Request> = read(&dir.join("configuration.json"))?;
    let result: Value = read(&dir.join("result.json"))?;
    let input = inv["inputs"]
        .as_array()
        .ok_or("missing inputs")?
        .iter()
        .find(|r| r["task"] == task)
        .ok_or("unknown case")?;
    let offers = outcomes::offers(task, &attempt, &digest(&dir.join("attempt.json"))?)?;
    let selected = comparison::select(
        parent,
        &offers,
        &serde_json::from_value::<Vec<String>>(inv["tasks"].clone())?,
    )?;
    if read::<Value>(&dir.join("metadata.json"))? != metadata(inv, input)
        || result["metadata"] != metadata(inv, input)
        || input["configuration_sha256"]
            != json!(
                config
                    .as_ref()
                    .map(|c| serde_json::to_vec(c).map(|b| hash(&b)))
                    .transpose()?
            )
        || attempt.input_refusal
            != serde_json::from_value::<Option<String>>(input["input_refusal"].clone())?
        || read::<Vec<comparison::Offer>>(&dir.join("offers.json"))? != offers
        || read::<Value>(&dir.join("selection.json"))? != seal(&dir, &selected)?
        || result["selection"] != seal(&dir, &selected)?
        || result["selection_sha256"] != digest(&dir.join("selection.json"))?
        || result["construction_seconds"] != attempt.construction_seconds
        || result["total_seconds"]
            .as_f64()
            .is_none_or(|v| !v.is_finite() || v < attempt.construction_seconds)
        || result["eligible_offers"] != offers.len()
        || result["unseen_identification"] != false
    {
        return Err("retained original configuration, result or selection changed".into());
    }
    outcomes::check(
        task,
        input,
        &attempt,
        &digest(&dir.join("attempt.json"))?,
        &result["evaluation"],
    )?;
    Ok(result)
}
