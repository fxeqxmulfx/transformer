use super::*;
fn metadata(inv: &Value, x: &Inputs) -> Value {
    json!({"source_fingerprint":fingerprint(),"binary_sha256":inv["binary_sha256"],"producer_commit":inv["producer_commit"],
        "protocol":protocol().expect("embedded protocol"),"input":x.record,"parent_sha256":inv["parent_sha256"],
        "training_gradient_evaluations":0,"standalone_forward_certification":true,"new_model_construction":false})
}
fn selection(dir: &Path, references: &Value) -> Result<Value> {
    Ok(
        json!({"references":references,"input_sha256":hash(&bytes(&dir.join("input.json"))?),
        "parents_sha256":hash(&bytes(&dir.join("parents.json"))?),"proposal_sha256":hash(&bytes(&dir.join("proposal.json"))?),
        "report_sha256":hash(&bytes(&dir.join("report.json"))?)}),
    )
}
pub fn run(root: &Path, name: &str) -> Result<()> {
    let (inv, parent) = inventory(root)?;
    let parent_root = Path::new(
        inv["modular_parent_root"]
            .as_str()
            .ok_or("missing modular parent root")?,
    );
    let raw = bytes(&parent_root.join(name).join("report.json"))?;
    execute(root, name, &inv, &parent, inputs(name)?, &raw)
}
pub(super) fn execute(
    root: &Path,
    name: &str,
    inv: &Value,
    parent: &Value,
    x: Inputs,
    raw: &[u8],
) -> Result<()> {
    check_input(inv, name, &x)?;
    let parents = parent::choices(parent, name, &x.record)?;
    let proposal = parent::proposal(raw, parent, name, &x)?;
    let dir = root.join(name);
    fs::create_dir(&dir)?;
    let meta = metadata(inv, &x);
    write(&dir.join("metadata.json"), &meta)?;
    write(&dir.join("input.json"), &x.input)?;
    write(&dir.join("parents.json"), &parents)?;
    write_raw(&dir.join("proposal.json"), raw)?;
    let started = std::time::Instant::now();
    let report = adapter::run(&proposal, policy()?)?;
    adapter::check(&proposal, &policy()?, &report)?;
    let seconds = started.elapsed().as_secs_f64();
    write(&dir.join("report.json"), &report)?;
    let references = adapter::select(&proposal, &report, &parents)?;
    let seal = selection(&dir, &references)?;
    write(&dir.join("selection.json"), &seal)?;
    // Selection is sealed before reading any original held-out target metric.
    let evaluated = evaluate::evaluate(&proposal, &report, &x, &parents, &references)?;
    write(
        &dir.join("result.json"),
        &json!({"metadata":meta,"selection":seal,
        "selection_sha256":hash(&bytes(&dir.join("selection.json"))?),"construction_seconds":seconds,
        "certificate_count":report.certificate_count,"buffer_preflight_bytes":report.buffer_preflight_bytes,
        "parent_views":report.parent_views,"ce_gradients":report.ce_gradients,"evaluation":evaluated}),
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
    let proposal = parent::proposal(&bytes(&dir.join("proposal.json"))?, parent, name, &x)?;
    let result: Value = read(&dir.join("result.json"))?;
    let report: adapter::Report = read(&dir.join("report.json"))?;
    if read::<search::Input>(&dir.join("input.json"))? != x.input
        || read::<Value>(&dir.join("metadata.json"))? != metadata(inv, &x)
        || result["metadata"] != metadata(inv, &x)
        || read::<Vec<Value>>(&dir.join("parents.json"))? != parents
        || report.policy != policy()?
        || result["certificate_count"] != report.certificate_count
        || result["buffer_preflight_bytes"] != report.buffer_preflight_bytes
        || result["parent_views"].as_array() != Some(&report.parent_views)
        || result["ce_gradients"] != 0
    {
        return Err("identification original input/parent/metadata/result changed".into());
    }
    search::replay(&x.input, &proposal)?;
    adapter::replay(&proposal, &report)?;
    let references = adapter::select(&proposal, &report, &parents)?;
    let seal = selection(&dir, &references)?;
    if read::<Value>(&dir.join("selection.json"))? != seal
        || result["selection"] != seal
        || result["selection_sha256"] != hash(&bytes(&dir.join("selection.json"))?)
        || result["evaluation"]
            != evaluate::evaluate(&proposal, &report, &x, &parents, &references)?
    {
        return Err("identification sealed CE/structure or original split metrics changed".into());
    }
    Ok(
        json!({"task":name,"replayed":true,"source_fingerprint":fingerprint(),"parent_sha256":inv["parent_sha256"],
        "result_sha256":hash(&bytes(&dir.join("result.json"))?),"selection_sha256":result["selection_sha256"],
        "references":references,"full_modular_parent_replay":true,"categorical_certificate_replay":true,
        "raw_latent_ce_metrics_audit":true,"unconditional_ground_truth_identification":false}),
    )
}
