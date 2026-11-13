//! Seal the observed-only choice before metrics; replay complete states from input.
use super::*;
fn provenance(inv: &Value, x: &Inputs) -> Result<Value> {
    Ok(
        json!({"source_fingerprint":fingerprint(),"binary_sha256":inv["binary_sha256"],"protocol":protocol()?,
        "producer_commit":inv["producer_commit"],"input":x.record,"updates":0,
        "training_gradient_evaluations":0,"standalone_forward_construction":true}),
    )
}
fn evaluate(r: &Report, x: &Inputs) -> Result<Value> {
    let samples = search::validate(&x.input.categorical)?;
    let mut records = vec![];
    let (mut best, mut best_ce) = (None, f64::INFINITY);
    for (id, model) in candidates(r)? {
        let audit = model.audit(&samples)?;
        let ce = audit["train_ce"].as_f64().ok_or("missing raw CE")?;
        if ce + 1e-12 < best_ce {
            best = Some(id.clone());
            best_ce = ce;
        }
        let mut metrics = serde_json::Map::new();
        for (split, examples) in &x.task.splits {
            metrics.insert(split.clone(), metrics::evaluate(&model, examples)?);
        }
        if (metrics["train"]["loss"]
            .as_f64()
            .ok_or("missing training metric")?
            - ce)
            .abs()
            > 1e-10
        {
            return Err("ordinary CE differs between raw table and original metric".into());
        }
        let primary = metrics[&x.task.selection]["sequence_accuracy"]
            .as_f64()
            .ok_or("missing primary split")?;
        let heldout = metrics
            .iter()
            .filter(|(name, _)| name.as_str() != "train")
            .map(|(_, m)| {
                m["sequence_accuracy"]
                    .as_f64()
                    .ok_or("missing heldout metric")
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if heldout.is_empty() {
            return Err("no heldout splits".into());
        }
        records.push(
            json!({"id":id,"train_ce":ce,"audit":audit,"metrics":metrics,
            "model_sha256":hash(&serde_json::to_vec(&model)?),"primary_pass":primary>=0.99,
            "all_splits_pass":heldout.iter().all(|&a|a>=0.99)}),
        );
    }
    if best != selected(r) {
        return Err("raw CE order differs from saved construction choice".into());
    }
    Ok(
        json!({"status":if best.is_some() {"complete"} else {"proposal_ineligible"},
        "selected":best,"candidates":records}),
    )
}
pub fn run(root: &Path, name: &str) -> Result<()> {
    execute(root, name, &inventory(root)?, inputs(name)?)
}
pub(super) fn execute(root: &Path, name: &str, inv: &Value, x: Inputs) -> Result<()> {
    check_record(inv, name, &x)?;
    let dir = root.join(name);
    fs::create_dir(&dir)?;
    let meta = provenance(inv, &x)?;
    write(&dir.join("metadata.json"), &meta)?;
    write(&dir.join("input.json"), &x.input)?;
    let started = std::time::Instant::now();
    let (old, report) = construct(&x.input)?;
    let seconds = started.elapsed().as_secs_f64();
    if report.buffer_bound_bytes
        > x.record["buffer_preflight_bytes"]
            .as_u64()
            .ok_or("missing bound")?
    {
        return Err("constructed report exceeds declared preflight".into());
    }
    write(&dir.join("earlier-proposal.json"), &old)?;
    write(&dir.join("report.json"), &report)?;
    let choice = json!({"selected":selected(&report),"source_fingerprint":fingerprint(),
        "input_sha256":hash(&bytes(&dir.join("input.json"))?),
        "earlier_sha256":hash(&bytes(&dir.join("earlier-proposal.json"))?),
        "report_sha256":hash(&bytes(&dir.join("report.json"))?)});
    write(&dir.join("selection.json"), &choice)?;
    // No metrics are used by construct or its choice. This seal precedes all
    // evaluation of candidate outputs against the held-out targets.
    let evaluated = evaluate(&report, &x)?;
    write(
        &dir.join("result.json"),
        &json!({"metadata":meta,"choice":choice,
        "selection_sha256":hash(&bytes(&dir.join("selection.json"))?),
        "construction_seconds":seconds,"evaluation":evaluated}),
    )
}
pub fn replay(root: &Path, name: &str) -> Result<Value> {
    replay_with(root, name, &inventory(root)?, inputs(name)?)
}
pub(super) fn replay_with(root: &Path, name: &str, inv: &Value, x: Inputs) -> Result<Value> {
    check_record(inv, name, &x)?;
    let dir = root.join(name);
    let meta = provenance(inv, &x)?;
    let result_bytes = bytes(&dir.join("result.json"))?;
    let result: Value = serde_json::from_slice(&result_bytes)?;
    if read::<Value>(&dir.join("metadata.json"))? != meta
        || result["metadata"] != meta
        || read::<interval_search::Input>(&dir.join("input.json"))? != x.input
    {
        return Err("density original input/provenance mismatch".into());
    }
    let saved: Report = read(&dir.join("report.json"))?;
    let (old, rebuilt) = construct(&x.input)?;
    if read::<Value>(&dir.join("earlier-proposal.json"))? != old || saved != rebuilt {
        return Err("density full state/proposal replay mismatch".into());
    }
    let choice = json!({"selected":selected(&saved),"source_fingerprint":fingerprint(),
        "input_sha256":hash(&bytes(&dir.join("input.json"))?),
        "earlier_sha256":hash(&bytes(&dir.join("earlier-proposal.json"))?),
        "report_sha256":hash(&bytes(&dir.join("report.json"))?)});
    if read::<Value>(&dir.join("selection.json"))? != choice
        || result["choice"] != choice
        || result["selection_sha256"] != hash(&bytes(&dir.join("selection.json"))?)
        || result["evaluation"] != evaluate(&saved, &x)?
    {
        return Err("density selection, raw audit or original metrics changed".into());
    }
    Ok(
        json!({"replayed":true,"task":name,"source_fingerprint":fingerprint(),
        "result_sha256":hash(&result_bytes),"selection_sha256":result["selection_sha256"],
        "choice":choice,"full_proposal_state_replay":true,"raw_latent_ce_metrics_audit":true}),
    )
}
