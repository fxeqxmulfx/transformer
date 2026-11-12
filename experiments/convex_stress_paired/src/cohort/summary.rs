//! Every planned cell with its sealed replay: metrics of all splits, failures and costs.
use super::*;

pub(super) fn validate_record(root: &Path, c: &Case, inv: &Value) -> Result<Value> {
    let out = root.join(c.id());
    let result_bytes = bytes(&out.join("result.json"), 8 * 1024 * 1024)?;
    let r: Value = serde_json::from_slice(&result_bytes)?;
    let seal = read(&root.join(format!("{}.replay.json", c.id())))?;
    let shape: Shape = serde_json::from_value(r["shape"].clone())?;
    shape.validate()?;
    if r["case"] != serde_json::to_value(c)?
        || r["id"] != c.id()
        || r["source_fingerprint"] != fingerprint()
        || r["protocol"] != protocol()?
        || r["producer_commit"] != inv["producer_commit"]
        || r["status"] != "complete"
        || seal["replayed"] != true
        || seal["id"] != c.id()
        || seal["source_fingerprint"] != fingerprint()
        || seal["result_sha256"] != hash(&result_bytes)
        || seal["status"] != r["status"]
        || seal["model_sha256"] != r["model_sha256"]
        || seal["diagnostic_state_sha256"] != r["diagnostic_state_sha256"]
        || seal["audit"]["raw_latent_coordinate_audit"] != true
        || r["model_sha256"]
            != hash(&bytes(
                &out.join("model.json"),
                table_limit(shape, shape.orbits + shape.vocab),
            )?)
        || r["diagnostic_state_sha256"]
            != hash(&bytes(
                &out.join("diagnostic_state.json"),
                table_limit(shape, shape.orbits),
            )?)
    {
        return Err("missing, stale or unverified paired replay seal".into());
    }
    Ok(r)
}
/// The unchanged original rule: at least 99% exact sequences on the primary
/// split, and on every held-out split.
fn candidate(root: &Path, c: &Case, r: &Value) -> Result<Value> {
    let selection = r["selection"].as_str().ok_or("missing primary split")?;
    let primary = number(&r["metrics"][selection], "sequence_accuracy")?;
    let heldout: Vec<_> = r["metrics"]
        .as_object()
        .ok_or("missing original metrics")?
        .iter()
        .filter(|(name, _)| name.as_str() != "train")
        .map(|(_, m)| number(m, "sequence_accuracy"))
        .collect::<Result<_>>()?;
    if heldout.is_empty() {
        return Err("missing original held-out metrics".into());
    }
    Ok(
        json!({"kind":"paired_fold","id":c.id(),"task":c.task,"training_mode":c.arm,
        "model_path":root.join(c.id()).join("model.json"),"model_sha256":r["model_sha256"],
        "vocab":r["shape"]["vocab"],"latent_states":r["latent_states"],
        "train_ce":number(&r["metrics"]["train"],"loss")?,"ideal_ce_excess":r["ideal_ce_excess"],
        "metrics":r["metrics"],"selection":selection,"primary_pass":primary>=0.99,
        "all_splits_pass":heldout.iter().all(|&a| a >= 0.99),"stop_reason":r["stop_reason"],
        "updates":r["updates"],"examples_presented":r["examples_presented"],
        "training_gradient_evaluations":r["training_gradient_evaluations"],"costs":r["costs"],
        "first_primary_99_update":r["first_primary_99_update"],
        "uniform_mixture_alpha":r["uniform_mixture_alpha"],"fit_defect":r["fit_defect"],
        "full_operator_residual":r["full_operator_residual"],
        "elapsed_seconds":r["elapsed_seconds"],
        "total_seconds_including_data":r["total_seconds_including_data"]}),
    )
}
pub fn summarize(root: &Path) -> Result<Value> {
    let inv = inventory(root)?;
    let mut candidates = vec![];
    for c in cases()? {
        let r = validate_record(root, &c, &inv)?;
        candidates.push(candidate(root, &c, &r)?);
    }
    let tally = |key: &str, value: Value| candidates.iter().filter(|c| c[key] == value).count();
    Ok(
        json!({"source_fingerprint":fingerprint(),"protocol":protocol()?,
        "producer_commit":inv["producer_commit"],"cells":candidates.len(),
        "reached_ideal_ce":tally("stop_reason", "ideal_ce_excess".into()),
        "stopped_at_update_cap":tally("stop_reason", "budget".into()),
        "primary_pass_cells":tally("primary_pass", true.into()),
        "all_split_pass_cells":tally("all_splits_pass", true.into()),
        "candidates":candidates,"integrated_with_older_candidates":false,
        "full35_result":false,"fresh_transformer_comparison":false}),
    )
}
