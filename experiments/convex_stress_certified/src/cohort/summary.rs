//! Retain every refusal and endpoint, with content-bound replay evidence.
use super::*;

pub(super) fn validate_record(root: &Path, c: &Case, inv: &Value) -> Result<Value> {
    let out = root.join(c.id());
    let raw = bytes(&out.join("result.json"), 8 * 1024 * 1024)?;
    let r: Value = serde_json::from_slice(&raw)?;
    let seal = read(&root.join(format!("{}.replay.json", c.id())))?;
    if r["case"] != serde_json::to_value(c)?
        || r["id"] != c.id()
        || r["source_fingerprint"] != fingerprint()
        || r["protocol"] != protocol()?
        || r["producer_commit"] != inv["producer_commit"]
        || seal["replayed"] != true
        || seal["discovery_repeated"] != true
        || seal["source_fingerprint"] != fingerprint()
        || seal["id"] != c.id()
        || seal["result_sha256"] != hash(&raw)
        || seal["status"] != r["status"]
        || seal["discovery_sha256"] != r["discovery_sha256"]
        || r["discovery_sha256"] != hash(&bytes(&out.join("discovery.json"), 64 * 1024 * 1024)?)
    {
        return Err("missing, stale or unverified certified replay seal".into());
    }
    match r["status"].as_str() {
        Some("uncertified") => {
            if r["covered_cells"] != 0
                || ["model.json", "state.json", "trace.jsonl"]
                    .iter()
                    .any(|s| out.join(s).exists())
            {
                return Err("refused cell contains an unverified candidate".into());
            }
        }
        Some("complete") => {
            let shape: Shape = serde_json::from_value(r["shape"].clone())?;
            preflight(shape, RUN_MEMORY_BYTES)?;
            if seal["full_update_replay"] != true
                || seal["audit"]["raw_latent_coordinate_audit"] != true
            {
                return Err("missing numerical replay or raw audit".into());
            }
            for (key, file, limit) in [
                (
                    "model_sha256",
                    "model.json",
                    table_limit(shape, shape.orbits + shape.vocab),
                ),
                (
                    "state_sha256",
                    "state.json",
                    table_limit(shape, 2 * shape.orbits),
                ),
                ("trace_sha256", "trace.jsonl", 64 * 1024 * 1024),
            ] {
                if seal[key] != r[key] || r[key] != hash(&bytes(&out.join(file), limit)?) {
                    return Err("changed certified endpoint artifact".into());
                }
            }
        }
        _ => return Err("incomplete certified cell".into()),
    }
    Ok(r)
}
fn candidate(root: &Path, c: &Case, r: &Value) -> Result<Value> {
    let selection = r["selection"].as_str().ok_or("missing primary split")?;
    let primary = number(&r["metrics"][selection], "sequence_accuracy")?;
    let heldout: Vec<_> = r["metrics"]
        .as_object()
        .ok_or("missing metrics")?
        .iter()
        .filter(|(key, _)| key.as_str() != "train")
        .map(|(_, m)| number(m, "sequence_accuracy"))
        .collect::<Result<_>>()?;
    if heldout.is_empty() {
        return Err("missing held-out metrics".into());
    }
    Ok(
        json!({"kind":"certified_paired","id":c.id(),"task":c.task,"training_mode":c.arm,
        "model_path":root.join(c.id()).join("model.json"),"model_sha256":r["model_sha256"],
        "vocab":r["shape"]["vocab"],"train_ce":number(&r["metrics"]["train"],"loss")?,
        "ideal_ce_excess":r["ideal_ce_excess"],"metrics":r["metrics"],"selection":selection,
        "primary_pass":primary>=0.99,"all_splits_pass":heldout.iter().all(|&x|x>=0.99),
        "covered_cells":r["covered_cells"],"full_training_fit":r["full_training_fit"],
        "stop_reason":r["stop_reason"],"updates":r["updates"],"costs":r["costs"],
        "discovery_costs":r["discovery_costs"],"iterate_weight":r["iterate_weight"],
        "snapshot_norm_evaluations":r["snapshot_norm_evaluations"],"diagnostic_costs":r["diagnostic_costs"],
        "fit_defect":r["fit_defect"],"full_operator_residual":r["full_operator_residual"],
        "total_seconds_including_data":r["total_seconds_including_data"]}),
    )
}
pub fn summarize(root: &Path) -> Result<Value> {
    let inv = inventory(root)?;
    let (mut records, mut candidates) = (vec![], vec![]);
    for c in cases()? {
        let r = validate_record(root, &c, &inv)?;
        if r["status"] == "complete" {
            candidates.push(candidate(root, &c, &r)?);
        }
        records.push(r);
    }
    Ok(
        json!({"source_fingerprint":fingerprint(),"producer_commit":inv["producer_commit"],
        "protocol":protocol()?,"cells":records.len(),"uncertified_cells":records.len()-candidates.len(),
        "primary_pass_cells":candidates.iter().filter(|r|r["primary_pass"]==true).count(),
        "all_split_pass_cells":candidates.iter().filter(|r|r["all_splits_pass"]==true).count(),
        "records":records,"candidates":candidates,"integrated_with_older_candidates":false,
        "full35_result":false,"fresh_transformer_comparison":false}),
    )
}
