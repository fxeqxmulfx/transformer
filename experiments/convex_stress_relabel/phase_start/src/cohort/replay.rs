//! Exact constructor reproduction followed by the independent raw CE/Fourier audit.
use super::*;
pub fn replay(root: &Path, id: &str) -> Result<Value> {
    let c = case(id)?;
    let inv = inventory(root)?;
    let x = inputs(&c)?;
    replay_with(root, c, inv, x)
}
pub(super) fn replay_with(root: &Path, c: Case, inv: Value, x: Inputs) -> Result<Value> {
    check_task(&inv, &c, &x)?;
    let id = c.id();
    let out = root.join(&id);
    let bytes = fs::read(out.join("result.json"))?;
    let r: Value = serde_json::from_slice(&bytes)?;
    if r["case"] != serde_json::to_value(&c)?
        || r["id"] != id
        || r["source_fingerprint"] != fingerprint()
        || r["protocol"] != protocol()?
        || r["producer_commit"] != inv["producer_commit"]
        || r["parents"] != inv["parents"]
        || r["vocab"] != x.task.vocab
        || r["data_fingerprints"] != x.row["data_fingerprints"]
        || r["selection"] != x.task.selection
        || r["discovery"] != serde_json::to_value(&x.discovery)?
        || r["coding"] != serde_json::to_value(&x.coding)?
        || r["explicit_bytes_upper_bound"] != x.explicit_bytes
        || r["training_examples"] != x.task.splits["train"].len()
        || r["coding_examples_presented"] != x.task.splits["train"].len()
        || r["updates"] != 0
        || r["training_gradient_evaluations"] != 0
        || r["optimizer_examples_presented"] != 0
    {
        return Err("changed phase provenance, data or construction accounting".into());
    }
    let (expected, details) = construct(&x)?;
    if r["proposal"] != details {
        return Err("training-only constructor did not reproduce".into());
    }
    let seal = |audit: Value| {
        json!({"replayed":true,"id":id,"source_fingerprint":fingerprint(),
        "result_sha256":hash(&bytes),"status":r["status"],"model_sha256":r["model_sha256"],"audit":audit})
    };
    let Some(expected) = expected else {
        if r["status"] != "proposal_ineligible"
            || out.join("model.json").exists()
            || !r["model_sha256"].is_null()
        {
            return Err("rejected phase proposal has invalid saved evidence".into());
        }
        return Ok(seal(json!({"rejected_proposal_reproduced":true})));
    };
    if r["status"] != "complete" || r["stop_reason"] != "exact_observed_fit_constructor" {
        return Err("missing constructed endpoint".into());
    }
    let path = out.join("model.json");
    let n = x.task.vocab;
    if fs::metadata(&path)?.len() > 32 * (n as u64).pow(3) + 1024 * 1024 {
        return Err("oversize phase model JSON".into());
    }
    let model_bytes = fs::read(path)?;
    if r["model_sha256"] != hash(&model_bytes) {
        return Err("changed phase model digest".into());
    }
    let saved: cold::Model = serde_json::from_slice(&model_bytes)?;
    drop(model_bytes);
    if saved.coding != expected.coding
        || saved.table.vocab != expected.table.vocab
        || saved.table.radius != expected.table.radius
        || saved.table.nuclear_budget != expected.table.nuclear_budget
        || saved.table.probabilities != expected.table.probabilities
        || r["radius"] != saved.table.radius
        || r["nuclear_budget"] != saved.table.nuclear_budget
    {
        return Err("saved phase coordinates differ from reconstructed endpoint".into());
    }
    drop(expected);
    let mut audit = cold::cohort::audit::validate(&saved, &x.task, &r)?;
    audit["constructor_reproduced"] = true.into();
    audit["training_derivatives"] = 0.into();
    Ok(seal(audit))
}
