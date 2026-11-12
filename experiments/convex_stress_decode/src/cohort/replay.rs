//! Rebuild categorical proposals, then inspect the literal saved probability tables.
use super::*;
pub fn replay(root: &Path, id: &str) -> Result<Value> {
    let c = case(id)?;
    let inv = inventory(root)?;
    let x = inputs(&c)?;
    replay_with(root, c, inv, x)
}
pub(super) fn replay_with(root: &Path, c: Case, inv: Value, x: Inputs) -> Result<Value> {
    check_task(&inv, &c, &x)?;
    let out = root.join(c.id());
    let result_bytes = bytes(&out.join("result.json"), 8 * 1024 * 1024)?;
    let r: Value = serde_json::from_slice(&result_bytes)?;
    let expected_metadata = metadata(&c, &inv, &x);
    if read(&out.join("metadata.json"))? != expected_metadata
        || expected_metadata
            .as_object()
            .unwrap()
            .iter()
            .any(|(key, value)| r[key] != *value)
    {
        return Err("changed decoded provenance, data or construction accounting".into());
    }
    let mut proposal = construct(&x)?;
    let expected = proposal.model.take();
    let proposal_bytes = bytes(&out.join("proposal.json"), 8 * 1024 * 1024)?;
    let saved_proposal: Value = serde_json::from_slice(&proposal_bytes)?;
    if saved_proposal != serde_json::to_value(&proposal)?
        || r["proposal_sha256"] != hash(&proposal_bytes)
        || r["proposal_cost"] != serde_json::to_value(&proposal.cost)?
        || r["proposal_exact_fit"] != proposal.exact_fit
        || r["proposal_best_errors"] != serde_json::to_value(proposal.best_errors)?
        || r["stop_reason"] != proposal.reason
    {
        return Err("training-only decoder trace did not reproduce".into());
    }
    let seal = |audit: Value| {
        json!({"replayed":true,"id":c.id(),"source_fingerprint":fingerprint(),
        "result_sha256":hash(&result_bytes),"proposal_sha256":r["proposal_sha256"],
        "model_sha256":r["model_sha256"],"status":r["status"],"audit":audit})
    };
    let Some(expected) = expected else {
        if r["status"] != "proposal_ineligible"
            || out.join("model.json").exists()
            || !r["model_sha256"].is_null()
            || !r["metrics"].is_null()
        {
            return Err("absent decoder proposal has invalid saved outcome".into());
        }
        return Ok(seal(
            json!({"constructor_reproduced":true,"rejected_proposal_reproduced":true}),
        ));
    };
    if r["status"] != "complete" {
        return Err("missing decoded endpoint".into());
    }
    let model_bytes = bytes(
        &out.join("model.json"),
        32 * (x.task.vocab as u64).pow(2) + 1024 * 1024,
    )?;
    if r["model_sha256"] != hash(&model_bytes) {
        return Err("changed decoded model digest".into());
    }
    let saved: Model = serde_json::from_slice(&model_bytes)?;
    if saved != expected {
        return Err("saved decoded tables differ from reconstructed endpoint".into());
    }
    drop(expected);
    let mut audit = saved.audit(&x.observations)?;
    let metrics = metrics_all(&saved, &x.task)?;
    if audit != r["audit"]
        || metrics != r["metrics"]
        || r["ideal_ce_excess"] != number(&metrics["train"], "loss")? - ideal_loss(saved.vocab)
        || r["audit_examples_presented"] != x.observations.len()
    {
        return Err("saved-table spectra, full-vocabulary CE or original metrics differ".into());
    }
    audit["raw_latent_coordinate_audit"] = true.into();
    audit["constructor_reproduced"] = true.into();
    Ok(seal(audit))
}
