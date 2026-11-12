//! Training-only comparison with every previously selected training mode.
use super::*;
use std::collections::BTreeSet;
fn candidate(root: &Path, c: &Case, r: &Value) -> Result<Value> {
    let selection = r["selection"].as_str().ok_or("missing primary split")?;
    let primary = number(&r["metrics"][selection], "sequence_accuracy")?;
    let heldout: Vec<_> = r["metrics"]
        .as_object()
        .ok_or("missing original metrics")?
        .iter()
        .filter(|(name, _)| name.as_str() != "train")
        .collect();
    if heldout.is_empty() {
        return Err("missing original held-out metrics".into());
    }
    let all = heldout
        .iter()
        .map(|(_, m)| number(m, "sequence_accuracy"))
        .collect::<Result<Vec<_>>>()?
        .iter()
        .all(|&a| a >= 0.99);
    Ok(
        json!({"kind":"decode_construct","id":c.id(),"model_path":root.join(c.id()).join("model.json"),
        "model_sha256":r["model_sha256"],"proposal_sha256":r["proposal_sha256"],"vocab":r["vocab"],
        "train_ce":number(&r["metrics"]["train"],"loss")?,"ideal_ce_excess":r["ideal_ce_excess"],
        "metrics":r["metrics"],"selection":selection,"primary_pass":primary>=0.99,"all_splits_pass":all,
        "updates":r["updates"],"training_gradient_evaluations":r["training_gradient_evaluations"],
        "proposal_cost":r["proposal_cost"],"proposal_exact_fit":r["proposal_exact_fit"],
        "data_validation_examples":r["data_validation_examples"],"audit_examples_presented":r["audit_examples_presented"],
        "optimizer_examples_presented":r["optimizer_examples_presented"],
        "standalone_seconds_including_data_and_proposal":r["total_seconds_including_data_and_proposal"]}),
    )
}
pub(super) fn validate_record(root: &Path, c: &Case, inv: &Value) -> Result<Value> {
    let out = root.join(c.id());
    let result_bytes = bytes(&out.join("result.json"), 8 * 1024 * 1024)?;
    let r: Value = serde_json::from_slice(&result_bytes)?;
    let seal = read(&root.join(format!("{}.replay.json", c.id())))?;
    if r["case"] != serde_json::to_value(c)?
        || r["id"] != c.id()
        || r["source_fingerprint"] != fingerprint()
        || r["protocol"] != protocol()?
        || r["producer_commit"] != inv["producer_commit"]
        || r["parents"] != inv["parents"]
        || seal["replayed"] != true
        || seal["id"] != c.id()
        || seal["source_fingerprint"] != fingerprint()
        || seal["result_sha256"] != hash(&result_bytes)
        || seal["status"] != r["status"]
        || seal["model_sha256"] != r["model_sha256"]
        || seal["proposal_sha256"] != r["proposal_sha256"]
        || seal["audit"]["constructor_reproduced"] != true
        || r["proposal_sha256"] != hash(&bytes(&out.join("proposal.json"), 8 * 1024 * 1024)?)
    {
        return Err("missing or stale decoded replay seal".into());
    }
    if r["status"] == "complete" {
        let n = r["vocab"].as_u64().ok_or("missing decoder dimension")?;
        if !(2..=193).contains(&n)
            || seal["audit"]["raw_latent_coordinate_audit"] != true
            || r["model_sha256"]
                != hash(&bytes(
                    &out.join("model.json"),
                    32 * n.pow(2) + 1024 * 1024,
                )?)
        {
            return Err("unverified decoded endpoint".into());
        }
    } else if r["status"] != "proposal_ineligible"
        || out.join("model.json").exists()
        || seal["audit"]["rejected_proposal_reproduced"] != true
    {
        return Err("unresolved planned decoder case".into());
    }
    Ok(r)
}
pub fn summarize(root: &Path) -> Result<Value> {
    let inv = inventory(root)?;
    let parent_root = Path::new(
        inv["parents"]["phase_root"]
            .as_str()
            .ok_or("missing phase root")?,
    );
    let parent = parent_summary(parent_root)?;
    if parent != read(&root.join("parent-summary.json"))?
        || inv["parents"]["phase_summary_sha256"]
            != hash(&bytes(&parent_root.join("summary.json"), 64 * 1024 * 1024)?)
    {
        return Err("phase/cold parent changed since preparation".into());
    }
    let records = cases()?
        .into_iter()
        .map(|c| {
            let r = validate_record(root, &c, &inv)?;
            Ok((c, r))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut choices = vec![];
    let mut seen = BTreeSet::new();
    for prior in parent["choices"]
        .as_array()
        .ok_or("missing parent choices")?
    {
        let task = prior["task"].as_str().ok_or("missing parent task")?;
        let mode = prior["cold_training_mode"]
            .as_str()
            .ok_or("missing parent mode")?;
        if !["full_batch", "minibatch", "forward_only"].contains(&mode)
            || !seen.insert((task, mode))
        {
            return Err("invalid or repeated parent choice".into());
        }
        let (c, r) = records
            .iter()
            .find(|(c, _)| c.task == task)
            .ok_or("unplanned parent task")?;
        let mut candidates = vec![];
        if r["status"] == "complete" {
            candidates.push(candidate(root, c, r)?);
        }
        candidates.push(prior["selected"].clone());
        let n = r["vocab"]
            .as_u64()
            .ok_or("missing original decoder vocabulary")? as usize;
        if candidates.iter().any(|x| x["vocab"] != n) {
            return Err("candidate vocabularies differ".into());
        }
        let losses = candidates
            .iter()
            .map(|x| number(x, "train_ce"))
            .collect::<Result<Vec<_>>>()?;
        let winner = convex_stress_relabel::choose_by_train(
            &losses,
            ideal_loss(n),
            number(&protocol()?, "ideal_ce_tolerance")?,
        )?;
        choices.push(
            json!({"task":task,"prior_training_mode":mode,"selected":candidates[winner],
            "baseline":prior,"candidates":candidates,"decoder_status":r["status"],
            "decoder_stop_reason":r["stop_reason"],"decoder_cost":r["proposal_cost"]}),
        );
    }
    if records.len() != 8 || choices.len() != 24 {
        return Err("decoder comparison scope incomplete".into());
    }
    Ok(
        json!({"source_fingerprint":fingerprint(),"protocol":protocol()?,"parents":inv["parents"],
        "cells":records.len(),"complete":records.iter().filter(|(_,r)|r["status"]=="complete").count(),
        "proposal_ineligible":records.iter().filter(|(_,r)|r["status"]=="proposal_ineligible").count(),
        "exact_fit_proposals":records.iter().filter(|(_,r)|r["proposal_exact_fit"]==true).count(),
        "primary_pass_choices":choices.iter().filter(|c|c["selected"]["primary_pass"]==true).count(),
        "all_split_pass_choices":choices.iter().filter(|c|c["selected"]["all_splits_pass"]==true).count(),
        "construction_endpoints_evaluated_once":true,"choices":choices,
        "full35_result":false,"fresh_transformer_comparison":false}),
    )
}
