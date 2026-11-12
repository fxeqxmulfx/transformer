//! Compare fixed-code constructed optima with every frozen cold training mode.
use super::*;
fn candidate(kind: &str, id: &str, path: &Path, r: &Value) -> Result<Value> {
    let selection = r["selection"]
        .as_str()
        .ok_or("missing original selection split")?;
    let primary = number(&r["metrics"][selection], "sequence_accuracy")?;
    let heldout: Vec<_> = r["metrics"]
        .as_object()
        .ok_or("missing metrics")?
        .iter()
        .filter(|(name, _)| name.as_str() != "train")
        .collect();
    if heldout.is_empty() {
        return Err("no original held-out splits".into());
    }
    let all = heldout
        .iter()
        .map(|(_, m)| number(m, "sequence_accuracy"))
        .collect::<Result<Vec<_>>>()?
        .iter()
        .all(|x| *x >= 0.99);
    Ok(
        json!({"kind":kind,"id":id,"model_path":path.join("model.json"),"model_sha256":r["model_sha256"],
        "train_ce":number(&r["metrics"]["train"],"loss")?,"ideal_ce_excess":r["ideal_ce_excess"],
        "vocab":r["vocab"],"metrics":r["metrics"],"selection":selection,"primary_pass":primary>=0.99,
        "all_splits_pass":all,"updates":r["updates"],"training_gradient_evaluations":r["training_gradient_evaluations"],
        "coding_examples_presented":if kind=="phase_construct" {&r["coding_examples_presented"]} else {&r["proposal_examples_presented"]},
        "phase_examples_presented":if kind=="phase_construct" {r["proposal"]["phase_examples_presented"].clone()} else {json!(0)},
        "fit_verification_examples":if kind=="phase_construct" {r["proposal"]["fit_verification_examples"].clone()} else {json!(0)},
        "optimizer_examples_presented":if kind=="phase_construct" {&r["optimizer_examples_presented"]} else {&r["examples_presented"]},
        "standalone_seconds_including_data_and_proposal":r["total_seconds_including_data_and_proposal"]}),
    )
}
pub fn summarize(root: &Path) -> Result<Value> {
    let inv = inventory(root)?;
    let p = protocol()?;
    let cold_root = Path::new(
        inv["parents"]["cold_root"]
            .as_str()
            .ok_or("missing cold root")?,
    );
    let graph_root = Path::new(
        inv["parents"]["graph_root"]
            .as_str()
            .ok_or("missing graph root")?,
    );
    let (parent, graph) = parent_summaries(cold_root, graph_root)?;
    if parent != read(&root.join("cold-summary.json"))?
        || graph != read(&root.join("graph-summary.json"))?
        || inv["parents"]["cold_summary_sha256"] != hash(&fs::read(cold_root.join("summary.json"))?)
        || inv["parents"]["graph_summary_sha256"]
            != hash(&fs::read(graph_root.join("summary.json"))?)
    {
        return Err("phase parents changed since prepare".into());
    }
    let mut records = vec![];
    for c in cases()? {
        let id = c.id();
        let out = root.join(&id);
        let bytes = fs::read(out.join("result.json"))?;
        let r: Value = serde_json::from_slice(&bytes)?;
        let seal = read(&root.join(format!("{id}.replay.json")))?;
        if r["case"] != serde_json::to_value(&c)?
            || r["source_fingerprint"] != fingerprint()
            || r["protocol"] != p
            || r["producer_commit"] != inv["producer_commit"]
            || r["parents"] != inv["parents"]
            || seal["replayed"] != true
            || seal["source_fingerprint"] != fingerprint()
            || seal["id"] != id
            || seal["result_sha256"] != hash(&bytes)
            || seal["status"] != r["status"]
            || seal["model_sha256"] != r["model_sha256"]
        {
            return Err("missing or stale phase replay seal".into());
        }
        if r["status"] == "complete" {
            let n = r["vocab"].as_u64().ok_or("missing model size")?;
            if !(2..=193).contains(&n)
                || fs::metadata(out.join("model.json"))?.len() > 32 * n.pow(3) + 1024 * 1024
                || seal["audit"]["raw_coordinate_audit"] != true
                || seal["audit"]["constructor_reproduced"] != true
                || r["model_sha256"] != hash(&fs::read(out.join("model.json"))?)
            {
                return Err("unverified phase endpoint".into());
            }
        } else if r["status"] != "proposal_ineligible"
            || seal["audit"]["rejected_proposal_reproduced"] != true
            || out.join("model.json").exists()
        {
            return Err("unresolved phase candidate".into());
        }
        records.push((c, r));
    }
    let mut choices = vec![];
    for prior in parent["choices"].as_array().ok_or("missing cold choices")? {
        let task = prior["task"].as_str().ok_or("missing cold task")?;
        let mut candidates = vec![];
        for (c, r) in records
            .iter()
            .filter(|(c, r)| c.task == task && r["status"] == "complete")
        {
            candidates.push(candidate(
                "phase_construct",
                &c.id(),
                &root.join(c.id()),
                r,
            )?);
        }
        for old in prior["candidates"]
            .as_array()
            .ok_or("missing cold candidates")?
        {
            let id = old["id"].as_str().ok_or("missing cold case")?;
            let record = read(&cold_root.join(id).join("result.json"))?;
            candidates.push(candidate("cold", id, &cold_root.join(id), &record)?);
        }
        let n = candidates
            .first()
            .and_then(|r| r["vocab"].as_u64())
            .ok_or("no usable candidate")? as usize;
        if candidates.iter().any(|r| r["vocab"] != n) {
            return Err("candidate vocabularies differ".into());
        }
        let losses = candidates
            .iter()
            .map(|c| number(c, "train_ce"))
            .collect::<Result<Vec<_>>>()?;
        let winner = cold::choose_by_train(
            &losses,
            cold::ideal_loss(n),
            number(&p, "ideal_ce_tolerance")?,
        )?;
        choices.push(json!({"task":task,"cold_training_mode":prior["arm"],"selected":candidates[winner],
            "baseline":prior,"candidates":candidates.iter().map(|c|json!({"kind":c["kind"],"id":c["id"],
            "train_ce":c["train_ce"],"model_sha256":c["model_sha256"],"updates":c["updates"],
            "standalone_seconds_including_data_and_proposal":c["standalone_seconds_including_data_and_proposal"]})).collect::<Vec<_>>()}));
    }
    if records.len() != 16 || choices.len() != 24 {
        return Err("phase comparison scope incomplete".into());
    }
    Ok(
        json!({"source_fingerprint":fingerprint(),"protocol":p,"parents":inv["parents"],"cells":records.len(),
        "complete":records.iter().filter(|(_,r)|r["status"]=="complete").count(),
        "proposal_ineligible":records.iter().filter(|(_,r)|r["status"]=="proposal_ineligible").count(),
        "primary_pass_choices":choices.iter().filter(|c|c["selected"]["primary_pass"]==true).count(),
        "all_split_pass_choices":choices.iter().filter(|c|c["selected"]["all_splits_pass"]==true).count(),
        "phase_candidates_evaluated_once":true,"choices":choices,"full35_result":false,"fresh_transformer_comparison":false}),
    )
}
