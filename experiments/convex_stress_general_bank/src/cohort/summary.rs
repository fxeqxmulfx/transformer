//! Include every planned decoder slot and process outcome; require replay for scores.
use super::*;
pub(super) fn validate_record(root: &Path, c: &Case, inv: &Value) -> Result<Value> {
    let out = root.join(c.id());
    let raw = bytes(&out.join("result.json"), 64 * 1024 * 1024)?;
    let r: Value = serde_json::from_slice(&raw)?;
    let seal = read(&root.join(format!("{}.replay.json", c.id())))?;
    let task = inv["tasks"]
        .as_array()
        .ok_or("missing inventory tasks")?
        .iter()
        .find(|r| r["task"] == c.task)
        .ok_or("missing task")?;
    if r["case"] != serde_json::to_value(c)?
        || r["id"] != c.id()
        || r["source_fingerprint"] != fingerprint()
        || r["protocol"] != protocol()?
        || r["producer_commit"] != inv["producer_commit"]
        || r["task_record"] != *task
        || seal["replayed"] != true
        || seal["discovery_repeated"] != true
        || seal["id"] != c.id()
        || seal["source_fingerprint"] != fingerprint()
        || seal["result_sha256"] != hash(&raw)
        || seal["discovery_sha256"] != r["discovery_sha256"]
        || seal["status"] != r["status"]
        || seal["selected_decoder"] != r["selected_decoder"]
    {
        return Err("missing or stale replay seal".into());
    }
    let bound = task["explicit_bytes_upper_bound"]
        .as_u64()
        .filter(|&v| v <= RUN_MEMORY_BYTES)
        .ok_or("invalid file bound")?;
    if r["discovery_sha256"] != hash(&bytes(&out.join("discovery.json"), bound)?) {
        return Err("changed discovery artifact".into());
    }
    let models = r["candidates"]
        .as_array()
        .filter(|v| v.len() == 4)
        .ok_or("incomplete decoder slots")?;
    let seals = seal["candidates"]
        .as_array()
        .filter(|v| v.len() == models.len())
        .ok_or("incomplete candidate seals")?;
    for (k, (model, sealed)) in models.iter().zip(seals).enumerate() {
        let dir = out.join(format!("d{k}"));
        if model["decoder_index"] != k
            || sealed["decoder_index"] != k
            || sealed["status"] != model["status"]
        {
            return Err("decoder slot mismatch".into());
        }
        match model["status"].as_str() {
            Some("refused") => {
                if dir.exists() || sealed["refusal_repeated"] != true {
                    return Err("unverified refusal".into());
                }
            }
            Some("complete") => {
                if sealed["full_update_replay"] != true
                    || sealed["audit"]["raw_latent_coordinate_audit"] != true
                {
                    return Err("missing numerical replay/raw audit".into());
                }
                for (file, suffix) in [("model", "json"), ("state", "json"), ("trace", "jsonl")] {
                    let key = format!("{file}_sha256");
                    if model[&key] != sealed[&key]
                        || model[&key]
                            != hash(&bytes(&dir.join(format!("{file}.{suffix}")), bound)?)
                    {
                        return Err("changed endpoint artifact".into());
                    }
                }
            }
            _ => return Err("nonterminal candidate".into()),
        }
    }
    if r["selected_decoder"]
        != serde_json::to_value(choose(models, count(&task["shape"], "vocab")?)?)?
    {
        return Err("summary selection mismatch".into());
    }
    Ok(r)
}
fn candidate(root: &Path, c: &Case, r: &Value, k: usize) -> Result<Value> {
    let model = &r["candidates"][k];
    let selection = r["task_record"]["selection"]
        .as_str()
        .ok_or("missing primary split")?;
    let primary = number(&model["metrics"][selection], "sequence_accuracy")?;
    let heldout = model["metrics"]
        .as_object()
        .ok_or("missing metrics")?
        .iter()
        .filter(|(name, _)| name.as_str() != "train")
        .map(|(_, m)| number(m, "sequence_accuracy"))
        .collect::<Result<Vec<_>>>()?;
    if heldout.is_empty() {
        return Err("missing held-out splits".into());
    }
    Ok(
        json!({"kind":"general_bank","id":c.id(),"task":c.task,"training_mode":c.arm,"decoder_index":k,
        "model_path":root.join(c.id()).join(format!("d{k}/model.json")),"model_sha256":model["model_sha256"],
        "vocab":r["task_record"]["shape"]["vocab"],"train_ce":model["train_ce"],"ideal_ce_excess":model["ideal_ce_excess"],
        "selection":selection,"metrics":model["metrics"],"primary_pass":primary>=0.99,"all_splits_pass":heldout.iter().all(|&a|a>=0.99),
        "covered_cells":r["covered_cells"],"stop_reason":model["stop_reason"],"updates":model["updates"],"costs":model["costs"],
        "producer_decoder_checks":r["producer_decoder_checks"],"checker_decoder_checks":r["checker_decoder_checks"],
        "total_seconds_including_data":r["total_seconds_including_data"]}),
    )
}
pub fn summarize(root: &Path) -> Result<Value> {
    let root = root.canonicalize()?;
    let inv = inventory(&root)?;
    let (mut records, mut choices, mut all_candidates) = (vec![], vec![], vec![]);
    for c in cases()? {
        let code = |kind: &str| -> Result<i32> {
            Ok(
                fs::read_to_string(root.join(format!("{}.{kind}.exit", c.id())))?
                    .trim()
                    .parse()?,
            )
        };
        let run_exit = code("run")?;
        let replay_exit = if run_exit == 0 {
            Some(code("replay")?)
        } else {
            None
        };
        if run_exit != 0 || replay_exit != Some(0) {
            let failure = json!({"case":c,"id":c.id(),"status":"process_failure","run_exit":run_exit,"replay_exit":replay_exit,
                "run_log_sha256":hash(&bytes(&root.join(format!("{}.log",c.id())),64*1024*1024)?),
                "run_resources":root.join(format!("{}.resources",c.id())),"planned_decoder_slots":4});
            choices.push(json!({"task":c.task,"training_mode":c.arm,"status":"process_failure","selected":null}));
            records.push(failure);
            continue;
        }
        let r = validate_record(&root, &c, &inv)?;
        let mut candidates = vec![];
        for (k, m) in r["candidates"]
            .as_array()
            .ok_or("missing candidates")?
            .iter()
            .enumerate()
        {
            if m["status"] == "complete" {
                candidates.push(candidate(&root, &c, &r, k)?);
            }
        }
        let selected = r["selected_decoder"]
            .as_u64()
            .map(|k| candidate(&root, &c, &r, k as usize))
            .transpose()?;
        choices.push(json!({"task":c.task,"training_mode":c.arm,"status":r["status"],"selected":selected,"candidates":candidates}));
        all_candidates.extend(candidates);
        records.push(r);
    }
    let by_arm:Vec<_>=["full_batch","minibatch","forward_only"].into_iter().map(|mode|{
        let rows:Vec<_>=choices.iter().filter(|c|c["training_mode"]==mode).collect();
        json!({"training_mode":mode,"cases":rows.len(),"selected_models":rows.iter().filter(|r|!r["selected"].is_null()).count(),
            "primary_passes":rows.iter().filter(|r|r["selected"]["primary_pass"]==true).count(),
            "all_split_passes":rows.iter().filter(|r|r["selected"]["all_splits_pass"]==true).count()})
    }).collect();
    Ok(
        json!({"source_fingerprint":fingerprint(),"protocol":protocol()?,"producer_commit":inv["producer_commit"],
        "cases":records.len(),"planned_decoder_slots":records.len()*4,"complete_model_candidates":all_candidates.len(),
        "process_failures":records.iter().filter(|r|r["status"]=="process_failure").count(),"records":records,
        "candidates":all_candidates,"choices":choices,"by_arm":by_arm,"integrated_with_older_candidates":false,
        "full35_result":false,"fresh_transformer_comparison":false}),
    )
}
