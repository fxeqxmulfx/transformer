//! Recompute discovery and every update, then independently audit raw coordinates.
use super::*;
pub fn replay(root: &Path, id: &str) -> Result<Value> {
    let c = case(id)?;
    replay_with(root, c.clone(), inventory(root)?, inputs(&c)?)
}
pub(super) fn replay_with(root: &Path, c: Case, inv: Value, x: Inputs) -> Result<Value> {
    check_task(&inv, &c, &x)?;
    let out = root.join(c.id());
    let raw = bytes(&out.join("result.json"), 64 * 1024 * 1024)?;
    let r: Value = serde_json::from_slice(&raw)?;
    let expected = metadata(&c, &inv, &x)?;
    if read(&out.join("metadata.json"))? != expected
        || expected
            .as_object()
            .ok_or("invalid metadata")?
            .iter()
            .any(|(k, v)| r[k] != *v)
    {
        return Err("changed case provenance".into());
    }
    let raw_discovery = bytes(&out.join("discovery.json"), x.explicit_bytes)?;
    let record: Record = serde_json::from_slice(&raw_discovery)?;
    if r["discovery_sha256"] != hash(&raw_discovery)
        || serde_json::to_value(record.limits)? != protocol()?["search"]
    {
        return Err("changed discovery configuration or digest".into());
    }
    drop(raw_discovery);
    let found = search::replay(&x.problem, &record)?;
    let coverage = record.selected.as_ref().map_or(0, |cert| {
        cert.answers.iter().filter(|a| a.len() == 1).count()
    });
    if r["producer_decoder_checks"] != record.producer_checks
        || r["checker_decoder_checks"] != record.checker_checks
        || r["search_stop"] != serde_json::to_value(record.stop)?
        || r["covered_cells"] != coverage
    {
        return Err("changed discovery costs, stop or coverage".into());
    }
    let candidates = r["candidates"]
        .as_array()
        .filter(|v| v.len() == x.problem.decoders.len())
        .ok_or("incomplete decoder slots")?;
    let mut seals = vec![];
    let mut recomputed = vec![];
    for (k, saved) in candidates.iter().enumerate() {
        let dir = out.join(format!("d{k}"));
        if let Some(reason) = candidate_refusal(found.verified.as_ref(), k) {
            let expected = json!({"decoder_index":k,"status":"refused","reason":reason});
            if *saved != expected || dir.exists() {
                return Err("changed refused decoder slot".into());
            }
            recomputed.push(expected);
            seals.push(json!({"decoder_index":k,"status":"refused","refusal_repeated":true}));
            continue;
        }
        let verified = found.verified.as_ref().ok_or("missing certificate")?;
        let model_raw = bytes(&dir.join("model.json"), x.explicit_bytes)?;
        let state_raw = bytes(&dir.join("state.json"), x.explicit_bytes)?;
        let trace_raw = bytes(&dir.join("trace.jsonl"), x.explicit_bytes)?;
        if saved["model_sha256"] != hash(&model_raw)
            || saved["state_sha256"] != hash(&state_raw)
            || saved["trace_sha256"] != hash(&trace_raw)
        {
            return Err("changed candidate artifact digest".into());
        }
        let model: Model = serde_json::from_slice(&model_raw)?;
        let state: State = serde_json::from_slice(&state_raw)?;
        let trace: Vec<Value> = std::str::from_utf8(&trace_raw)?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        drop((model_raw, state_raw, trace_raw));
        let mut cursor = 0;
        let fitted = run::fit(&c, &x, verified, k, |v| {
            audit::close(
                v,
                trace.get(cursor).ok_or("missing replay snapshot")?,
                1e-12,
            )?;
            cursor += 1;
            Ok(())
        })?;
        if fitted.model != model || fitted.state != state || cursor != trace.len() {
            return Err("saved state, endpoint or trace differs from replay".into());
        }
        for (key, value) in fitted.report.as_object().ok_or("invalid fitted report")? {
            audit::close(value, &saved[key], 1e-10)?;
        }
        model.certified(verified)?;
        recomputed.push(fitted.report);
        seals.push(json!({"decoder_index":k,"status":"complete","full_update_replay":true,
            "model_sha256":saved["model_sha256"],"state_sha256":saved["state_sha256"],"trace_sha256":saved["trace_sha256"]}));
    }
    let selected = choose(&recomputed, x.shape.vocab)?;
    let status = if selected.is_some() {
        "complete"
    } else if found.verified.as_ref().is_some_and(Verified::inconsistent) {
        "bank_inconsistent"
    } else {
        "no_candidate"
    };
    if r["selected_decoder"] != serde_json::to_value(selected)? || r["status"] != status {
        return Err("changed CE-only selection or case status".into());
    }
    // As in training, validate held-out metrics after repeating the CE choice.
    for (k, saved) in candidates
        .iter()
        .enumerate()
        .filter(|(_, r)| r["status"] == "complete")
    {
        let model: Model = serde_json::from_slice(&bytes(
            &out.join(format!("d{k}/model.json")),
            x.explicit_bytes,
        )?)?;
        if metrics_all(&model, &x.task)? != saved["metrics"] {
            return Err("changed original split metrics".into());
        }
        seals[k]["audit"] = audit::validate(
            &model,
            found.verified.as_ref().ok_or("missing certificate")?,
            &x.task,
            saved,
        )?;
    }
    Ok(
        json!({"replayed":true,"id":c.id(),"source_fingerprint":fingerprint(),"result_sha256":hash(&raw),
        "discovery_sha256":r["discovery_sha256"],"discovery_repeated":true,"status":status,
        "selected_decoder":selected,"candidates":seals}),
    )
}
