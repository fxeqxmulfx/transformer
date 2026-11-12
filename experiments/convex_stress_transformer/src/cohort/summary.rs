//! Keep every original task, unsuccessful caps, and original primary/all-split outcomes.
use super::*;
pub(super) fn verified_record(root: &Path, c: &Case, inv: &Value) -> Result<Value> {
    let out = root.join(c.id());
    let r = read(&out.join("result.json"))?;
    let seal = read(&root.join(format!("{}.replay.json", c.id())))?;
    let spec = inv["specifications"]
        .as_array()
        .ok_or("missing specifications")?
        .iter()
        .find(|s| s["id"] == c.id())
        .ok_or("missing original transformer specification")?;
    let metadata = json!({"specification":spec,"binding":hash(&serde_json::to_vec(spec)?),
        "producer_commit":inv["producer_commit"],"parent":inv["parent"]});
    if r["status"] != "complete"
        || r["case"] != serde_json::to_value(c)?
        || r["id"] != c.id()
        || r["metadata"] != metadata
        || read(&out.join("metadata.json"))? != metadata
        || seal["replayed"] != true
        || seal["id"] != c.id()
        || seal["source_fingerprint"] != fingerprint()
        || seal["result_sha256"] != file_hash(&out.join("result.json"))?
        || seal["updates_sha256"] != r["updates_sha256"]
        || r["updates_sha256"] != file_hash(&out.join("updates.jsonl"))?
        || seal["observations_sha256"] != r["observations_sha256"]
        || r["observations_sha256"] != file_hash(&out.join("observations.jsonl"))?
        || seal["endpoint_body_sha256"] != r["endpoint"]["checkpoint"]["body_sha256"]
        || seal["observations_replayed"] != r["observations"]
        || seal["final_update_bit_exact"] != true
        || seal["original_metrics_replayed"] != true
        || seal["complete_training_rerun"] != false
    {
        return Err("missing or stale original transformer replay seal".into());
    }
    replay::lines(&out.join("observations.jsonl"), |o| {
        let step = o["step"].as_u64().ok_or("missing checkpoint step")?;
        let mut snapshots = vec![(&o["checkpoint"], format!("at-{step:06}.bin"))];
        if step > 0 {
            snapshots.push((&o["before"], format!("before-{step:06}.bin")));
        }
        for (s, name) in snapshots {
            if s["file"] != name || s["file_sha256"] != file_hash(&out.join(name))? {
                return Err("observed transformer checkpoint changed after replay".into());
            }
        }
        Ok(())
    })?;
    Ok(r)
}
pub fn summarize(root: &Path) -> Result<Value> {
    let inv = inventory(root)?;
    let parent = Path::new(
        inv["parent"]["root"]
            .as_str()
            .ok_or("missing decoded parent root")?,
    );
    if parent_summary(parent)? != read(&root.join("parent-summary.json"))?
        || inv["parent"]["summary_sha256"] != file_hash(&parent.join("summary.json"))?
    {
        return Err("decoded queue parent changed since prepare".into());
    }
    let mut rows = vec![];
    for c in cases()? {
        let r = verified_record(root, &c, &inv)?;
        let primary = r["primary_split"]
            .as_str()
            .ok_or("missing original primary split")?;
        let all = r["metrics"]
            .as_object()
            .ok_or("missing original split metrics")?;
        let heldout = all
            .iter()
            .filter(|(name, _)| name.as_str() != "train")
            .map(|(_, m)| number(m, "sequence_accuracy"))
            .collect::<Result<Vec<_>>>()?;
        if heldout.is_empty() {
            return Err("missing held-out transformer metrics".into());
        }
        rows.push(json!({"case":c,"checkpoint":root.join(c.id()).join(r["endpoint"]["checkpoint"]["file"].as_str().ok_or("missing endpoint path")?),
            "checkpoint_body_sha256":r["endpoint"]["checkpoint"]["body_sha256"],"stop_reason":r["stop_reason"],
            "progress":r["progress"],"primary_split":primary,"metrics":r["metrics"],"observations":r["observations"],
            "primary_pass":number(&r["metrics"][primary],"sequence_accuracy")?>=0.99,"all_splits_pass":heldout.iter().all(|&v|v>=0.99),
            "total_seconds_including_data":r["total_seconds_including_data"],"historical_seed1":r["historical_seed1"]}));
    }
    if rows.len() != 35 {
        return Err("original transformer comparison scope incomplete".into());
    }
    Ok(
        json!({"source_fingerprint":fingerprint(),"protocol":protocol()?,"producer_commit":inv["producer_commit"],"parent":inv["parent"],
        "cells":rows.len(),"primary_pass":rows.iter().filter(|r|r["primary_pass"]==true).count(),
        "all_split_pass":rows.iter().filter(|r|r["all_splits_pass"]==true).count(),
        "original_cap_failures":rows.iter().filter(|r|r["stop_reason"]=="original_update_cap").count(),"results":rows,
        "all35_conventional_transformer_evaluated":true,"convex_full35_success_claim":false}),
    )
}
