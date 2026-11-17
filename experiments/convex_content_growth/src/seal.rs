//! All-cell barrier and independent reconstruction of every frozen phase base.
use crate::{
    Result,
    experiment::{self, Cell},
    model::Model,
    source,
};
use bilinear::{experiment as original, guard};
use serde_json::{Value, json};
use std::{fs, path::Path};
pub fn all(workspace: &Path, storage: &Path, cohort: &Path) -> Result<()> {
    let fingerprint = experiment::fingerprint();
    let binary = original::hash(&fs::read(std::env::current_exe()?)?);
    let p = experiment::protocol()?;
    for cell in experiment::cells() {
        guard::admit(512 * 1024 * 1024, workspace, storage)?;
        let path = cell.path(cohort);
        let m = original::score::read(&path.join("metadata.json"))?;
        let r = original::score::read(&path.join("training_receipt.json"))?;
        let stages = p["stages"].as_u64().ok_or("stages")? as usize;
        if m["source_fingerprint"] != fingerprint
            || m["binary_sha256"] != binary
            || m["protocol"] != p
            || m["input"] != source::pin(&cell)?
            || m["cell"] != json!(cell)
            || r["source_fingerprint"] != fingerprint
            || r["cell"] != json!(cell)
            || r["status"] != "sealed"
            || r["heldout_scored"] != false
            || r["updates"] != stages * p["updates_per_stage"].as_u64().ok_or("updates")? as usize
            || r["stage_receipts"].as_array().map(Vec::len) != Some(stages)
        {
            return Err("incomplete or changed growth cohort seal".into());
        }
        for n in 1..=stages {
            let stage = experiment::stage(&path, n);
            let bytes = fs::read(stage.join("training_receipt.json"))?;
            let s: Value = serde_json::from_slice(&bytes)?;
            if r["stage_receipts"][n - 1] != original::hash(&bytes)
                || s["source_fingerprint"] != fingerprint
                || s["cell"] != json!(cell)
                || s["stage"] != n
                || s["updates"] != p["updates_per_stage"]
                || s["status"] != "sealed"
                || s["heldout_scored"] != false
                || (n == stages && r["final_sha256"] != s["final_sha256"])
            {
                return Err("changed sealed growth stage".into());
            }
            for (file, key) in [
                ("endpoint.json", "endpoint_sha256"),
                ("cache.json", "cache_sha256"),
                ("history.jsonl", "history_sha256"),
                ("training.jsonl", "trace_sha256"),
            ] {
                if s[key] != original::hash(&fs::read(stage.join(file))?) {
                    return Err("changed sealed growth artifact".into());
                }
            }
        }
    }
    Ok(())
}
pub fn initial(
    workspace: &Path,
    storage: &Path,
    cohort: &Path,
    cell: &Cell,
) -> Result<(Model, Value, baseline::data::Task)> {
    let (model, row, task) = source::load(workspace, storage, cell)?;
    let m = original::score::read(&cell.path(cohort).join("metadata.json"))?;
    if m["initial_sha256"] != original::hash(&serde_json::to_vec(&model)?)
        || m["original"] != row
        || m["selection"] != task.selection
    {
        return Err("original growth scoring identity changed".into());
    }
    Ok((model, row, task))
}
pub fn transition(initial: &Model, path: &Path, active: usize) -> Result<Model> {
    let r = original::score::read(&path.join("training_receipt.json"))?;
    let endpoint: Model = serde_json::from_slice(&fs::read(path.join("endpoint.json"))?)?;
    endpoint.validate()?;
    if r["initial_sha256"] != original::hash(&serde_json::to_vec(initial)?)
        || r["base_sha256"] != initial.base_hash(active)?
        || endpoint.base_hash(active)? != initial.base_hash(active)?
        || r["active"] != active
        || r["active_mass"] != initial.blocks[active].weight
        || r["final_sha256"] != original::hash(&serde_json::to_vec(&endpoint)?)
    {
        return Err("stage initial state or held mixture changed".into());
    }
    Ok(endpoint)
}
