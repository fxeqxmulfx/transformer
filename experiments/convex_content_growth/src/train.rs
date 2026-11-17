//! Equal-update stage continuation or complete convex growth, no held-out reads.
use crate::{
    Result,
    experiment::{self, Cell},
    phase, source,
};
use bilinear::experiment as original;
use serde_json::json;
use std::{fs, path::Path};
pub fn run(workspace: &Path, storage: &Path, cohort: &Path, cell: &Cell) -> Result<()> {
    let p = experiment::protocol()?;
    let (mut model, row, task) = source::load(workspace, storage, cell)?;
    let path = cell.path(cohort);
    fs::create_dir_all(path.parent().ok_or("parent")?)?;
    fs::create_dir(&path)?;
    let fingerprint = experiment::fingerprint();
    original::write_json(
        &path.join("metadata.json"),
        &json!({"source_fingerprint":fingerprint,"binary_sha256":original::hash(&fs::read(std::env::current_exe()?)?),"protocol":p,"input":source::pin(cell)?,"original":row,"selection":task.selection,"cell":cell,"initial_sha256":original::hash(&serde_json::to_vec(&model)?)}),
    )?;
    if row["case"]["batch"] != p["batch"] {
        return Err("original batch changed".into());
    }
    let mut receipts = Vec::new();
    for n in 1..=p["stages"].as_u64().ok_or("stages")? as usize {
        let active = experiment::prepare(&mut model, cell, n, &p)?;
        let context = phase::Context {
            workspace,
            storage,
            path: &path,
            cell,
            p: &p,
            stage: n,
            active,
            fingerprint: &fingerprint,
        };
        phase::run(&mut model, &task.splits["train"], &context)?;
        receipts.push(original::hash(&fs::read(
            experiment::stage(&path, n).join("training_receipt.json"),
        )?));
    }
    original::write_json(
        &path.join("training_receipt.json"),
        &json!({"source_fingerprint":fingerprint,"cell":cell,"updates":p["stages"].as_u64().ok_or("stages")?*p["updates_per_stage"].as_u64().ok_or("updates")?,"stage_receipts":receipts,"final_sha256":original::hash(&serde_json::to_vec(&model)?),"status":"sealed","heldout_scored":false}),
    )?;
    Ok(())
}
