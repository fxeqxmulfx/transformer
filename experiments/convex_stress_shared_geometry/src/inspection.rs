//! Hash-bound training-only inspection. Save the full forecast before fitting;
//! replay every candidate, choice, full state and original-record update exactly.
use crate::*;
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path};
fn write<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}
pub fn run(parent_path: &Path, expected: &str, role: &str, mode: Mode, root: &Path) -> Result<()> {
    if root.exists() {
        return Err("fresh inspection directory required".into());
    }
    let bytes = fs::metadata(parent_path)?.len();
    let parse_bound = bytes
        .checked_mul(32)
        .and_then(|x| x.checked_add(1 << 30))
        .ok_or("parse forecast overflow")?;
    if bytes > 1 << 30 || parse_bound > 56 * 1024u64.pow(3) {
        return Err("parent parse allowance refused".into());
    }
    let input = {
        let bytes = fs::read(parent_path)?;
        if hash(&bytes) != expected {
            return Err("complete parent hash differs".into());
        }
        let parent: Value = serde_json::from_slice(&bytes)?;
        convex_stress_shared_head::inspection::extract(&parent, role, expected)?
    };
    let policy = training::policies()?
        .into_iter()
        .find(|p| p.mode == mode)
        .ok_or("mode absent")?;
    let bounds = limits::forecast(&input, &policy)?;
    if bounds.buffer_bytes + parse_bound > 56 * 1024u64.pow(3) {
        return Err("parse plus component buffers refused".into());
    }
    fs::create_dir(root)?;
    write(&root.join("input.json"), &input)?;
    write(
        &root.join("forecast.json"),
        &json!({"parent_bytes":bytes,"parent_parse_buffer_forecast":parse_bound,
        "role":role,"policy":policy,"bounds":bounds,"original_heldout_evaluations":0}),
    )?;
    let outcome = training::run(input.clone(), policy)?;
    write(&root.join("outcome.json"), &outcome)?;
    let saved: training::Outcome = serde_json::from_slice(&fs::read(root.join("outcome.json"))?)?;
    training::replay(&input, &saved)?;
    let selected = outcome.initialization.pooled_scores[outcome.initialization.selected]
        .as_ref()
        .unwrap();
    let score = fit::score(
        &outcome.model,
        &features::Encoder::new(outcome.model.bank).records(&input),
        &(0..input.records.len()).collect::<Vec<_>>(),
    );
    write(
        &root.join("summary.json"),
        &json!({"mode":mode,"role":role,"parent_sha256":expected,
        "input_sha256":outcome.input_sha256,"model_sha256":digest(&outcome.model)?,"initial_model_sha256":outcome.initial_model_sha256,
        "initial_ce":outcome.initial_ce,"final_ce":outcome.final_ce,"training_score":score,
        "bank":outcome.model.bank,"ridge":outcome.initialization.refit.ridge,"direction":outcome.initialization.refit.direction,
        "gain":outcome.initialization.gain,"calibrated":outcome.initialization.refit.calibrated,
        "separated_blocks":outcome.initialization.refit.interval_gaps.iter().filter(|gap|gap.is_some_and(|g|g>0.)).count(),
        "conditional_fold_score":selected,
        "local_fold_scores":outcome.initialization.local_controls.iter().map(|c|&c.score).collect::<Vec<_>>(),
        "work":outcome.work,"bounds":outcome.bounds,"maximum_product_residual":outcome.maximum_product_residual,
        "state_replay_exact":true,"original_heldout_evaluations":0,"unknown_target_certified":false}),
    )
}
