//! Test whether a saved solution survives a single row projection. The
//! source checkpoint is read-only. Every label is reconstructed from training.
use super::{Completion, Counts, checkpoint_with_batch};
use crate::{Result, data::Example, engine::Engine, hash, margins, metrics, suite::publish};
use serde_json::{Value, json};
use std::path::Path;

fn score(engine: &mut Engine, examples: &[Example], observed: &[bool]) -> Result<Value> {
    let mut counts = [Counts::default(), Counts::default()];
    let mut packed = metrics::Packed::default();
    let indices: Vec<_> = (0..examples.len() as u32).collect();
    for batch in indices.chunks(engine.config.max_batch) {
        packed.fill(examples, batch);
        for (&i, row) in batch.iter().zip(margins::rows(engine, &packed)?) {
            counts[usize::from(observed[i as usize])].add(row);
        }
    }
    Ok(
        json!({"hidden":counts[0].json(),"observed":counts[1].json(),
        "projection_row_norms":engine.projection_row_norms()?}),
    )
}

pub(crate) fn check(dir: &Path, first: bool, radius: f32) -> Result<Value> {
    let (record, mut engine) = checkpoint_with_batch(dir, first, 32)?;
    let completion = Completion::original_split()?;
    let examples = completion.examples();
    let before = score(&mut engine, &examples, &completion.observed)?;
    if before["hidden"]["strict_correct"] != record["strict"]["strict_correct"] {
        return Err("reconstructed hidden score disagrees with the source checkpoint".into());
    }
    engine.clip_projection_rows(radius)?;
    let after = score(&mut engine, &examples, &completion.observed)?;
    let bytes: Vec<_> = engine
        .weights()?
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let kind = if first { "first-strict" } else { "terminal" };
    let report = json!({"case":record["identity"]["case"],"seed":record["identity"]["seed"],
        "step":record["step"],"source_weights_sha256":record["weights_sha256"],
        "projected_weights_sha256":hash(&bytes),"training_sha256":crate::division::TRAIN_HASH,
        "radius":radius,"evaluation_batch":engine.config.max_batch,"before":before,"after":after});
    publish(
        dir,
        &format!("row-projection-{kind}-{radius}.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}
