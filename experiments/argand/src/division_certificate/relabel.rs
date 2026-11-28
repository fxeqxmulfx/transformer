//! Compare a saved model to the same model in the reversed vocabulary.
//! Both forward passes use the native FP32 kernels. Labels for all legal
//! prompts come from the unknown-factor completion of the training split.
use super::{Completion, Counts, checkpoint};
use crate::{Result, division, margins, metrics, suite::publish};
use serde_json::{Value, json};
use std::path::Path;

pub(crate) fn check(dir: &Path) -> Result<Value> {
    let (record, mut engine) = checkpoint(dir, false)?;
    let completion = Completion::original_split()?;
    let examples = completion.examples();
    let order: Vec<_> = (0..examples.len() as u32).collect();
    let shape = division::shape();
    let vocab = shape.vocab;
    let stride = 2 * shape.width;
    let original = engine.weights()?;
    let mut renamed = original.clone();
    let rows = original[..vocab * stride].chunks_exact(stride);
    for (target, source) in renamed[..vocab * stride]
        .chunks_exact_mut(stride)
        .zip(rows.rev())
    {
        target.copy_from_slice(source);
    }

    let mut batch = metrics::Packed::default();
    let mut counts = [Counts::default(), Counts::default()];
    let mut compared = 0_u64;
    let mut bit_differences = 0_u64;
    let mut max_error = 0.0_f64;
    let mut changed_strict = 0;
    for indices in order.chunks(engine.config.max_batch) {
        batch.fill(&examples, indices);
        engine.restore(&original)?;
        let before = margins::masked_logits(&mut engine, &batch)?;
        let original_rows = margins::rows_from_logits(&before, vocab, &batch)?;
        for id in batch.tokens.iter_mut().chain(&mut batch.labels) {
            if (*id as usize) < vocab {
                *id = vocab as u32 - 1 - *id;
            }
        }
        engine.restore(&renamed)?;
        let after = margins::masked_logits(&mut engine, &batch)?;
        let renamed_rows = margins::rows_from_logits(&after, vocab, &batch)?;
        for (a, b) in before.chunks_exact(vocab).zip(after.chunks_exact(vocab)) {
            for (&x, &y) in a.iter().zip(b.iter().rev()) {
                if !x.is_finite() || !y.is_finite() {
                    return Err("nonfinite logit in paired vocabulary replay".into());
                }
                compared += 1;
                bit_differences += u64::from(x.to_bits() != y.to_bits());
                max_error = max_error.max((f64::from(x) - f64::from(y)).abs());
            }
        }
        for ((i, a), b) in indices.iter().zip(original_rows).zip(renamed_rows) {
            changed_strict += usize::from((a.gap > 0.0) != (b.gap > 0.0));
            counts[usize::from(completion.observed[*i as usize])].add(b);
        }
    }
    let report = json!({
        "case": division::NAME, "seed": record["identity"]["seed"], "step": record["step"],
        "weights_sha256": record["weights_sha256"], "observed_sha256": division::TRAIN_HASH,
        "permutation": "t -> 238 - t", "positions_per_prompt": shape.max_length,
        "compared_logits": compared, "bit_differences": bit_differences,
        "max_absolute_error": max_error, "changed_strict_answers": changed_strict,
        "renamed_hidden": counts[0].json(), "renamed_observed": counts[1].json()
    });
    publish(
        dir,
        "relabel-replay.json",
        &serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}
