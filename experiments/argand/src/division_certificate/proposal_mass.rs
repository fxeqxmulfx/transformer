//! Conditional success of one production Gaussian perturbation at fixed weights.
//! A failed batch suffices to reject a copy. Every accepted copy sees every row.
use super::{checkpoint, reconstructed_hidden};
use crate::{
    Result, checkpoint_proposal::add_gaussian, engine::adamw::FIRST_NOISE_STDDEV, hash, margins,
    metrics::Packed, random::Random,
};
use serde_json::{Value, json};
use std::{fs, path::Path};

pub(crate) fn measure(cell: &Path, seed_file: &Path, draws: usize) -> Result<Value> {
    let seed_bytes = fs::read(seed_file)?;
    let seeds: Vec<u32> = std::str::from_utf8(&seed_bytes)?
        .split_whitespace()
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    if draws == 0 || draws > seeds.len() {
        return Err("draw count must be positive and within the supplied seed file".into());
    }
    let (record, mut engine) = checkpoint(cell, false)?;
    let base = engine.weights()?;
    let hidden = reconstructed_hidden()?;
    let order: Vec<_> = (0..u32::try_from(hidden.len())?).collect();
    let mut batches = Vec::new();
    let mut strict = 0;
    let mut worst = 0;
    let mut minimum = f64::INFINITY;
    for indices in order.chunks(engine.config.max_batch) {
        let mut batch = Packed::default();
        batch.fill(&hidden, indices);
        for row in margins::rows(&mut engine, &batch)? {
            strict += usize::from(row.gap > 0.0);
            if row.gap < minimum {
                minimum = row.gap;
                worst = batches.len();
            }
        }
        batches.push(batch);
    }
    if record["strict"]["strict_correct"] != strict || record["strict"]["total"] != hidden.len() {
        return Err("reloaded hidden strict score differs from the source checkpoint".into());
    }
    // Preserve each original batch and its order internally for FP32 reproducibility.
    batches.swap(0, worst);
    let mut copy = base.clone();
    let mut accepted = Vec::new();
    let mut passed_first_batch = 0;
    let mut smallest_first_gap = f64::INFINITY;
    let mut largest_first_gap = f64::NEG_INFINITY;
    for &seed in &seeds[..draws] {
        copy.copy_from_slice(&base);
        add_gaussian(&mut copy, &mut Random::new(seed), FIRST_NOISE_STDDEV);
        engine.restore(&copy)?;
        let mut success = true;
        for (i, batch) in batches.iter().enumerate() {
            let gap = margins::rows(&mut engine, batch)?
                .iter()
                .map(|row| row.gap)
                .fold(f64::INFINITY, f64::min);
            if i == 0 {
                smallest_first_gap = smallest_first_gap.min(gap);
                largest_first_gap = largest_first_gap.max(gap);
                passed_first_batch += usize::from(gap > 0.0);
            }
            if gap <= 0.0 {
                success = false;
                break;
            }
        }
        if success {
            accepted.push(seed);
        }
    }
    Ok(
        json!({"identity":record["identity"],"step":record["step"],"weights_sha256":record["weights_sha256"],
        "gpu":engine.cuda.name(),"parameters":base.len(),"sigma":FIRST_NOISE_STDDEV,"draws":draws,
        "seed_file_sha256":hash(&seed_bytes),"ordinary":{"strict_correct":strict,"total":hidden.len(),"min_gap":minimum},
        "successes":accepted.len(),"success_seeds":accepted,"first_batch_index":worst,
        "passed_first_batch":passed_first_batch,"first_batch_min_gap_range":[smallest_first_gap,largest_first_gap]}),
    )
}
