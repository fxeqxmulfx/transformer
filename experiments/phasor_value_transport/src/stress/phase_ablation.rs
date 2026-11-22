//! Matched interventions on the sine and centered cosine outputs of a trained input-phase read.
use super::*;
use crate::head::{Head, MEMORY_WIDTH};

pub fn run(name: &str, seed: u32, out: &Path) -> Result<()> {
    if architecture::HEAD != Head::InputPhase {
        return Err("phase ablation requires the input-phase-read architecture".into());
    }
    let (row, mut task) = load(name)?;
    if matches!(name, "basis-parity16" | "basis-parity20" | "basis-parity24") {
        catalog::add_parity_probes(&mut task)?;
    }
    let cfg = task_config(&row, &task, false)?;
    let dir = out.join(name).join(format!("seed{seed}"));
    let (saved, original) = embedding::verified_weights(&row, seed, &cfg, &dir)?;
    let score = |cfg: Config, weights: &[f32]| -> Result<Value> {
        let bytes: Vec<_> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(benchmark::score_checkpoint(cfg, &task, &bytes)?.0)
    };
    let baseline = score(cfg.clone(), &original)?;
    if baseline[&task.selection]["correct"] != saved["observation"]["correct"] {
        return Err("phase checkpoint failed to reproduce its held-out score".into());
    }
    let increment = architecture::memory_start(&cfg);
    let output = increment + cfg.head.phase_inputs() * cfg.width;
    let channels = 2 * MEMORY_WIDTH;
    let span = output..output + channels * cfg.width;
    let mut scores = serde_json::Map::new();
    let mut removed = original.clone();
    for (name, range) in [
        ("sine_removed", 0..MEMORY_WIDTH),
        ("cosine_removed", MEMORY_WIDTH..channels),
    ] {
        removed.copy_from_slice(&original);
        for row in removed[span.clone()].chunks_exact_mut(channels) {
            row[range.clone()].fill(0.);
        }
        scores.insert(name.into(), score(cfg.clone(), &removed)?);
    }
    removed[span.clone()].fill(0.);
    scores.insert("phase_removed".into(), score(cfg.clone(), &removed)?);
    // InputPhaseRead.inputPhaseFeature_zero identifies the double ablation
    // with the trained stream backbone. Verify the two actual dense forwards.
    let mut stream = cfg.clone();
    stream.head = Head::Stream;
    let frequencies = layout::frequency_offset(&cfg);
    if layout::frequency_offset(&stream) != increment || frequencies != span.end {
        return Err("unexpected layout between the stream read and rotary frequencies".into());
    }
    let stream_weights = [&original[..increment], &original[frequencies..]].concat();
    if score(stream.clone(), &stream_weights)? != scores["phase_removed"] {
        return Err("phase removal changed an underlying stream score".into());
    }
    let length = cfg.max_length.min(4);
    let mut rng = random::Random::new(526);
    let dense: Vec<_> = (0..cfg.width * length)
        .map(|_| rng.gauss(0., 1.) as f32)
        .collect();
    let infer = |mut cfg: Config, weights: &[f32]| {
        cfg.max_batch = 1;
        cfg.max_length = length;
        Engine::new(cfg, weights)?.infer_embeddings(&dense, 1, length)
    };
    let actual = infer(cfg, &removed)?;
    let control = infer(stream, &stream_weights)?;
    let identical = |a: &[f32], b: &[f32]| {
        a.iter()
            .map(|v| v.to_bits())
            .eq(b.iter().map(|v| v.to_bits()))
    };
    if !identical(&actual.hidden, &control.hidden) || !identical(&actual.logits, &control.logits) {
        return Err("phase removal changed an off-table stream feature or logit".into());
    }
    println!(
        "{}",
        json!({"case":name,"seed":seed,"weights_sha256":saved["weights_sha256"],
        "model":MODEL,"kernite_version":KERNITE_VERSION,"selection":task.selection,
        "step":saved["observation"]["step"],"baseline":baseline,"ablations":scores,
        "stream_control":{"bit_identical_features":actual.hidden.len(),"bit_identical_logits":actual.logits.len()}})
    );
    Ok(())
}
