//! Fixed-weight removals of the learned projections in the triple phase read.
use super::*;
use crate::{
    head::{Head, MEMORY_WIDTH},
    slot_memory,
};

fn partition_error(full: &[f32], base: &[f32], sine: &[f32], cosine: &[f32]) -> Result<f64> {
    if ![base.len(), sine.len(), cosine.len()]
        .iter()
        .all(|&n| n == full.len())
    {
        return Err("phase projection result shapes disagree".into());
    }
    let mut worst = 0.0_f64;
    for (((&a, &b), &c), &d) in full.iter().zip(base).zip(sine).zip(cosine) {
        let [a, b, c, d] = [a, b, c, d].map(f64::from);
        let error = ((a + b) - (c + d)).abs() / (1. + a.abs() + b.abs() + c.abs() + d.abs());
        if !error.is_finite() {
            return Err("nonfinite phase projection partition".into());
        }
        worst = worst.max(error);
    }
    Ok(worst)
}

pub fn run(name: &str, seed: u32, out: &Path) -> Result<()> {
    if !architecture::HEAD.has_triple_phase() {
        return Err("phase projection removal requires a triple phase read".into());
    }
    let (row, mut task) = load(name)?;
    if matches!(name, "basis-parity16" | "basis-parity20" | "basis-parity24") {
        catalog::add_parity_probes(&mut task)?;
    }
    let cfg = task_config(&row, &task, false)?;
    let dir = out.join(name).join(format!("seed{seed}"));
    let (saved, original) = embedding::verified_weights(&row, seed, &cfg, &dir)?;
    let span = slot_memory::phase_outputs(&cfg);
    if span.end != layout::frequency_offset(&cfg) {
        return Err("phase projections do not end at the expected parameter boundary".into());
    }
    let mut model = Engine::new(cfg.clone(), &original)?;
    let mut dense_cfg = cfg.clone();
    dense_cfg.max_batch = 1;
    dense_cfg.max_length = cfg.max_length.min(4);
    let len = dense_cfg.max_length;
    let mut dense_model = Engine::new(dense_cfg.clone(), &original)?;
    let mut rng = random::Random::new(73251);
    let input: Vec<_> = (0..len * cfg.width)
        .map(|_| rng.gauss(0., 1.) as f32)
        .collect();
    let mut dense = BTreeMap::new();
    let mut scores = BTreeMap::new();
    let mut weights = original.clone();
    for arm in [
        "full",
        "without_sine",
        "without_cosine",
        "without_phase",
        "phase_only",
    ] {
        weights.copy_from_slice(&original);
        if arm == "phase_only" {
            crate::slot_checks::disable_reads(&cfg, &mut weights, &[0, 1, 2, 3]);
        } else {
            for outputs in weights[span.clone()]
                .as_chunks_mut::<{ 2 * MEMORY_WIDTH }>()
                .0
            {
                if matches!(arm, "without_sine" | "without_phase") {
                    outputs[..MEMORY_WIDTH].fill(0.);
                }
                if matches!(arm, "without_cosine" | "without_phase") {
                    outputs[MEMORY_WIDTH..].fill(0.);
                }
            }
        }
        model.parameters.write(&weights)?;
        let mut splits = BTreeMap::new();
        for (split, examples) in &task.splits {
            if split == &task.selection || split == "test" || split.starts_with("test/") {
                let r = metrics::evaluate(&mut model, examples, true)?;
                splits.insert(split, json!({"correct":r["exact"],"total":examples.len()}));
            }
        }
        scores.insert(arm, json!(splits));
        if arm != "phase_only" {
            dense_model.parameters.write(&weights)?;
            dense.insert(arm, dense_model.infer_embeddings(&input, 1, len)?);
        }
    }
    let full = &scores["full"][&task.selection];
    if [full["correct"].clone(), full["total"].clone()]
        != [
            saved["observation"]["correct"].clone(),
            saved["observation"]["total"].clone(),
        ]
    {
        return Err("matched phase ablation missed the saved baseline".into());
    }
    let (a, b, c, d) = (
        &dense["full"],
        &dense["without_phase"],
        &dense["without_cosine"],
        &dense["without_sine"],
    );
    let hidden_error = partition_error(&a.hidden, &b.hidden, &c.hidden, &d.hidden)?;
    let logit_error = partition_error(&a.logits, &b.logits, &c.logits, &d.logits)?;
    if hidden_error > 0.00005 || logit_error > 0.00005 {
        return Err(format!(
            "dense phase partition differs: hidden={hidden_error} logits={logit_error}"
        )
        .into());
    }
    drop(model);
    drop(dense_model);
    // Remove the entire phase parameter block. The remaining weights belong
    // to the corresponding original trunk, including its learned frequencies.
    let base_head = if let Some(search) = cfg.head.peer_search() {
        Head::PeerTriple(search)
    } else if cfg.head.uses_lookup_capacity() {
        Head::LookupTripleTransport
    } else {
        Head::TripleTransport
    };
    dense_cfg.head = base_head;
    let mut base_weights = original[..slot_memory::phase_start(&cfg)].to_vec();
    base_weights.extend_from_slice(&original[span.end..]);
    if base_weights.len() != layout::total_count(&dense_cfg) {
        return Err("phase removal produced an invalid trunk checkpoint".into());
    }
    let mut base = Engine::new(dense_cfg, &base_weights)?;
    let baseline = base.infer_embeddings(&input, 1, len)?;
    if baseline.hidden != b.hidden || baseline.logits != b.logits {
        return Err("zeroed projections differ from the corresponding triple trunk".into());
    }
    println!(
        "{}",
        json!({"case":name,"seed":seed,"model":MODEL,"weights_sha256":saved["weights_sha256"],
        "kernite_version":trained_with(&saved)?,"scored_with_kernite":KERNITE_VERSION,
        "step":saved["observation"]["step"],"selection":task.selection,"scores":scores,
        "dense":{"hidden":a.hidden.len(),"logits":a.logits.len(),"hidden_relative_error":hidden_error,"logit_relative_error":logit_error,
        "zero_projection_trunk":base_head.id(),"matches_trunk":true}})
    );
    Ok(())
}
