//! Score an arbitrary sequence of already embedded vectors with a saved stress checkpoint.
use super::*;

pub(super) fn verified_weights(
    row: &Value,
    seed: u32,
    cfg: &Config,
    dir: &Path,
) -> Result<(Value, Vec<f32>)> {
    let result: Value = serde_json::from_slice(&fs::read(dir.join("result.json"))?)?;
    report::validate(row, seed, dir, &result)?;
    let bytes = fs::read(dir.join("weights.f32"))?;
    if result["model"] != MODEL
        || result["weights_sha256"] != hash(&bytes)
        || result["suite_sha256"] != suite_hash(row)
        || result["fp32_affines"] != precision::FP32_AFFINE_REFERENCE
        || result["observation"]["case"] != row["name"]
        || result["observation"]["seed"] != seed
        || bytes.len() != 4 * layout::total_count(cfg)
    {
        return Err("checkpoint identity, shape, or precision mismatch".into());
    }
    let weights = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    Ok((result, weights))
}

pub fn run(name: &str, seed: u32, out: &Path, input_file: &Path) -> Result<()> {
    let input: Vec<Vec<f32>> = serde_json::from_reader(fs::File::open(input_file)?)?;
    let (row, task) = load(name)?;
    let mut cfg = task_config(&row, &task, false)?;
    let length = input.len();
    let width = cfg.width;
    if length == 0
        || input
            .iter()
            .any(|v| v.len() != width || v.iter().any(|x| !x.is_finite()))
    {
        return Err(format!(
            "expected a nonempty sequence of finite {width}-coordinate embeddings"
        )
        .into());
    }
    // CUDA attention addresses a head-by-length-by-length buffer with u32 indices.
    let valid_addresses = length
        .checked_mul(length)
        .and_then(|n| n.checked_mul(cfg.heads))
        .and_then(|n| u32::try_from(n).ok())
        .is_some();
    if !valid_addresses {
        return Err("embedding sequence exceeds CUDA attention addressing".into());
    }
    cfg.max_batch = 1;
    cfg.max_length = length;
    let vocab = cfg.vocab;
    let dir = out.join(name).join(format!("seed{seed}"));
    let (saved, weights) = verified_weights(&row, seed, &cfg, &dir)?;
    let mut e = Engine::new(cfg, &weights)?;
    let y = e.infer_embeddings(&input.into_iter().flatten().collect::<Vec<_>>(), 1, length)?;
    println!(
        "{}",
        json!({"model":MODEL, "weights_sha256":saved["weights_sha256"], "kernite_version":KERNITE_VERSION,
        "hidden":y.hidden.chunks_exact(width).collect::<Vec<_>>(), "logits":y.logits.chunks_exact(vocab).collect::<Vec<_>>() })
    );
    Ok(())
}
