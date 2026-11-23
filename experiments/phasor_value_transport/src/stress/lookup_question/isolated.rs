//! Final-position votes at unit gain, projected against the entire tied
//! vocabulary. Inputs pass through the learned geometric dictionary.
use super::*;

/// Use the pinned DiffGPT base_ops::rms warp's sum order. The host
/// reciprocal square root can round differently from CUDA rsqrt.approx.
fn normalized(row: &[f32]) -> Vec<f32> {
    let mut lanes = [0.0f32; 32];
    for (j, &x) in row.iter().enumerate() {
        lanes[j % 32] += x * x;
    }
    for stride in [16, 8, 4, 2, 1] {
        for i in 0..stride {
            lanes[i] += lanes[i + stride];
        }
    }
    let scale = (lanes[0] / row.len() as f32 + f32::EPSILON).sqrt().recip();
    row.iter().map(|&x| scale * x).collect()
}

fn classify(embeddings: &[f32], hidden: &[f32]) -> (usize, bool) {
    let mut winner = (0, false);
    let mut highest = f32::NEG_INFINITY;
    for (id, word) in embeddings.chunks_exact(hidden.len()).enumerate() {
        let score = crate::slot_memory::dot(word, hidden);
        if score > highest {
            highest = score;
            winner = (id, false);
        } else if score == highest {
            winner.1 = true;
        }
    }
    winner
}

pub fn run(seed: u32, out: &Path, passes: usize, selection: &str, gain: &str) -> Result<()> {
    if passes == 0 {
        return Err("an isolated read requires a positive pass count".into());
    }
    let (row, task) = data()?;
    let cfg = task_config(&row, &task, true)?;
    let (mut weights, point) = checkpoint(seed, out, &row, &cfg)?;
    let digest = weight_digest(&weights);
    let range = coefficients::table_range(&cfg)?;
    let params = coefficients::transform(&mut weights[range.clone()], selection, gain)?;
    let table = &weights[range];
    let layout = crate::slot_memory::Layout::new(&cfg);
    let store = &weights[architecture::memory_start(&cfg)..];
    let mut routing = crate::slot_memory::Routing::new(cfg.head);
    routing.geometry(&layout, store);
    let d = cfg.width;
    let embeddings = &weights[..cfg.vocab * d];
    let values: Vec<_> = embeddings.chunks_exact(d).map(normalized).collect();
    let addresses: Vec<_> = values
        .iter()
        .map(|v| routing.nearest(&layout, store, v))
        .collect();
    let names = vocabulary();
    let mut scores = BTreeMap::new();
    for (split, rows) in &task.splits {
        let mut correct = 0;
        let mut ties = 0;
        let mut fixture = None;
        for example in rows {
            let prompt = &example.prompt;
            let keys: Vec<_> = prompt
                .iter()
                .map(|&token| addresses[token as usize])
                .collect();
            let votes = crate::typed_lookup::final_votes(&keys, passes, table);
            let mut hidden = vec![0.; d];
            for (&v, &token) in votes.iter().zip(prompt) {
                if v != 0. {
                    for (dst, &x) in hidden.iter_mut().zip(&values[token as usize]) {
                        *dst += v * x;
                    }
                }
            }
            let (predicted, tied) = classify(embeddings, &hidden);
            correct += usize::from(example.answer.as_slice() == [predicted as i64]);
            ties += usize::from(tied);
            if split == "test/mira" {
                fixture = Some(&names[predicted]);
            }
        }
        let mut score = json!({"correct":correct,"total":rows.len(),"max_logit_ties":ties});
        if let Some(word) = fixture {
            score["prediction"] = json!(word);
        }
        scores.insert(split, score);
    }
    println!(
        "{}",
        json!({"model":MODEL,"seed":seed,"step":point["step"],"weights_sha256":digest,
        "protocol_sha256":row["catalog_sha256"],"training_kernite":KERNITE_VERSION,"passes":passes,
        "evaluation":"isolated unit-gain read with tied full-vocabulary projection on CPU",
        "coefficients":params,"scores":scores})
    );
    Ok(())
}
