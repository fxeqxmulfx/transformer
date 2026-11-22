//! Post-training evaluations of the neutral-free block languages. The target
//! follows BlockCut.mem_altPlus_succ_iff. The original data oracle separately
//! recounts every prefix before a probe is handed to the checkpoint scorer.
use super::*;

const LETTERS: usize = 10;
pub(super) const EXHAUSTIVE: &str = "test/exhaustive-10-letters";

fn word(bits: usize, width: usize) -> Vec<i64> {
    (0..width)
        .rev()
        .map(|i| 9 + ((bits >> i) & 1) as i64)
        .collect()
}

fn labeled(spec: &Value, letters: Vec<i64>, blocks: usize) -> basis::Result<Example> {
    let mut tokens = vec![1];
    tokens.extend(letters);
    let mut count = 0;
    let mut labels = vec![-100];
    for pair in tokens.windows(2) {
        count += usize::from(pair[0] != pair[1]);
        labels.push(if tokens[1] == 9 && count == blocks {
            16
        } else {
            15
        });
    }
    let row = basis::wire::example("blocks", tokens, labels);
    basis::independent::check(spec, &row)?;
    Ok(catalog::sequence_example(row))
}

pub(super) fn add(task: &mut Task, name: &str) -> basis::Result<()> {
    let mut spec = basis::dataset::specification(name)?;
    if spec["task"] != "blocks" || spec["neutral_fraction"] != 0.0 {
        return Err("expected the neutral-free binary block language".into());
    }
    let blocks = basis::generator::number(&spec, "blocks")?;
    spec["min_length"] = Value::Null;
    spec["length"] = json!(LETTERS + 1);
    let exhaustive = (0..1 << LETTERS)
        .map(|bits| labeled(&spec, word(bits, LETTERS), blocks))
        .collect::<basis::Result<Vec<_>>>()?;
    task.splits.insert(EXHAUSTIVE.into(), exhaustive);

    spec["length"] = json!(128);
    let mut alternating = Vec::new();
    for bits in 0..1 << 6 {
        let mut letters = word(bits, 6);
        let mut last = letters[5];
        while letters.len() < 127 {
            last = 19 - last;
            letters.push(last);
        }
        alternating.push(labeled(&spec, letters, blocks)?);
    }
    task.splits
        .insert("test/alternating-tail-128".into(), alternating);
    Ok(())
}

/// Evaluate words in length and lexicographic order, with no suffix and
/// every label masked. Recheck an observed mistake on its own before returning it.
pub(super) fn shortest_error(engine: &mut Engine, words: &[Example]) -> Result<Value> {
    let batch = engine.config.max_batch;
    for depth in 1..=LETTERS {
        let length = depth + 1;
        let count = 1 << depth;
        for first in (0..count).step_by(batch) {
            let end = (first + batch).min(count);
            // All ten-letter words starting with a given prefix are contiguous.
            let source = |index: usize| &words[index << (LETTERS - depth)];
            let tokens: Vec<_> = (first..end)
                .flat_map(|i| source(i).tokens[..length].iter().map(|&t| t as u32))
                .collect();
            let masked = vec![u32::MAX; tokens.len()];
            engine.forward(&tokens, &masked, end - first, length)?;
            let predictions = engine.predictions.read()?;
            for i in first..end {
                let expected = source(i).targets[depth] as u32;
                let offset = (i - first) * length;
                if predictions[offset + depth] == expected {
                    continue;
                }
                let prefix = &tokens[offset..offset + length];
                engine.forward(prefix, &masked[..length], 1, length)?;
                let predicted = engine.predictions.read()?[depth];
                if predicted != expected {
                    let letters: String = prefix[1..]
                        .iter()
                        .map(|&t| if t == 9 { 'a' } else { 'b' })
                        .collect();
                    return Ok(
                        json!({"letters":letters,"expected_token":expected,"predicted_token":predicted}),
                    );
                }
            }
        }
    }
    Ok(Value::Null)
}
