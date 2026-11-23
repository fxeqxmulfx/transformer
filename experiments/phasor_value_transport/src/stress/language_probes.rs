//! Post-training evaluations of the block languages. After neutral letters
//! are removed, the target follows BlockCut.mem_altPlus_succ_iff. The
//! independent data oracle separately recounts every prefix.
use super::*;

const LETTERS: usize = 10;
const REPEATS: usize = 32;
pub(super) const EXHAUSTIVE: &str = "test/exhaustive-10-letters";
pub(super) const STUTTER: &str = "test/stutter-8x32";


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

fn exhaustive(spec: &mut Value, blocks: usize, width: usize) -> basis::Result<Vec<Example>> {
    spec["length"] = json!(width + 1);
    (0..1 << width)
        .map(|bits| labeled(spec, word(bits, width), blocks))
        .collect()
}

pub(super) fn add(task: &mut Task, name: &str, extended: bool) -> basis::Result<()> {
    let mut spec = basis::dataset::specification(name)?;
    if spec["task"] != "blocks" || spec["neutral_fraction"] != 0.0 {
        return Err("expected the neutral-free binary block language".into());
    }
    let blocks = basis::generator::number(&spec, "blocks")?;
    spec["min_length"] = Value::Null;
    task.splits
        .insert(EXHAUSTIVE.into(), exhaustive(&mut spec, blocks, LETTERS)?);

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
    if extended {
        task.splits.insert(
            "test/exhaustive-12-letters".into(),
            exhaustive(&mut spec, blocks, 12)?,
        );
        let seeds = exhaustive(&mut spec, blocks, 8)?;
        spec["length"] = json!(1 + 8 * REPEATS);
        let mut stretched = Vec::with_capacity(seeds.len());
        for seed in seeds {
            let letters = seed.tokens[1..]
                .iter()
                .flat_map(|&t| std::iter::repeat_n(t, REPEATS))
                .collect();
            let long = labeled(&spec, letters, blocks)?;
            let (groups, tail) = long.targets[1..].as_chunks::<REPEATS>();
            if !tail.is_empty()
                || groups.len() != seed.targets.len() - 1
                || !groups
                    .iter()
                    .zip(&seed.targets[1..])
                    .all(|(copies, label)| copies.iter().all(|t| t == label))
            {
                return Err("stretched prefix labels differ from the original word".into());
            }
            stretched.push(long);
        }
        task.splits.insert(STUTTER.into(), stretched);
    }
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

fn final_prediction(engine: &mut Engine, prefix: &[u32]) -> Result<u32> {
    engine.forward(prefix, &vec![u32::MAX; prefix.len()], 1, prefix.len())?;
    Ok(engine.predictions.read()?[prefix.len() - 1])
}

/// Locate the first missed prefix in probe order. Score the prefix again
/// in isolation, then remove the repetitions and score its original word.
/// All three predictions are made with every output label masked.
pub(super) fn stutter_error(engine: &mut Engine, words: &[Example]) -> Result<Value> {
    for rows in words.chunks(engine.config.max_batch) {
        let length = rows[0].tokens.len();
        let tokens: Vec<_> = rows
            .iter()
            .flat_map(|row| row.tokens.iter().map(|&t| t as u32))
            .collect();
        engine.forward(&tokens, &vec![u32::MAX; tokens.len()], rows.len(), length)?;
        let predictions = engine.predictions.read()?;
        for (i, row) in rows.iter().enumerate() {
            let offset = i * length;
            for (pos, &label) in row.targets.iter().enumerate().skip(1) {
                let batched = predictions[offset + pos];
                if batched as i64 == label {
                    continue;
                }
                let prefix = &tokens[offset..=offset + pos];
                let mut original = vec![prefix[0]];
                original.extend(prefix[1..].iter().step_by(REPEATS));
                let letters: String = original[1..]
                    .iter()
                    .map(|&t| if t == 9 { 'a' } else { 'b' })
                    .collect();
                return Ok(json!({
                    "source_prefix": letters,
                    "payload_position": pos,
                    "expected_token": label,
                    "batched_token": batched,
                    "standalone_token": final_prediction(engine, prefix)?,
                    "unstretched_token": final_prediction(engine, &original)?,
                }));
            }
        }
    }
    Err("masked repeated-word predictions did not reproduce the measured error".into())
}
