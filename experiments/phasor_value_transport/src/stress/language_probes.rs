//! Post-training evaluations of the neutral-free block languages. The target
//! follows BlockCut.mem_altPlus_succ_iff. The original data oracle separately
//! recounts every prefix before a probe is handed to the checkpoint scorer.
use super::*;

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
    spec["length"] = json!(11);
    let exhaustive = (0..1 << 10)
        .map(|bits| labeled(&spec, word(bits, 10), blocks))
        .collect::<basis::Result<Vec<_>>>()?;
    task.splits
        .insert("test/exhaustive-10-letters".into(), exhaustive);

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
