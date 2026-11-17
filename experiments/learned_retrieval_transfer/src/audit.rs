//! Decode the retained class scores in original record order. The label sets
//! are used only for a post-training breakdown of raw accuracy.
use crate::{Result, data::Example, hash, inputs};
use learned_retrieval::cohort::{raw_file, read};
use serde_json::{Value, json};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

fn split(
    bytes: &[u8],
    rows: &[Example],
    vocabulary: usize,
    supervised: &BTreeSet<i64>,
    report: &Value,
) -> Result<Value> {
    if report["raw_logits_sha256"] != hash(bytes) {
        return Err("retained raw-logit fingerprint mismatch".into());
    }
    let mut vectors = bytes.chunks_exact(vocabulary * 8);
    let mut correct = [0usize; 2];
    let mut targets = [0usize; 2];
    let mut exact = 0;
    let mut loss = 0.;
    for row in rows {
        let mut whole = true;
        for &target in row.targets.iter().filter(|&&y| y >= 0) {
            let bytes = vectors.next().ok_or("missing complete logit vector")?;
            let z: Vec<_> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|x| f64::from_le_bytes(*x))
                .collect();
            if z.iter().any(|x| !x.is_finite()) {
                return Err("nonfinite retained logit".into());
            }
            let mut winner = 0;
            for (k, value) in z.iter().enumerate().skip(1) {
                if value.total_cmp(&z[winner]) == Ordering::Greater {
                    winner = k;
                }
            }
            let actual = usize::try_from(target)?;
            let target_score = z.get(actual).ok_or("target outside original vocabulary")?;
            let maximum = z[winner];
            loss +=
                z.iter().map(|x| (x - maximum).exp()).sum::<f64>().ln() + maximum - target_score;
            let hit = winner == actual;
            whole &= hit;
            let group = usize::from(!supervised.contains(&target));
            correct[group] += usize::from(hit);
            targets[group] += 1;
        }
        exact += usize::from(whole);
    }
    let total: usize = targets.iter().sum();
    let metrics = &report["metrics"];
    if vectors.next().is_some()
        || !vectors.remainder().is_empty()
        || total == 0
        || metrics["examples"] != rows.len()
        || metrics["targets"] != total
        || metrics["correct"] != correct.iter().sum::<usize>()
        || metrics["exact"] != exact
        || (metrics["loss"].as_f64().ok_or("missing full-class CE")? - loss / total as f64).abs()
            > 1e-12
    {
        return Err("independent decoding disagrees with the original scorer".into());
    }
    Ok(
        json!({"supervised_answer":{"targets":targets[0],"correct":correct[0]},
        "never_supervised_answer":{"targets":targets[1],"correct":correct[1]}}),
    )
}

pub fn run(name: &str, root: &Path) -> Result<Value> {
    let (p, data, _) = inputs(name)?;
    let supervised: BTreeSet<_> = data.task.splits["train"]
        .iter()
        .flat_map(|e| e.targets.iter().copied().filter(|&y| y >= 0))
        .collect();
    let mut input_counts: BTreeMap<i64, usize> = data
        .task
        .splits
        .values()
        .flatten()
        .flat_map(|row| row.targets.iter().copied())
        .filter(|y| *y >= 0 && !supervised.contains(y))
        .map(|y| (y, 0))
        .collect();
    for id in data.task.splits["train"].iter().flat_map(|row| &row.tokens) {
        if let Some(count) = input_counts.get_mut(id) {
            *count += 1;
        }
    }
    let dir = root.join("scores");
    let scores = read(&dir.join("scores.json"))?;
    let mut cells = serde_json::Map::new();
    for mode in &p.modes {
        for &seed in &p.seeds {
            let cell = p.cell(mode, seed)?;
            for name in [format!("{cell}-initial"), cell] {
                let mut splits = serde_json::Map::new();
                for (label, rows) in &data.task.splits {
                    let bytes = fs::read(raw_file(&dir, &name, label))?;
                    let groups = split(
                        &bytes,
                        rows,
                        data.task.vocab,
                        &supervised,
                        &scores[&name]["splits"][label],
                    )?;
                    splits.insert(label.clone(), groups);
                }
                cells.insert(name, Value::Object(splits));
            }
        }
    }
    if scores.as_object().ok_or("missing cells")?.len() != cells.len() {
        return Err("unexpected score cells".into());
    }
    Ok(
        json!({"case":name,"training_answer_classes":supervised.len(),
        "training_input_counts_for_never_supervised_answers":input_counts,"cells":cells}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn altered_score_counts_or_truncated_vectors_are_rejected() -> Result<()> {
        let (p, d, _) = inputs("suite-lookup")?;
        let rows = &d.task.splits["validation"][..1];
        let m = p.initial(d.task.vocab, 1)?;
        let (mut report, mut raw) = learned_retrieval::evaluation::prefix_split(&m, rows)?;
        let seen = BTreeSet::new();
        let groups = split(&raw, rows, m.vocabulary, &seen, &report)?;
        assert_eq!(groups["never_supervised_answer"]["targets"], 2);
        report["metrics"]["targets"] = json!(1);
        assert!(split(&raw, rows, m.vocabulary, &seen, &report).is_err());
        report["metrics"]["targets"] = json!(2);
        raw.pop();
        report["raw_logits_sha256"] = json!(hash(&raw));
        assert!(split(&raw, rows, m.vocabulary, &seen, &report).is_err());
        Ok(())
    }
}
