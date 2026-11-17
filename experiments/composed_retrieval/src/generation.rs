//! Independently frame teacher-forced and greedy queries in the original
//! per-example order. The trace identifies the prefix seen at every call.
use crate::Result;
use baseline::{data::Example, hash};
use serde_json::{Value, json};
use synthetic_data::generative::scoring;

pub fn validate_source(row: &Example, vocabulary: usize) -> Result<()> {
    let p = row.prompt.len();
    if p < 3
        || row.prompt.first() != Some(&1)
        || row.prompt.last() != Some(&18)
        || row.limit != p - 1
        || row.answer.len() != row.limit
        || row.answer.last() != Some(&17)
        || row.answer[..row.limit - 1].contains(&17)
        || row.field.is_some()
        || row.row_index.is_some()
        || row.tokens.len() != p + row.limit - 1
        || row.changes.len() != row.tokens.len()
        || row.targets.len() != row.tokens.len()
        || row.tokens[..p] != row.prompt
        || row.tokens[p..] != row.answer[..row.limit - 1]
        || row.targets[..p - 1].iter().any(|&t| t != -100)
        || row.targets[p - 1..] != row.answer
        || row
            .tokens
            .iter()
            .chain(&row.answer)
            .any(|&t| t < 0 || t as usize >= vocabulary)
    {
        return Err(
            "invalid original Rewrite mask, prompt, shifted answer or input-derived limit".into(),
        );
    }
    Ok(())
}

pub fn record_prefix(trace: &mut Vec<u8>, prefix: &[i64]) {
    trace.extend((prefix.len() as u64).to_le_bytes());
    for token in prefix {
        trace.extend(token.to_le_bytes());
    }
}

fn winner(vector: &[u8]) -> Result<usize> {
    let mut best = None;
    for (i, &bytes) in vector.as_chunks::<8>().0.iter().enumerate() {
        let z = f64::from_le_bytes(bytes);
        if !z.is_finite() {
            return Err("nonfinite generated-object logit".into());
        }
        if best.is_none_or(|(_, maximum)| z.total_cmp(&maximum).is_gt()) {
            best = Some((i, z));
        }
    }
    best.map(|(i, _)| i)
        .ok_or_else(|| "empty vocabulary".into())
}

/// Reconstruct model-written prefixes from argmax decisions. Check the
/// original scorer against the independent archived generation semantics.
/// Return the teacher vectors in masked record order for CE and exposure.
pub fn decode(
    rows: &[Example],
    vocabulary: usize,
    bytes: &[u8],
    prefix_sha256: &str,
    metrics: &Value,
) -> Result<Vec<u8>> {
    if rows.is_empty() || vocabulary <= 18 {
        return Err("empty Rewrite split or invalid vocabulary".into());
    }
    let mut vectors = bytes.chunks_exact(vocabulary.checked_mul(8).ok_or("vector overflow")?);
    let mut teacher = Vec::new();
    let mut trace = Vec::new();
    let (mut targets, mut correct, mut exact, mut ended) = (0, 0, 0, 0);
    let (mut written, mut extra, mut changes, mut change_hits) = (0, 0, 0, 0);
    let mut classes = vec![[0usize; 2]; vocabulary];
    for row in rows {
        validate_source(row, vocabulary)?;
        for (pos, _) in row.targets.iter().enumerate().filter(|&(_, &y)| y >= 0) {
            record_prefix(&mut trace, &row.tokens[..=pos]);
            teacher.extend_from_slice(vectors.next().ok_or("missing teacher-forced logits")?);
        }
        let output = scoring::rollout(&row.prompt, row.limit, vocabulary, |prefix| {
            record_prefix(&mut trace, prefix);
            winner(vectors.next().ok_or("missing free-generation logits")?)
        })?;
        let result = scoring::score(&row.answer, &output)?;
        targets += result.targets;
        correct += result.correct;
        exact += usize::from(result.exact);
        ended += usize::from(result.ended);
        written += result.generated_tokens;
        extra += result.extra_tokens;
        changes += result.transition_targets;
        change_hits += result.transition_correct;
        for (i, &label) in row.answer.iter().enumerate() {
            let label = label as usize;
            classes[label][0] += usize::from(output.get(i) == Some(&label));
            classes[label][1] += 1;
        }
    }
    if vectors.next().is_some() || !vectors.remainder().is_empty() || hash(&trace) != prefix_sha256
    {
        return Err("extra or partial vectors, or mismatched model-written prefix trace".into());
    }
    let fraction = |n: usize, d: usize| json!(n as f64 / d as f64);
    let teacher_metrics = &metrics["teacher_forced"];
    let present = classes.iter().filter(|g| g[1] > 0).count();
    let balanced = classes
        .iter()
        .filter(|g| g[1] > 0)
        .map(|g| g[0] as f64 / g[1] as f64)
        .sum::<f64>()
        / present as f64;
    if metrics["examples"] != rows.len()
        || metrics["example_count"] != rows.len()
        || metrics["targets"] != targets
        || metrics["target_count"] != targets
        || metrics["correct"] != correct
        || metrics["exact"] != exact
        || metrics["generated_tokens"] != written
        || metrics["extra_tokens"] != extra
        || metrics["sequence_accuracy"] != fraction(exact, rows.len())
        || metrics["final_answer_accuracy"] != fraction(exact, rows.len())
        || metrics["token_accuracy"] != fraction(correct, targets)
        || metrics["eos_rate"] != fraction(ended, rows.len())
        || metrics["transition_count"] != changes
        || metrics["transition_accuracy"] != fraction(change_hits, changes)
        || metrics["balanced_accuracy"] != json!(balanced)
        || metrics["loss"] != teacher_metrics["loss"]
        || metrics["example_loss"] != teacher_metrics["example_loss"]
        || teacher_metrics["exact"] != exact
        || metrics["classes"].as_object().map(|x| x.len()) != Some(present)
    {
        return Err("independent greedy rollout disagrees with the original split metrics".into());
    }
    for (id, [hit, total]) in classes.into_iter().enumerate().filter(|(_, g)| g[1] > 0) {
        let class = &metrics["classes"][id.to_string()];
        if class["correct"] != hit
            || class["count"] != total
            || class["accuracy"] != fraction(hit, total)
        {
            return Err("original generated-class counts differ".into());
        }
    }
    Ok(teacher)
}

#[cfg(test)]
mod tests {
    use super::*;
    use baseline::metrics::{self, Predictor};
    use std::cell::RefCell;

    struct Fixture {
        first: usize,
        close: bool,
        queries: RefCell<Vec<Vec<i64>>>,
        raw: RefCell<Vec<u8>>,
    }
    impl Predictor for Fixture {
        fn vocab(&self) -> usize {
            40
        }
        fn logits(&self, prefix: &[i64]) -> Vec<f64> {
            self.queries.borrow_mut().push(prefix.to_vec());
            let next = match prefix.last() {
                Some(18) => self.first,
                Some(36) => 37,
                Some(37) => 17,
                Some(39) if self.close => 17,
                _ => 39,
            };
            let mut z = vec![0.0_f64; self.vocab()];
            z[next] = 2.;
            self.raw
                .borrow_mut()
                .extend(z.iter().flat_map(|s| s.to_le_bytes()));
            z
        }
    }
    fn fixture() -> Example {
        Example {
            task: "copy".into(),
            tokens: vec![1, 36, 37, 18, 36, 37],
            targets: vec![-100, -100, -100, 36, 37, 17],
            changes: vec![false, false, false, true, true, true],
            prompt: vec![1, 36, 37, 18],
            answer: vec![36, 37, 17],
            limit: 3,
            field: None,
            row_index: None,
        }
    }

    #[test]
    fn original_scorer_feeds_back_its_error_and_preserves_the_full_target_denominator() -> Result<()>
    {
        let rows = [fixture()];
        for close in [true, false] {
            let model = Fixture {
                first: 39,
                close,
                queries: RefCell::default(),
                raw: RefCell::default(),
            };
            let m = metrics::evaluate(&model, &rows)?;
            let mut queries = model.queries.into_inner();
            assert_eq!(queries[3], rows[0].prompt);
            assert_eq!(queries[4], vec![1, 36, 37, 18, 39]);
            assert_eq!(queries.len(), if close { 5 } else { 6 });
            assert_eq!(m["teacher_forced"]["correct"], 2);
            assert_eq!(m["correct"], 0);
            assert_eq!(m["targets"], 3);
            assert_eq!(m["generated_tokens"], if close { 2 } else { 3 });
            assert_eq!(m["eos_rate"], if close { 1. } else { 0. });
            let digest = |qs: &[Vec<i64>]| {
                let mut trace = vec![];
                for q in qs {
                    record_prefix(&mut trace, q);
                }
                hash(&trace)
            };
            let trace = digest(&queries);
            let mut raw = model.raw.into_inner();
            assert_eq!(decode(&rows, 40, &raw, &trace, &m)?, raw[..3 * 40 * 8]);
            queries[4][4] = 36;
            assert!(decode(&rows, 40, &raw, &digest(&queries), &m).is_err());
            let mut changed = m.clone();
            changed["correct"] = json!(1);
            assert!(decode(&rows, 40, &raw, &trace, &changed).is_err());
            raw.push(0);
            assert!(decode(&rows, 40, &raw, &trace, &m).is_err());
        }
        let mut invalid = fixture();
        invalid.limit -= 1;
        assert!(validate_source(&invalid, 40).is_err());
        Ok(())
    }

    #[test]
    fn exact_answers_and_immediate_eos_use_the_same_original_target_mask() -> Result<()> {
        let rows = [fixture()];
        for (first, written, hits) in [(36, 3, 3), (17, 1, 0)] {
            let model = Fixture {
                first,
                close: true,
                queries: RefCell::default(),
                raw: RefCell::default(),
            };
            let m = metrics::evaluate(&model, &rows)?;
            let mut trace = Vec::new();
            for query in model.queries.into_inner() {
                record_prefix(&mut trace, &query);
            }
            assert_eq!(m["generated_tokens"], written);
            assert_eq!(m["correct"], hits);
            assert_eq!(m["targets"], 3);
            assert_eq!(m["exact"], usize::from(hits == 3));
            assert_eq!(m["eos_rate"], 1.);
            let raw = model.raw.into_inner();
            assert_eq!(
                decode(&rows, 40, &raw, &hash(&trace), &m)?,
                raw[..3 * 40 * 8]
            );
        }
        Ok(())
    }
}
