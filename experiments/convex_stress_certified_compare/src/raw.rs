//! Port the read-only task347 audit atf0e57fb5 to CM63 certificates. Reconstruct
//! every source row, all class scores, raw decision, CE, transition and rollout.
use crate::{adapter::Input, model::Model, *};
use dataset::data::Example;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

struct Reader {
    inner: BufReader<File>,
    lines: u64,
}
impl Reader {
    fn next(&mut self) -> Result<Value> {
        let mut bytes = vec![];
        (&mut self.inner)
            .take(4 * 1024 * 1024 + 1)
            .read_until(b'\n', &mut bytes)?;
        if bytes.is_empty() || bytes.len() > 4 * 1024 * 1024 || bytes.last() != Some(&b'\n') {
            return Err("raw audit encountered missing, oversized or incomplete record".into());
        }
        self.lines += 1;
        Ok(serde_json::from_slice(&bytes)?)
    }
    fn positioned(&mut self, phase: &str, split: &str, row: usize) -> Result<Value> {
        let record = self.next()?;
        if record["phase"] != phase || record["split"] != split || record["row"] != row {
            return Err("raw phase, split or original row ordering differs".into());
        }
        Ok(record)
    }
}
fn number(v: &Value) -> Result<usize> {
    Ok(v.as_u64().ok_or("missing raw integer")? as usize)
}
fn close(a: f64, b: f64) -> Result<()> {
    if !a.is_finite() || !b.is_finite() || (a - b).abs() > 1e-10 {
        return Err("raw floating measurement differs".into());
    }
    Ok(())
}
fn ce(logits: &[f64], label: usize) -> f64 {
    let peak = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exponential_sum = 0.;
    for &z in logits {
        exponential_sum += (z - peak).exp();
    }
    peak + exponential_sum.ln() - logits[label]
}
fn query(
    record: &Value,
    model: &certified::model::ModelEvaluator<'_>,
    prefix: &[i64],
    numerical: &mut evaluation::NumericalCounts,
) -> Result<(Vec<f64>, usize, bool, f64)> {
    let (expected, winner, residual, certificate) = evaluation::inspect(model, prefix, numerical)?;
    let logits: Vec<f64> = serde_json::from_value(record["query"]["logits"].clone())?;
    if logits.len() != expected.logits.len()
        || logits.is_empty()
        || logits.iter().any(|x| !x.is_finite())
        || record["query"] != serde_json::to_value(&expected)?
        || record["numerical"] != certificate
    {
        return Err("raw vector omits classes or contains nonfinite logits".into());
    }
    let mut predicted = 0;
    for i in 0..logits.len() {
        if logits[i] > logits[predicted] {
            predicted = i;
        }
    }
    if residual > 1e-7
        || predicted != winner
        || record["predicted"] != predicted
        || record["query"]["target_realizability_certified"] != false
    {
        return Err(
            "raw winner, independent complete products or claimed identification differs".into(),
        );
    }
    close(
        record["independent_residual"]
            .as_f64()
            .ok_or("missing independent residual")?,
        residual,
    )?;
    Ok((
        logits,
        predicted,
        !record["query"]["active_observable_coverage"]
            .as_bool()
            .ok_or("missing coverage diagnostic")?,
        residual,
    ))
}

#[derive(Default)]
struct Counts {
    targets: usize,
    correct: usize,
    exact: usize,
    transitions: usize,
    transition_correct: usize,
    teacher_targets: usize,
    teacher_correct: usize,
    teacher_exact: usize,
    loss: f64,
    ended: usize,
    extra: usize,
    generated: usize,
    unseen: usize,
    unseen_correct: usize,
    uncovered: usize,
    residual: f64,
}
fn check_counts(c: &Counts, examples: usize, generative: bool, saved: &Value) -> Result<()> {
    let m = &saved["metrics"];
    for (key, value) in [
        ("examples", examples),
        ("targets", c.targets),
        ("correct", c.correct),
        ("exact", c.exact),
        ("transition_targets", c.transitions),
        ("transition_correct", c.transition_correct),
        ("generated_tokens", c.generated),
        ("extra_tokens", c.extra),
    ] {
        if m[key] != value {
            return Err(format!("raw {key} recount differs").into());
        }
    }
    for (key, value) in [
        ("token_accuracy", c.correct as f64 / c.targets as f64),
        ("sequence_accuracy", c.exact as f64 / examples as f64),
        ("final_answer_accuracy", c.exact as f64 / examples as f64),
    ] {
        close(m[key].as_f64().ok_or("missing score ratio")?, value)?;
    }
    if c.transitions > 0 {
        close(
            m["transition_accuracy"]
                .as_f64()
                .ok_or("missing transition accuracy")?,
            c.transition_correct as f64 / c.transitions as f64,
        )?;
    } else if !m["transition_accuracy"].is_null() {
        return Err("empty transition denominator credited".into());
    }
    let t = &m["teacher_forced"];
    for (key, value) in [
        ("targets", c.teacher_targets),
        ("correct", c.teacher_correct),
        ("exact", c.teacher_exact),
    ] {
        if t[key] != value {
            return Err("teacher-forced raw recount differs".into());
        }
    }
    close(
        t["loss"].as_f64().ok_or("missing ordinary CE")?,
        c.loss / c.teacher_targets as f64,
    )?;
    close(
        t["token_accuracy"]
            .as_f64()
            .ok_or("missing teacher token ratio")?,
        c.teacher_correct as f64 / c.teacher_targets as f64,
    )?;
    close(
        t["sequence_accuracy"]
            .as_f64()
            .ok_or("missing teacher exact ratio")?,
        c.teacher_exact as f64 / examples as f64,
    )?;
    if generative {
        close(
            m["eos_rate"].as_f64().ok_or("missing EOS rate")?,
            c.ended as f64 / examples as f64,
        )?;
    } else if !m["eos_rate"].is_null() {
        return Err("prefix task received a generated EOS score".into());
    }
    if saved["unseen_label_targets"] != c.unseen
        || saved["unseen_label_correct"] != c.unseen_correct
        || saved["uncovered_scored_queries"] != c.uncovered
    {
        return Err("coverage or unseen-label recount differs".into());
    }
    close(
        saved["independent_complex_max_residual"]
            .as_f64()
            .ok_or("missing maximum residual")?,
        c.residual,
    )
}

pub fn verify(
    model: &Model,
    input: &Input,
    splits: &BTreeMap<String, Vec<Example>>,
    path: &Path,
    saved: &Value,
) -> Result<Value> {
    if let Some(rows) = splits.get("train")
        && adapter::input(input.vocabulary, rows)? != *input
    {
        return Err("raw readback source training pool changed".into());
    }
    if saved["raw_sha256"] != artifacts::digest(path)? {
        return Err("raw readback file digest differs".into());
    }
    let evaluator = certified::model::ModelEvaluator::new(model)?;
    let mut numerical = evaluation::NumericalCounts::default();
    let mut reader = Reader {
        inner: BufReader::with_capacity(1024 * 1024, File::open(path)?),
        lines: 0,
    };
    if saved["splits"]
        .as_object()
        .ok_or("missing original split set")?
        .keys()
        .collect::<Vec<_>>()
        != splits.keys().collect::<Vec<_>>()
    {
        return Err("raw original split set differs".into());
    }
    let seen: BTreeSet<_> = input.observations.iter().map(|r| r.label).collect();
    let (mut train_id, mut original_queries) = (0usize, 0u64);
    for (name, rows) in splits {
        let mut total = Counts::default();
        let generative = !rows[0].prompt.is_empty();
        for (row, e) in rows.iter().enumerate() {
            let mut predictions = vec![0; e.tokens.len()];
            let mut all = true;
            for (position, &label) in e.targets.iter().enumerate().filter(|(_, y)| **y >= 0) {
                let record = reader.positioned("teacher", name, row)?;
                if record["position"] != position || record["target"] != label {
                    return Err("raw original teacher target differs".into());
                }
                if record["tokens"] != json!(&e.tokens[..=position]) {
                    return Err("raw causal teacher prefix differs".into());
                }
                let (z, p, uncovered, residual) =
                    query(&record, &evaluator, &e.tokens[..=position], &mut numerical)?;
                if name == "train" {
                    train_id += 1;
                }
                let loss = ce(&z, label as usize);
                close(record["ce"].as_f64().ok_or("missing raw CE")?, loss)?;
                if record["predicted"] != p {
                    return Err("raw winner changed".into());
                }
                let hit = p == label as usize;
                predictions[position] = p;
                all &= hit;
                total.teacher_targets += 1;
                total.teacher_correct += usize::from(hit);
                total.loss += loss;
                total.residual = total.residual.max(residual);
                original_queries += 1;
                if !generative {
                    total.uncovered += usize::from(uncovered);
                    if !seen.contains(&(label as usize)) {
                        total.unseen += 1;
                        total.unseen_correct += usize::from(hit);
                    }
                }
            }
            total.teacher_exact += usize::from(all);
            let mut generated = vec![];
            if generative {
                let mut prefix = e.prompt.clone();
                for step in 0..e.limit {
                    let record = reader.positioned("generation", name, row)?;
                    if record["step"] != step || record["tokens"] != json!(prefix) {
                        return Err("own-output generation prefix or step differs".into());
                    }
                    let (_, p, uncovered, residual) =
                        query(&record, &evaluator, &prefix, &mut numerical)?;
                    generated.push(p);
                    total.uncovered += usize::from(uncovered);
                    total.residual = total.residual.max(residual);
                    original_queries += 1;
                    if p == 17 {
                        break;
                    }
                    prefix.push(p as i64);
                }
            }
            let result = reader.positioned("result", name, row)?;
            let mut targets = 0;
            let mut correct = 0;
            let mut transitions = 0;
            let mut transition_correct = 0;
            let mut previous = None;
            let source: Vec<(usize, i64)> = if generative {
                e.answer.iter().copied().enumerate().collect()
            } else {
                e.targets
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, y)| *y >= 0)
                    .collect()
            };
            for (position, y) in source {
                let hit = if generative {
                    generated.get(position).copied() == Some(y as usize)
                } else {
                    predictions[position] == y as usize
                };
                targets += 1;
                correct += usize::from(hit);
                if previous != Some(y) {
                    transitions += 1;
                    transition_correct += usize::from(hit);
                }
                previous = Some(y);
                if generative && !seen.contains(&(y as usize)) {
                    total.unseen += 1;
                    total.unseen_correct += usize::from(hit);
                }
            }
            let exact = correct == targets && (!generative || generated.len() == e.answer.len());
            for (key, value) in [
                ("targets", targets),
                ("correct", correct),
                ("transition_targets", transitions),
                ("transition_correct", transition_correct),
            ] {
                if number(&result["score"][key])? != value {
                    return Err("original row score differs from scalar recount".into());
                }
            }
            if result["score"]["exact"] != exact {
                return Err("original exact-row outcome differs".into());
            }
            if generative {
                let ended = generated.last() == Some(&17);
                let extra = generated.len().saturating_sub(e.answer.len());
                if result["prompt"] != json!(e.prompt)
                    || result["answer"] != json!(e.answer)
                    || result["generated"] != json!(generated)
                    || result["score"]["ended"] != ended
                    || result["score"]["extra_tokens"] != extra
                    || result["score"]["generated_tokens"] != generated.len()
                {
                    return Err("complete generated trajectory differs".into());
                }
                total.ended += usize::from(ended);
                total.extra += extra;
                total.generated += generated.len();
            } else if result["predictions"] != json!(predictions) {
                return Err("complete prefix decisions differ".into());
            }
            total.targets += targets;
            total.correct += correct;
            total.exact += usize::from(exact);
            total.transitions += transitions;
            total.transition_correct += transition_correct;
        }
        check_counts(&total, rows.len(), generative, &saved["splits"][name])?;
    }
    let mut extra = [0u8; 1];
    if reader.inner.read(&mut extra)? != 0
        || (splits.contains_key("train") && train_id != input.observations.len())
        || saved["actual_audit_calls"] != original_queries
        || saved["numerical"] != serde_json::to_value(&numerical)?
    {
        return Err("complete raw audit counts or trailing data differ".into());
    }
    Ok(
        json!({"raw_lines":reader.lines,"original_query_positions":original_queries,
        "independent_product_queries":original_queries,"original_training_references":train_id,
        "all_numerical_certificates_recomputed":true,
        "all_source_rows_recounted":true,"all_reported_integer_decisions_checked":true}),
    )
}
