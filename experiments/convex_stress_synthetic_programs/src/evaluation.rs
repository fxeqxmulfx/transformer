//! Complete all-class CE and original raw/exact scoring. Every teacher-forced
//! or own-output query is checked by a separate program interpreter and complex
//! multiplication. Target labels enter only loss/scoring after prediction.
use crate::{
    adapter::Input,
    model::{Model, Query},
    *,
};
use dataset::data::Example;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::Path,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Forecast {
    pub scalar_work: u64,
    pub calls: u64,
    pub disk_bytes: u64,
    pub buffer_bytes: u64,
}
pub fn forecast(v: usize, p: usize, splits: &BTreeMap<String, Vec<Example>>) -> Result<Forecast> {
    if !(2..=1024).contains(&v) || !(1..=64).contains(&p) || splits.is_empty() || splits.len() > 64
    {
        return Err("evaluation dimensions refused".into());
    }
    let (mut calls, mut work, mut bytes) = (0u128, 0u128, 0u128);
    for rows in splits.values() {
        if rows.is_empty() || rows.len() > 20000 {
            return Err("evaluation row count refused".into());
        }
        for e in rows {
            adapter::validate(e, v)?;
            let h = e.tokens.len().max(e.prompt.len() + e.limit) as u128;
            let n = e.targets.iter().filter(|&&y| y >= 0).count() as u128 + e.limit as u128;
            calls += n;
            // Four AST/render passes, primary and independent full products,
            // feature spans, CE and JSON construction; both interpreters are
            // linear in prefix length times the127-node grammar cap.
            work += n * p as u128 * (4096 * (h + 1) + 2048 * v as u128 + 8192);
            bytes += n * (16384 + 64 * (h + v as u128 + 4 * p as u128));
        }
    }
    let buffer = 256 * 1024u128.pow(2) + 64 * (4096 * 256 + p as u128 * v as u128 * 256);
    if work > 600_000_000_000_000 || bytes > 256 * 1024u128.pow(3) || buffer > joint::MAX_BYTES {
        return Err("evaluation work/storage forecast refused".into());
    }
    Ok(Forecast {
        scalar_work: work as u64,
        calls: calls as u64,
        disk_bytes: bytes as u64,
        buffer_bytes: buffer as u64,
    })
}
fn inspect(model: &Model, prefix: &[i64]) -> Result<(Query, usize, f64)> {
    let tokens = prefix
        .iter()
        .map(|&t| usize::try_from(t))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for p in &model.programs {
        if p.program.run(&tokens) != p.program.interpret(&tokens) {
            return Err("independent program interpreter differs".into());
        }
    }
    let q = model.query(&tokens)?;
    let independent = model.independent_logits(&tokens)?;
    if q.logits.iter().chain(&independent).any(|x| !x.is_finite())
        || q.logits.len() != independent.len()
    {
        return Err("nonfinite or incomplete class vector".into());
    }
    let residual = q
        .logits
        .iter()
        .zip(&independent)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let predicted = joint::argmax(&q.logits);
    if residual > 1e-7 || predicted != joint::argmax(&independent) {
        return Err("independent complete products or raw winners differ".into());
    }
    Ok((q, predicted, residual))
}
fn line(w: &mut BufWriter<std::fs::File>, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *w, value)?;
    w.write_all(b"\n")?;
    Ok(())
}

pub fn evaluate(
    model: &Model,
    input: &Input,
    splits: &BTreeMap<String, Vec<Example>>,
    path: &Path,
) -> Result<Value> {
    model.head.validate()?;
    if model.head.coordinates.len() != input.vocabulary {
        return Err("class vocabulary differs".into());
    }
    let bound = forecast(input.vocabulary, model.programs.len(), splits)?;
    let observed: BTreeSet<_> = input.observations.iter().map(|r| r.label).collect();
    let mut w = BufWriter::with_capacity(
        1024 * 1024,
        OpenOptions::new().write(true).create_new(true).open(path)?,
    );
    let mut out = BTreeMap::new();
    let mut calls = 0u64;
    for (name, rows) in splits {
        let generative = !rows[0].prompt.is_empty();
        if rows.iter().any(|e| e.prompt.is_empty() == generative) {
            return Err("mixed scoring layouts in split".into());
        }
        let (mut targets, mut correct, mut exact, mut transitions, mut changed_correct) =
            (0, 0, 0, 0, 0);
        let (mut teacher_targets, mut teacher_correct, mut teacher_exact, mut loss) = (0, 0, 0, 0.);
        let (mut ended, mut extra, mut generated_tokens, mut uncovered) = (0, 0, 0, 0);
        let (mut unseen, mut unseen_correct) = (0, 0);
        let mut residual = 0.0f64;
        for (row, e) in rows.iter().enumerate() {
            let mut predictions = vec![0; e.tokens.len()];
            let mut all = true;
            for (position, &label) in e.targets.iter().enumerate().filter(|(_, y)| **y >= 0) {
                let prefix = &e.tokens[..=position];
                let (q, pred, r) = inspect(model, prefix)?;
                let ce = joint::ce(&q.logits, label as usize);
                teacher_targets += 1;
                teacher_correct += usize::from(pred == label as usize);
                all &= pred == label as usize;
                loss += ce;
                calls += 1;
                residual = residual.max(r);
                predictions[position] = pred;
                if !generative {
                    uncovered += usize::from(!q.active_observable_coverage);
                    if !observed.contains(&(label as usize)) {
                        unseen += 1;
                        unseen_correct += usize::from(pred == label as usize);
                    }
                }
                line(
                    &mut w,
                    &json!({"split":name,"row":row,"phase":"teacher","position":position,
                    "tokens":prefix,"target":label,"predicted":pred,"ce":ce,"query":q,"independent_residual":r}),
                )?;
            }
            teacher_exact += usize::from(all);
            if generative {
                let mut step = 0;
                let generated = from_data(dataset::scoring::rollout(
                    &e.prompt,
                    e.limit,
                    input.vocabulary,
                    |prefix| {
                        // Convert the local checked error into the data scorer's
                        // thread-safe error type without dropping its message.
                        let (q, pred, r) = inspect(model, prefix).map_err(|e| e.to_string())?;
                        calls += 1;
                        residual = residual.max(r);
                        uncovered += usize::from(!q.active_observable_coverage);
                        line(
                            &mut w,
                            &json!({"split":name,"row":row,"phase":"generation","step":step,
                        "tokens":prefix,"predicted":pred,"query":q,"independent_residual":r}),
                        )
                        .map_err(|e| e.to_string())?;
                        step += 1;
                        Ok(pred)
                    },
                ))?;
                let s = from_data(dataset::scoring::score(&e.answer, &generated))?;
                // A separate scalar recount also covers wrong early EOS,
                // extra tokens and missing positions in an own-output rollout.
                let hits = e
                    .answer
                    .iter()
                    .enumerate()
                    .filter(|(i, y)| generated.get(*i).copied() == Some(**y as usize))
                    .count();
                if s.targets != e.answer.len()
                    || s.correct != hits
                    || s.exact != (hits == e.answer.len() && generated.len() == e.answer.len())
                {
                    return Err("original generation scorer differs from raw recount".into());
                }
                targets += s.targets;
                correct += s.correct;
                exact += usize::from(s.exact);
                transitions += s.transition_targets;
                changed_correct += s.transition_correct;
                ended += usize::from(s.ended);
                extra += s.extra_tokens;
                generated_tokens += s.generated_tokens;
                for (i, &y) in e.answer.iter().enumerate() {
                    if !observed.contains(&(y as usize)) {
                        unseen += 1;
                        unseen_correct += usize::from(generated.get(i) == Some(&(y as usize)));
                    }
                }
                line(
                    &mut w,
                    &json!({"split":name,"row":row,"phase":"result","prompt":e.prompt,
                    "answer":e.answer,"generated":generated,"score":s}),
                )?;
            } else {
                let s = from_data(dataset::scoring::prefix(e, &predictions))?;
                if s.exact != all || s.targets != e.targets.iter().filter(|&&y| y >= 0).count() {
                    return Err("prefix score differs".into());
                }
                targets += s.targets;
                correct += s.correct;
                exact += usize::from(s.exact);
                transitions += s.transition_targets;
                changed_correct += s.transition_correct;
                line(
                    &mut w,
                    &json!({"split":name,"row":row,"phase":"result","predictions":predictions,"score":s}),
                )?;
            }
        }
        if targets == 0 || targets != teacher_targets || !loss.is_finite() {
            return Err("empty or inconsistent score denominator".into());
        }
        let metric = json!({"examples":rows.len(),"targets":targets,"correct":correct,"exact":exact,
            "token_accuracy":correct as f64/targets as f64,"sequence_accuracy":exact as f64/rows.len() as f64,
            "final_answer_accuracy":exact as f64/rows.len() as f64,
            "transition_targets":transitions,"transition_correct":changed_correct,
            "transition_accuracy":if transitions>0 {Some(changed_correct as f64/transitions as f64)} else {None},
            "teacher_forced":{"targets":teacher_targets,"correct":teacher_correct,"exact":teacher_exact,"loss":loss/teacher_targets as f64,
                "token_accuracy":teacher_correct as f64/teacher_targets as f64,"sequence_accuracy":teacher_exact as f64/rows.len() as f64},
            "eos_rate":if generative {Some(ended as f64/rows.len() as f64)} else {None},
            "generated_tokens":generated_tokens,"extra_tokens":extra});
        out.insert(
            name,
            json!({"goals":dataset::sequences::adapter::goals(&metric),"metrics":metric,
            "unseen_label_targets":unseen,"unseen_label_correct":unseen_correct,
            "uncovered_scored_queries":uncovered,"independent_complex_max_residual":residual}),
        );
    }
    w.flush()?;
    w.get_ref().sync_all()?;
    if calls > bound.calls || std::fs::metadata(path)?.len() > bound.disk_bytes {
        return Err("actual evaluation exceeded forecast".into());
    }
    Ok(
        json!({"splits":out,"forecast":bound,"actual_audit_calls":calls,
        "raw_file":path.file_name().and_then(|s|s.to_str()).ok_or("invalid raw file name")?,
        "raw_sha256":artifacts::digest(path)?,"target_realizability_certified":false,"ordinary_ce_is_teacher_forced":true}),
    )
}
