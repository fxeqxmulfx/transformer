//! Original masks and own-output generation, with every actual all-class
//! vector streamed to disk. The independent interpreter/product is checked at
//! every supervised and generated prefix. Mode's final answer is diagnostic.
use crate::{model::Model, *};
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
    pub raw_bytes: u64,
}

pub fn forecast(v: usize, splits: &BTreeMap<String, Vec<data::Example>>) -> Result<Forecast> {
    if !(2..=1024).contains(&v) || splits.is_empty() || splits.len() > 64 {
        return Err("evaluation dimensions refused".into());
    }
    let (mut work, mut calls, mut bytes) = (0u128, 0u128, 0u128);
    for rows in splits.values() {
        if rows.is_empty() || rows.len() > 20_000 {
            return Err("evaluation rows refused".into());
        }
        for e in rows {
            let horizon = e
                .prompt
                .len()
                .checked_add(e.limit)
                .ok_or("horizon overflow")?;
            if e.prompt.is_empty()
                || horizon > 4096
                || e.answer.is_empty()
                || e.answer.len() > e.limit
            {
                return Err("generation horizon or original layout refused".into());
            }
            let count = (e.answer.len() + e.limit) as u128;
            let h = horizon as u128;
            // The inherited metric evaluator and the independent audit both
            // run full generation and all teacher-forced prefixes.
            calls += 2 * count;
            work += count
                * (512 * (h * h + h + v as u128 + 1) * (1 + horizon.ilog2() as u128)
                    + 512 * v as u128);
            bytes += count * (4096 + 32 * (h + v as u128));
        }
    }
    if work > 4_000_000_000_000 || bytes > 256u128 * 1024 * 1024 * 1024 {
        return Err("complete generation work or disk forecast refused".into());
    }
    Ok(Forecast {
        scalar_work: work as u64,
        calls: calls as u64,
        raw_bytes: bytes as u64,
    })
}

fn inspect(model: &Model, prefix: &[i64]) -> Result<(Vec<f64>, usize, bool, f64)> {
    let tokens: Vec<_> = prefix
        .iter()
        .map(|&t| usize::try_from(t))
        .collect::<std::result::Result<_, _>>()?;
    let atom = model.learned.program.run(&tokens).atom;
    if atom != model.learned.program.interpret(&tokens) {
        return Err("independent latent program differs on a scored prefix".into());
    }
    let (x, identified) = model.feature(&tokens);
    let z = model.head.logits(x);
    let independent = model.independent_logits(&tokens);
    if z.iter().chain(&independent).any(|x| !x.is_finite()) {
        return Err("nonfinite complete score".into());
    }
    let residual = z
        .iter()
        .zip(&independent)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let predicted = argmax(&z);
    if residual > 1e-7 || predicted != argmax(&independent) {
        return Err("independent complex products or raw winners differ".into());
    }
    Ok((z, predicted, identified, residual))
}

fn line(writer: &mut BufWriter<std::fs::File>, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

pub fn evaluate(
    model: &Model,
    input: &Input,
    splits: &BTreeMap<String, Vec<data::Example>>,
    final_token: bool,
    path: &Path,
) -> Result<Value> {
    model.head.validate()?;
    let bound = forecast(input.vocabulary, splits)?;
    let observed: BTreeSet<_> = input.observations.iter().map(|r| r.label).collect();
    let mut raw = BufWriter::with_capacity(
        1024 * 1024,
        OpenOptions::new().write(true).create_new(true).open(path)?,
    );
    let mut out = BTreeMap::new();
    let mut audit_calls = 0u64;
    for (name, rows) in splits {
        parent::dataset::input(input.vocabulary, rows)?;
        let mut metric = metrics::evaluate(model, rows)?;
        let (mut targets, mut correct, mut exact, mut ended, mut extra, mut generated_tokens) =
            (0, 0, 0, 0, 0, 0);
        let (mut teacher_correct, mut teacher_exact, mut teacher_loss) = (0, 0, 0.);
        let (mut missing, mut final_correct, mut unseen, mut unseen_correct) = (0, 0, 0, 0);
        let mut max_residual = 0.0f64;
        for (row_id, e) in rows.iter().enumerate() {
            let mut all_correct = true;
            for (position, &label) in e.targets.iter().enumerate().filter(|(_, y)| **y >= 0) {
                let prefix = &e.tokens[..=position];
                let (z, predicted, identified, residual) = inspect(model, prefix)?;
                let loss = ce(&z, label as usize);
                teacher_correct += usize::from(predicted == label as usize);
                all_correct &= predicted == label as usize;
                teacher_loss += loss;
                audit_calls += 1;
                max_residual = max_residual.max(residual);
                line(
                    &mut raw,
                    &json!({"split":name,"row":row_id,"phase":"teacher","position":position,
                    "tokens":prefix,"target":label,"predicted":predicted,"logits":z,"ce":loss,
                    "decoder_identified":identified,"independent_residual":residual}),
                )?;
            }
            teacher_exact += usize::from(all_correct);
            let mut step = 0;
            let generated = frequency::scoring::rollout(
                &e.prompt,
                e.limit,
                input.vocabulary,
                |prefix| {
                    let (z, predicted, identified, residual) = inspect(model, prefix)?;
                    audit_calls += 1;
                    missing += usize::from(!identified);
                    max_residual = max_residual.max(residual);
                    line(
                        &mut raw,
                        &json!({"split":name,"row":row_id,"phase":"generation","step":step,
                    "tokens":prefix,"predicted":predicted,"logits":z,"decoder_identified":identified,
                    "independent_residual":residual}),
                    )?;
                    step += 1;
                    Ok(predicted)
                },
            )?;
            let scored = frequency::scoring::score(&e.answer, &generated)?;
            let final_hit = frequency::scoring::final_answer(&e.answer, &generated, final_token)?;
            targets += scored.targets;
            correct += scored.correct;
            exact += usize::from(scored.exact);
            ended += usize::from(scored.ended);
            extra += scored.extra_tokens;
            generated_tokens += scored.generated_tokens;
            final_correct += usize::from(final_hit);
            for (i, &label) in e.answer.iter().enumerate() {
                if !observed.contains(&(label as usize)) {
                    unseen += 1;
                    unseen_correct += usize::from(generated.get(i) == Some(&(label as usize)));
                }
            }
            line(
                &mut raw,
                &json!({"split":name,"row":row_id,"phase":"result","prompt":e.prompt,
                "answer":e.answer,"generated":generated,"score":scored,"final_answer_correct":final_hit}),
            )?;
        }
        if metric["targets"] != targets
            || metric["correct"] != correct
            || metric["exact"] != exact
            || metric["generated_tokens"] != generated_tokens
            || metric["teacher_forced"]["correct"] != teacher_correct
            || metric["teacher_forced"]["exact"] != teacher_exact
            || (metric["teacher_forced"]["loss"]
                .as_f64()
                .ok_or("missing teacher CE")?
                - teacher_loss / targets as f64)
                .abs()
                > 1e-8
            || (metric["eos_rate"].as_f64().ok_or("missing EOS rate")?
                - ended as f64 / rows.len() as f64)
                .abs()
                > 1e-12
        {
            return Err("original metric evaluator differs from independent scoring".into());
        }
        // The inherited helper only recognizes earlier final-token task names.
        // Preserve the original Mode diagnostic without changing exact success.
        metric["final_answer_accuracy"] = json!(final_correct as f64 / rows.len() as f64);
        metric["extra_tokens"] = json!(extra);
        out.insert(name,json!({"goals":parent::sequences::adapter::goals(&metric),"metrics":metric,
            "unseen_label_targets":unseen,"unseen_label_correct":unseen_correct,
            "generated_unidentified_queries":missing,"independent_complex_max_residual":max_residual}));
    }
    raw.flush()?;
    raw.get_ref().sync_all()?;
    if 2 * audit_calls > bound.calls || std::fs::metadata(path)?.len() > bound.raw_bytes {
        return Err("actual evaluation calls or disk output exceeded forecast".into());
    }
    Ok(
        json!({"splits":out,"forecast":bound,"actual_audit_calls":audit_calls,
        "raw_file":path.file_name().and_then(|s|s.to_str()).ok_or("invalid raw filename")?,
        "raw_sha256":parent::sequences::runner::artifacts::digest(path)?,
        "target_realizability_certified":false,"ordinary_ce_is_teacher_forced":true}),
    )
}
