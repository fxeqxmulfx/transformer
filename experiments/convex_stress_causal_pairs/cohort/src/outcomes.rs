use super::*;
fn offer(task: &str, e: &binding::Endpoint, report_sha256: &str) -> Result<comparison::Offer> {
    let t = e.training.as_ref().ok_or("offered mode lacks training")?;
    let audit = e.audit.as_ref().ok_or("offered mode lacks audit")?;
    if e.error.is_some() || t.policy != e.policy || (t.final_ce - audit.train_ce).abs() > 1e-8 {
        return Err("offered endpoint audit or training differs".into());
    }
    Ok(comparison::Offer {
        task: task.into(),
        mode: e.policy.mode,
        train_ce: audit.train_ce,
        model_sha256: hash(&serde_json::to_vec(&t.model)?),
        report_sha256: report_sha256.into(),
    })
}
pub(crate) fn offers(
    task: &str,
    a: &execute::Attempt,
    sha: &str,
) -> Result<Vec<comparison::Offer>> {
    let Some(report) = &a.report else {
        return Ok(vec![]);
    };
    report
        .endpoints
        .iter()
        .filter(|e| e.error.is_none())
        .map(|e| offer(task, e, sha))
        .collect()
}
struct Tracked<'a> {
    model: &'a component::model::Model,
    known: std::cell::RefCell<Vec<bool>>,
}
impl metrics::Predictor for Tracked<'_> {
    fn vocab(&self) -> usize {
        self.model.vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let tokens: Vec<_> = prefix.iter().map(|&v| v as usize).collect();
        self.known.borrow_mut().push(self.model.known(&tokens));
        self.model.logits_tokens(&tokens)
    }
}
fn forecast(task: &data::Task, model: &component::model::Model) -> Result<u64> {
    let mut calls = 0u64;
    let mut prefix_tokens = 0u64;
    for examples in task.splits.values() {
        if examples.is_empty() {
            return Err("empty original metric split".into());
        }
        for e in examples {
            if e.tokens.is_empty()
                || e.tokens.len() > 4096
                || e.prompt.len() + e.limit > 4096
                || e.tokens.len() != e.targets.len()
                || e.changes.len() != e.targets.len()
                || e.tokens
                    .iter()
                    .chain(&e.prompt)
                    .any(|&x| x < 0 || x as usize >= task.vocab)
            {
                return Err("original evaluation exceeds prediction domain".into());
            }
            for (i, _) in e.targets.iter().enumerate().filter(|(_, y)| **y >= 0) {
                calls += 1;
                prefix_tokens += (i + 1) as u64;
            }
            if e.field.is_some() {
                calls += 1;
                prefix_tokens += e.tokens.len() as u64;
            }
            if !e.prompt.is_empty() {
                calls += e.limit as u64;
                prefix_tokens += (e.limit * (e.prompt.len() + e.limit)) as u64;
            }
        }
    }
    let states = model
        .blocks
        .iter()
        .map(|b| b.states())
        .max()
        .ok_or("empty complex model")? as u64;
    let work = calls * 8 * states * model.vocabulary as u64 + 16 * prefix_tokens;
    if work > 2_000_000_000 {
        return Err("original metrics exceed 2B scalar-work forecast".into());
    }
    Ok(work)
}
pub(crate) fn evaluate(
    task: &str,
    x: &inputs::Inputs,
    a: &execute::Attempt,
    sha: &str,
) -> Result<Value> {
    let Some(report) = &a.report else {
        return Ok(json!([]));
    };
    let mut rows = vec![];
    for e in &report.endpoints {
        if e.error.is_some() {
            rows.push(
                json!({"task":task,"training_mode":mode(e.policy.mode),"status":"refused",
                "error":e.error,"training_retained":e.training.is_some()}),
            );
            continue;
        }
        let o = offer(task, e, sha)?;
        let model = &e.training.as_ref().unwrap().model;
        model.validate()?;
        let forecast = forecast(&x.task, model)?;
        let (mut actual, mut masks) = (serde_json::Map::new(), serde_json::Map::new());
        for (split, examples) in &x.task.splits {
            let tracked = Tracked {
                model,
                known: std::cell::RefCell::new(vec![]),
            };
            actual.insert(split.clone(), metrics::evaluate(&tracked, examples)?);
            let known = tracked.known.into_inner();
            masks.insert(
                split.clone(),
                json!({"calls":known.len(),"known_calls":known.iter().filter(|&&k| k).count(),
                "unknown_calls":known.iter().filter(|&&k| !k).count(),"known_mask":known,
                "order":"original teacher-forced and greedy prediction call order"}),
            );
        }
        let actual = Value::Object(actual);
        let loss = actual["train"]["loss"]
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("missing original CE")?;
        if (loss - o.train_ce).abs() > 1e-8 {
            return Err("ordinary CE differs from original evaluator".into());
        }
        let (primary, all) = passes(&actual, &x.task.selection)?;
        rows.push(json!({"task":task,"training_mode":mode(o.mode),"status":"evaluated","reference":o.reference(),
            "model_sha256":o.model_sha256,"report_sha256":sha,"train_ce":o.train_ce,
            "primary_split":x.task.selection,"metrics":actual,"primary_pass":primary,"all_splits_pass":all,
            "unknown_prefixes":masks,"metric_scalar_work_forecast":forecast,"audit":e.audit,
            "unseen_identification":false}));
    }
    Ok(json!(rows))
}
pub(crate) fn check(
    task: &str,
    input: &Value,
    a: &execute::Attempt,
    sha: &str,
    evaluation: &Value,
) -> Result<()> {
    let rows = evaluation.as_array().ok_or("missing actual evaluations")?;
    let endpoints = a
        .report
        .as_ref()
        .map_or(&[][..], |r| r.endpoints.as_slice());
    if rows.len() != endpoints.len() {
        return Err("constructed endpoint evaluation omitted".into());
    }
    for (row, e) in rows.iter().zip(endpoints) {
        if row["task"] != task || row["training_mode"] != mode(e.policy.mode) {
            return Err("evaluated task/mode changed".into());
        }
        if e.error.is_some() {
            if row["status"] != "refused"
                || row["error"] != json!(e.error)
                || row["training_retained"] != e.training.is_some()
            {
                return Err("mode refusal changed".into());
            }
            continue;
        }
        let o = offer(task, e, sha)?;
        let (primary, all) = passes(
            &row["metrics"],
            input["primary_split"]
                .as_str()
                .ok_or("missing primary split")?,
        )?;
        if row["status"] != "evaluated"
            || row["reference"] != o.reference()
            || row["model_sha256"] != o.model_sha256
            || row["report_sha256"] != sha
            || row["train_ce"] != o.train_ce
            || row["audit"] != json!(e.audit)
            || row["primary_split"] != input["primary_split"]
            || row["primary_pass"] != primary
            || row["all_splits_pass"] != all
            || row["unseen_identification"] != false
            || row["metrics"]["train"]["loss"]
                .as_f64()
                .is_none_or(|v| !v.is_finite() || (v - o.train_ce).abs() > 1e-8)
            || row["metric_scalar_work_forecast"]
                .as_u64()
                .is_none_or(|w| w > 2_000_000_000)
        {
            return Err("actual saved head, CE or original metric binding changed".into());
        }
        let masks = row["unknown_prefixes"]
            .as_object()
            .ok_or("missing unknown-prefix masks")?;
        let metrics = row["metrics"].as_object().ok_or("missing split metrics")?;
        if masks.len() != metrics.len() || metrics.keys().any(|s| !masks.contains_key(s)) {
            return Err("prediction-mask split scope changed".into());
        }
        for m in masks.values() {
            let values: Vec<bool> = serde_json::from_value(m["known_mask"].clone())?;
            if m["calls"] != values.len()
                || m["known_calls"] != values.iter().filter(|&&k| k).count()
                || m["unknown_calls"] != values.iter().filter(|&&k| !k).count()
            {
                return Err("unknown-mask counts changed".into());
            }
        }
    }
    Ok(())
}
