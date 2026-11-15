//! Actual greedy generation, plus independent prefix/rank and full-complex
//! replay. Teacher-forced CE and raw generation denominators stay separate.
use crate::{model::Model, *};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize)]
struct Prediction {
    token: usize,
    margin: f64,
    consensus: Option<usize>,
    missing: bool,
    difference: f64,
}
fn inspect(
    model: &Model,
    table: &[Vec<Vec<f64>>],
    prefix: &[i64],
) -> Result<(Prediction, Vec<f64>)> {
    let tokens: Vec<_> = prefix
        .iter()
        .map(|&t| usize::try_from(t))
        .collect::<std::result::Result<_, _>>()?;
    let values = model.states(&tokens);
    let states: Vec<_> = values.iter().map(|v| v.state).collect();
    let actual = model.logits_states(&states);
    let independent: Vec<_> = model
        .programs
        .iter()
        .map(|p| p.interpret(&tokens))
        .collect();
    if independent
        .iter()
        .zip(&values)
        .any(|(&(s, b), v)| s != v.state || b != v.buffer_symbols)
    {
        return Err("independent positional/rank interpreter differs".into());
    }
    let checked: Vec<f64> = (0..model.head.vocabulary)
        .map(|c| {
            model
                .weights
                .iter()
                .zip(&independent)
                .enumerate()
                .map(|(j, (&w, &(s, _)))| w * table[j][s][c])
                .sum()
        })
        .collect();
    if actual.iter().chain(&checked).any(|z| !z.is_finite()) {
        return Err("nonfinite complete complex generation".into());
    }
    let difference = actual
        .iter()
        .zip(&checked)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let token = argmax(&actual);
    let margin = actual[token]
        - actual
            .iter()
            .enumerate()
            .filter(|&(c, _)| c != token)
            .map(|(_, &z)| z)
            .fold(f64::NEG_INFINITY, f64::max);
    if difference > 1e-8 || (margin > 1e-7 && token != argmax(&checked)) {
        return Err("independent complete complex logits or robust argmax differ".into());
    }
    let labels = model.labels(&states);
    let missing = labels.iter().any(Option::is_none);
    let consensus = if !missing && labels.iter().all(|y| *y == labels[0]) {
        labels[0]
    } else {
        None
    };
    if consensus.is_some_and(|y| y != token) {
        return Err("accepted dictionary disagrees with complete complex argmax".into());
    }
    Ok((
        Prediction {
            token,
            margin,
            consensus,
            missing,
            difference,
        },
        actual,
    ))
}

pub const SCORING_WORK_LIMIT: u64 = 160_000_000_000;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Forecast {
    pub scalar_work_forecast: u64,
    pub maximum_prediction_calls: u64,
}

/// Dimensions and original prompt/answer lengths only. No label values,
/// logits, target oracle or parameter fitting enters this admission check.
pub fn forecast(
    programs: usize,
    vocabulary: usize,
    splits: &BTreeMap<String, Vec<data::Example>>,
) -> Result<Forecast> {
    if !(1..=128).contains(&programs)
        || !(2..=126).contains(&vocabulary)
        || splits.is_empty()
        || splits.len() > 64
    {
        return Err("invalid generation forecast dimensions".into());
    }
    let (k, s, c) = (programs as u64, STATES as u64, vocabulary as u64);
    let mut calls = 0u64;
    let mut program_work = 0u128;
    for rows in splits.values() {
        if rows.is_empty() || rows.len() > 20_000 {
            return Err("invalid generation forecast rows".into());
        }
        for e in rows {
            if e.prompt.is_empty()
                || e.prompt.len() + e.limit > 4096
                || e.limit == 0
                || e.answer.is_empty()
                || e.answer.len() > e.limit
            {
                return Err("generative scoring layout or horizon refused".into());
            }
            let horizon = (e.prompt.len() + e.limit) as u64;
            // Base metrics and independent audit each visit teacher-forced
            // positions and at most every generated position. No early-EOS
            // saving is assumed before scoring or candidate selection.
            let count = 2 * (e.answer.len() + e.limit) as u64;
            calls += count;
            program_work += u128::from(count)
                * u128::from(k)
                * u128::from(64 * horizon + 16 * horizon * horizon);
        }
    }
    let work = u128::from(16 * k * s * s * c)
        + u128::from(calls) * u128::from(24 * k * s * c + 128 * c)
        + program_work;
    Ok(Forecast {
        scalar_work_forecast: work.try_into()?,
        maximum_prediction_calls: calls,
    })
}

pub fn evaluate(
    model: &Model,
    input: &Input,
    splits: &BTreeMap<String, Vec<data::Example>>,
) -> Result<Value> {
    model.validate()?;
    model.encode(input)?;
    for rows in splits.values() {
        dataset::input(model.head.vocabulary, rows)?;
    }
    let observed: BTreeSet<_> = input.observations.iter().map(|o| o.label).collect();
    let f = forecast(model.programs.len(), model.head.vocabulary, splits)?;
    let (work, calls) = (f.scalar_work_forecast, f.maximum_prediction_calls);
    if work > SCORING_WORK_LIMIT {
        return Err(
            format!("all-split generation forecast {work} exceeds {SCORING_WORK_LIMIT}").into(),
        );
    }
    let table = model.head.independent_table();
    let mut out = BTreeMap::new();
    let mut actual_audit_calls = 0u64;
    for (name, rows) in splits {
        let mut metric = metrics::evaluate(model, rows)?;
        let (mut correct, mut targets, mut exact, mut ended, mut extra, mut generated_tokens) =
            (0, 0, 0, 0, 0, 0);
        let (mut forced_correct, mut forced_exact, mut forced_loss) = (0, 0, 0.);
        let (mut accepted, mut missing, mut ties) = (0, 0, 0);
        let (mut unseen, mut unseen_correct) = (0, 0);
        let mut max_difference = 0.0f64;
        let mut predictions = vec![];
        for (row_index, e) in rows.iter().enumerate() {
            let mut teacher_exact = true;
            for (position, &label) in e.targets.iter().enumerate().filter(|(_, y)| **y >= 0) {
                let prefix = &e.tokens[..=position];
                let (p, logits) = inspect(model, &table, prefix)?;
                actual_audit_calls += 1;
                let hit = p.token == label as usize;
                teacher_exact &= hit;
                forced_correct += usize::from(hit);
                forced_loss += ce(&logits, label as usize);
                max_difference = max_difference.max(p.difference);
            }
            forced_exact += usize::from(teacher_exact);
            let mut steps = vec![];
            let generated = generative::scoring::rollout(
                &e.prompt,
                e.limit,
                model.head.vocabulary,
                |prefix| {
                    let (p, _) = inspect(model, &table, prefix)?;
                    let token = p.token;
                    actual_audit_calls += 1;
                    accepted += usize::from(p.consensus.is_some());
                    missing += usize::from(p.missing);
                    ties += usize::from(p.margin <= 1e-7);
                    max_difference = max_difference.max(p.difference);
                    steps.push(p);
                    Ok(token)
                },
            )?;
            let score = generative::scoring::score(&e.answer, &generated)?;
            for (i, &y) in e.answer.iter().enumerate() {
                if !observed.contains(&(y as usize)) {
                    unseen += 1;
                    unseen_correct += usize::from(generated.get(i) == Some(&(y as usize)));
                }
            }
            correct += score.correct;
            targets += score.targets;
            exact += usize::from(score.exact);
            ended += usize::from(score.ended);
            extra += score.extra_tokens;
            generated_tokens += score.generated_tokens;
            predictions
                .push(json!({"row":row_index,"generated":generated,"score":score,"steps":steps}));
        }
        if metric["correct"] != correct
            || metric["targets"] != targets
            || metric["exact"] != exact
            || metric["generated_tokens"] != generated_tokens
            || metric["teacher_forced"]["correct"] != forced_correct
            || metric["teacher_forced"]["exact"] != forced_exact
            || (metric["teacher_forced"]["loss"]
                .as_f64()
                .ok_or("missing teacher CE")?
                - forced_loss / targets as f64)
                .abs()
                > 1e-9
            || (metric["eos_rate"].as_f64().ok_or("missing EOS rate")?
                - ended as f64 / rows.len() as f64)
                .abs()
                > 1e-12
        {
            return Err("independent generated/teacher-forced counts or CE differ".into());
        }
        // The inherited helper hardcodes this field; use the exact original
        // generation scorer. All six original limits equal answer length.
        metric["extra_tokens"] = json!(extra);
        out.insert(name, json!({"goals":sequences::adapter::goals(&metric),"metrics":metric,
            "generated_accepted_queries":accepted,"generated_missing_queries":missing,"generated_ambiguous_queries":generated_tokens-accepted,
            "unseen_label_targets":unseen,"unseen_label_correct":unseen_correct,
            "roundoff_sensitive_queries":ties,"independent_complex_max_difference":max_difference,"predictions":predictions}));
    }
    Ok(
        json!({"splits":out,"scalar_work_forecast":work,"maximum_prediction_calls":calls,
        "actual_audit_calls":actual_audit_calls,"target_realizability_certified":false,
        "ordinary_ce_is_teacher_forced":true,"decoder_unknown_default":"explicit raw-state identity completion within vocabulary, otherwise0; unknown dictionaries remain None",
        "convex_slice":"affine masses of frozen training-learned dictionaries"}),
    )
}
