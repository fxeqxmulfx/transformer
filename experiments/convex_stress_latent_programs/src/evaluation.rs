//! All original splits, actual complete complex products and independent
//! causal interpretation. Consensus is conditional, never a target oracle.
use crate::{model::Model, *};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

pub fn evaluate(
    model: &Model,
    input: &Input,
    splits: &BTreeMap<String, Vec<data::Example>>,
) -> Result<Value> {
    model.validate()?;
    model.encode(input)?;
    let observed: BTreeSet<_> = input.observations.iter().map(|o| o.label).collect();
    let mut targets = 0u64;
    let mut length = 0usize;
    for rows in splits.values() {
        let request = sequences::adapter::request(model.head.vocabulary, rows)?;
        targets += request.discovery.input.observations.len() as u64;
        length = length.max(request.discovery.input.context);
    }
    let n = model.head.vocabulary as u64;
    let k = model.programs.len() as u64;
    let s = STATES as u64;
    let interpret_work: u64 = model
        .programs
        .iter()
        .map(|p| {
            let linear = 64 * length as u64;
            linear
                + match p {
                    program::Program::Signed { .. } => 16 * (length as u64).pow(2),
                    program::Program::Runs { .. } => 0,
                }
        })
        .sum();
    // Two producer passes, one independent full-product table, three program
    // passes, all raw classes and prediction serialization are charged.
    let work = 16 * k * s * s * n + targets * (24 * k * s * n + interpret_work + 128 * n);
    if work > WORK_LIMIT {
        return Err("all-split latent scoring exceeds16B scalar-work forecast".into());
    }
    let table = model.head.independent_table();
    let mut out = BTreeMap::new();
    for (name, rows) in splits {
        let metric = metrics::evaluate(model, rows)?;
        let (mut consensus, mut missing, mut unseen, mut unseen_correct) = (0, 0, 0, 0);
        let (mut correct, mut exact, mut ties) = (0, 0, 0);
        let mut max_difference = 0.0f64;
        let mut loss = 0.;
        let mut predictions = vec![];
        for (row_index, row) in rows.iter().enumerate() {
            let mut row_exact = true;
            for (position, &label) in row.targets.iter().enumerate().filter(|(_, y)| **y >= 0) {
                let prefix: Vec<_> = row.tokens[..=position]
                    .iter()
                    .map(|&t| t as usize)
                    .collect();
                let values = model.states(&prefix);
                let states: Vec<_> = values.iter().map(|v| v.state).collect();
                let actual = model.logits_states(&states);
                let independent: Vec<_> = model
                    .programs
                    .iter()
                    .map(|p| p.interpret(&prefix))
                    .collect();
                if independent.iter().zip(&values).any(|(&s, v)| s != v.state) {
                    return Err("independent causal program interpretation differs".into());
                }
                let checked: Vec<f64> = (0..model.head.vocabulary)
                    .map(|c| {
                        model
                            .weights
                            .iter()
                            .zip(&independent)
                            .enumerate()
                            .map(|(j, (&w, &s))| w * table[j][s][c])
                            .sum()
                    })
                    .collect();
                if actual.iter().chain(&checked).any(|z| !z.is_finite()) {
                    return Err("nonfinite complete complex prediction".into());
                }
                for (&a, &b) in actual.iter().zip(&checked) {
                    max_difference = max_difference.max((a - b).abs());
                }
                let prediction = argmax(&actual);
                let runner_up = actual
                    .iter()
                    .enumerate()
                    .filter(|(c, _)| *c != prediction)
                    .map(|(_, &v)| v)
                    .fold(f64::NEG_INFINITY, f64::max);
                let margin = actual[prediction] - runner_up;
                if max_difference > 1e-8 || (margin > 1e-7 && prediction != argmax(&checked)) {
                    return Err(
                        "independent complete complex logits or robust argmax differ".into(),
                    );
                }
                ties += usize::from(margin <= 1e-7);
                let labels = model.labels(&states);
                let absent = labels.iter().any(Option::is_none);
                let agreed = !absent && labels.iter().all(|y| *y == labels[0]);
                let unknown = !observed.contains(&(label as usize));
                let hit = prediction == label as usize;
                consensus += usize::from(agreed);
                missing += usize::from(absent);
                unseen += usize::from(unknown);
                unseen_correct += usize::from(unknown && hit);
                correct += usize::from(hit);
                row_exact &= hit;
                let raw_ce = ce(&actual, label as usize);
                loss += raw_ce;
                predictions.push(json!({"row":row_index,"position":position,"label":label,
                    "prediction":prediction,"ce":raw_ce,"margin":margin,
                    "consensus":if agreed { labels[0] } else { None },"any_missing":absent,
                    "label_unseen_in_training":unknown}));
            }
            exact += usize::from(row_exact);
        }
        if metric["correct"] != correct
            || metric["exact"] != exact
            || metric["targets"] != predictions.len()
            || (metric["loss"].as_f64().ok_or("missing raw CE")? - loss / predictions.len() as f64)
                .abs()
                > 1e-9
        {
            return Err("independent prediction trace and original metrics differ".into());
        }
        out.insert(name, json!({"goals":sequences::adapter::goals(&metric),"metrics":metric,
            "consensus_targets":consensus,"ambiguous_targets":predictions.len()-consensus,
            "any_missing_targets":missing,"unseen_label_targets":unseen,"unseen_label_correct":unseen_correct,
            "roundoff_sensitive_targets":ties,"independent_complex_max_difference":max_difference,
            "predictions":predictions}));
    }
    Ok(
        json!({"splits":out,"scalar_work_forecast":work,"target_realizability_certified":false,
        "complete_head_products":2 * targets * k * s * n,"independent_cache_products":k * s * s * n,
        "decoder_unknown_default":0,"convex_slice":"affine masses of frozen training-learned dictionaries"}),
    )
}
