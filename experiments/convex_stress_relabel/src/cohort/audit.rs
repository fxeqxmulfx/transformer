//! Raw saved-coordinate audit, following the CM10 auditor at 83382635.
//! CE, decoding and Fourier sums do not call producer implementations.
//! Frozen data construction and LAPACK SVD are shared; bounds are numerical.
use crate::{C, Model, Result, data, linalg};
use serde_json::{Value, json};

pub fn close(a: &Value, b: &Value, tolerance: f64) -> Result<()> {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) if a.len() == b.len() => {
            for (key, value) in a {
                close(value, b.get(key).ok_or("missing audit key")?, tolerance)?;
            }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            for (x, y) in a.iter().zip(b) {
                close(x, y, tolerance)?;
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (
                a.as_f64().ok_or("invalid audit number")?,
                b.as_f64().ok_or("invalid reported number")?,
            );
            if !a.is_finite() || !b.is_finite() || (a - b).abs() > tolerance {
                return Err(format!("audit number mismatch: {a} != {b}").into());
            }
        }
        _ if a == b => {}
        _ => return Err("audit structure mismatch".into()),
    }
    Ok(())
}

pub fn validate(model: &Model, task: &data::Task, report: &Value) -> Result<Value> {
    let n = task.vocab;
    let table = &model.table;
    let code = &model.coding.encode;
    let mut permutation = code.clone();
    permutation.sort_unstable();
    if !(2..=193).contains(&n)
        || table.vocab != n
        || permutation != (0..n).collect::<Vec<_>>()
        || table.probabilities.len() != n * n * n
        || table
            .probabilities
            .iter()
            .any(|x| !x.is_finite() || *x < -1e-10 || *x > 1. + 1e-10)
        || table
            .probabilities
            .chunks_exact(n)
            .any(|p| (p.iter().sum::<f64>() - 1.).abs() > 1e-10)
    {
        return Err("invalid raw table or class coding".into());
    }
    let budgets = match report["case"]["budget"].as_str() {
        Some("rank_one") => (n as f64, n as f64),
        Some("full_rank") => ((n as f64).sqrt(), n as f64 * (n as f64).sqrt()),
        _ => return Err("invalid raw budget".into()),
    };
    if (table.radius, table.nuclear_budget) != budgets {
        return Err("raw budget mismatch".into());
    }
    let mut diagnostics = serde_json::Map::new();
    for (split, examples) in &task.splits {
        let (mut ce, mut correct, mut transitions, mut transition_correct, mut fit) =
            (0., 0, 0, 0, 0_f64);
        let (mut counts, mut hits) = (vec![0; n], vec![0; n]);
        for e in examples {
            if e.tokens.len() != 2
                || e.targets.len() != 2
                || e.targets[0] != -100
                || e.targets[1] < 0
                || e.changes.len() != 2
                || !e.prompt.is_empty()
                || e.field.is_some()
                || e.tokens.iter().any(|t| *t < 0 || *t as usize >= n)
                || e.targets[1] as usize >= n
            {
                return Err("unsupported raw audit example".into());
            }
            let y = e.targets[1] as usize;
            let start = (e.tokens[1] as usize + n * e.tokens[0] as usize) * n;
            let probabilities = &table.probabilities[start..start + n];
            let mut best = 0;
            for c in 1..n {
                if probabilities[code[c]]
                    .total_cmp(&probabilities[code[best]])
                    .is_gt()
                {
                    best = c;
                }
            }
            let hit = usize::from(best == y);
            correct += hit;
            counts[y] += 1;
            hits[y] += hit;
            if e.changes[1] {
                transitions += 1;
                transition_correct += hit;
            }
            // Sum in original class order; keep every original softmax coordinate.
            ce += code
                .iter()
                .map(|&k| probabilities[k].exp())
                .sum::<f64>()
                .ln()
                - probabilities[code[y]];
            for c in 0..n {
                fit = fit.max((probabilities[code[c]] - f64::from(c == y)).abs());
            }
        }
        let count = examples.len();
        if count == 0 {
            return Err("empty raw audit split".into());
        }
        let accuracy = correct as f64 / count as f64;
        let classes: serde_json::Map<String,Value> = counts.iter().enumerate().filter(|(_, count)| **count > 0)
            .map(|(c, &count)| (c.to_string(), json!({"name":c.to_string(),"count":count,"correct":hits[c],"accuracy":hits[c] as f64/count as f64}))).collect();
        let balanced = counts
            .iter()
            .zip(&hits)
            .filter(|(count, _)| **count > 0)
            .map(|(&count, &hit)| hit as f64 / count as f64)
            .sum::<f64>()
            / classes.len() as f64;
        let metrics = json!({"loss":ce/count as f64,"example_loss":ce/count as f64,"sequence_accuracy":accuracy,"token_accuracy":accuracy,
            "balanced_accuracy":balanced,"transition_accuracy":if transitions > 0 { Some(transition_correct as f64/transitions as f64) } else { None },
            "correct":correct,"exact":correct,"targets":count,"target_count":count,"examples":count,"example_count":count,"transition_count":transitions,"classes":classes});
        close(&metrics, &report["metrics"][split], 1e-10)?;
        if split == "train" {
            close(&json!(fit), &report["fit_defect"], 1e-10)?;
            let ideal = (std::f64::consts::E + n as f64 - 1.).ln() - 1.;
            close(
                &json!(ce / count as f64 - ideal),
                &report["ideal_ce_excess"],
                1e-10,
            )?;
        }
        diagnostics.insert(
            split.clone(),
            json!({"examples":count,"correct":correct,"ce":ce/count as f64,"one_hot_defect":fit}),
        );
    }
    // One frequency matrix at a time: no all-frequency dense audit allocation.
    let mut norms = vec![];
    for f in 1..n {
        let codes: Vec<_> = (0..n)
            .map(|c| {
                let phase = std::f64::consts::TAU * (f * c) as f64 / n as f64;
                C::new(phase.cos(), phase.sin())
            })
            .collect();
        let mut matrix = vec![C::default(); n * n];
        for (cell, entry) in matrix.iter_mut().enumerate() {
            for (c, phase) in codes.iter().enumerate() {
                *entry += table.probabilities[cell * n + c] * phase;
            }
        }
        let values = linalg::svd(&matrix, n)?.values;
        let (operator, nuclear) = (values[0], values.iter().sum::<f64>());
        if operator > table.radius * (1. + 1e-9) || nuclear > table.nuclear_budget * (1. + 1e-9) {
            return Err("raw spectral bound violation".into());
        }
        norms.push(json!({"operator":operator,"nuclear":nuclear}));
    }
    close(&json!(norms), &report["moment_norms"], 1e-7)?;
    Ok(
        json!({"raw_coordinate_audit":true,"frequencies":n-1,"shared_lapack_svd":true,"interval_certificate":false,"diagnostics":diagnostics}),
    )
}
