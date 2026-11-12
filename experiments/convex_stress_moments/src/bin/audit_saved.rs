//! Independent saved-table audit. Do not call the training adapter, CE, FFT,
//! projections or metric implementation. Rebuild the frozen data, read raw
//! coordinates, use plain exp(logit), direct Fourier sums and the shared LAPACK
//! SVD. Thus transforms/CE/predictions are independent, but norms are not interval
//! certificates and their SVD implementation is shared with the producer.
use convex_stress_moments::{C, Result, data, hash, linalg};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{fs, path::Path};

#[derive(Deserialize)]
struct Saved {
    vocab: usize,
    radius: f64,
    probabilities: Vec<f64>,
}

fn close(a: &Value, b: &Value, path: &str) -> Result<()> {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) if a.len() == b.len() => {
            for (key, value) in a {
                close(
                    value,
                    b.get(key).ok_or("missing metric key")?,
                    &format!("{path}.{key}"),
                )?;
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().ok_or("invalid number")?;
            let b = b.as_f64().ok_or("invalid number")?;
            if !a.is_finite() || !b.is_finite() || (a - b).abs() > 1e-10 {
                return Err(format!("metric mismatch {path}: {a} != {b}").into());
            }
        }
        _ if a == b => {}
        _ => return Err(format!("metric mismatch {path}").into()),
    }
    Ok(())
}

fn audit(out: &Path, expected: &str) -> Result<()> {
    let report: Value = serde_json::from_slice(&fs::read(out.join("result.json"))?)?;
    let bytes = fs::read(out.join("model.json"))?;
    let protocol: Value = serde_json::from_str(include_str!("../../protocol.json"))?;
    if report["source_fingerprint"] != expected
        || report["model_sha256"] != hash(&bytes)
        || report["protocol"] != protocol
    {
        return Err("producer/model/protocol mismatch".into());
    }
    let name = report["task"].as_str().ok_or("missing task")?;
    if !protocol["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t == name)
    {
        return Err("unplanned task".into());
    }
    let suite: Value = serde_json::from_str(include_str!("../../../convex_stress/suite.json"))?;
    let row = suite["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["name"] == name)
        .ok_or("unknown task")?;
    let task = data::build(row)?;
    if report["data_fingerprints"] != row["data_fingerprints"]
        || report["selection"] != task.selection
    {
        return Err("data/evaluation mismatch".into());
    }
    let model: Saved = serde_json::from_slice(&bytes)?;
    let n = model.vocab;
    if n != task.vocab
        || !(2..=256).contains(&n)
        || model.probabilities.len() != n * n * n
        || !model.radius.is_finite()
        || model.radius <= 0.
        || model
            .probabilities
            .iter()
            .any(|p| !p.is_finite() || *p < -1e-10 || *p > 1. + 1e-10)
        || model
            .probabilities
            .chunks_exact(n)
            .any(|p| (p.iter().sum::<f64>() - 1.).abs() > 1e-10)
    {
        return Err("invalid raw probability table".into());
    }
    let expected_radius = match report["radius_name"].as_str() {
        Some("sqrt") => (n as f64).sqrt(),
        Some("vocab") => n as f64,
        _ => return Err("invalid radius".into()),
    };
    if model.radius != expected_radius {
        return Err("wrong saved radius".into());
    }
    let mut diagnostics = serde_json::Map::new();
    for (split, examples) in &task.splits {
        let (mut total_ce, mut correct, mut transitions, mut transition_correct) = (0., 0, 0, 0);
        let (mut minimum_margin, mut margin_sum, mut true_probability, mut fit) =
            (f64::INFINITY, 0., 0., 0_f64);
        let (mut counts, mut hits) = (vec![0; n], vec![0; n]);
        for e in examples {
            if e.tokens.len() != 2
                || e.targets.len() != 2
                || e.targets[0] != -100
                || e.targets[1] < 0
                || !e.prompt.is_empty()
                || e.field.is_some()
            {
                return Err("unexpected independent audit example".into());
            }
            let (a, b, y) = (
                e.tokens[0] as usize,
                e.tokens[1] as usize,
                e.targets[1] as usize,
            );
            if a >= n || b >= n || y >= n {
                return Err("invalid token or label".into());
            }
            let offset = (b + n * a) * n;
            let p = &model.probabilities[offset..offset + n];
            let mut winner = 0;
            for c in 1..n {
                if p[c].total_cmp(&p[winner]).is_gt() {
                    winner = c;
                }
            }
            let hit = usize::from(winner == y);
            correct += hit;
            counts[y] += 1;
            hits[y] += hit;
            if e.changes[1] {
                transitions += 1;
                transition_correct += hit;
            }
            total_ce += p.iter().map(|x| x.exp()).sum::<f64>().ln() - p[y];
            let other = p
                .iter()
                .enumerate()
                .filter(|(c, _)| *c != y)
                .map(|(_, p)| *p)
                .fold(f64::NEG_INFINITY, f64::max);
            let margin = p[y] - other;
            minimum_margin = minimum_margin.min(margin);
            margin_sum += margin;
            true_probability += p[y];
            for (c, p) in p.iter().enumerate() {
                fit = fit.max((p - f64::from(c == y)).abs());
            }
        }
        let count = examples.len();
        let accuracy = correct as f64 / count as f64;
        let classes: serde_json::Map<String,Value> = counts.iter().enumerate().filter(|(_,n)| **n>0).map(|(c,&n)|
            (c.to_string(),json!({"name":c.to_string(),"count":n,"correct":hits[c],"accuracy":hits[c] as f64/n as f64}))).collect();
        let balanced = counts
            .iter()
            .zip(&hits)
            .filter(|(n, _)| **n > 0)
            .map(|(&n, &h)| h as f64 / n as f64)
            .sum::<f64>()
            / classes.len() as f64;
        let metrics = json!({"loss":total_ce/count as f64,"example_loss":total_ce/count as f64,
            "sequence_accuracy":accuracy,"token_accuracy":accuracy,"balanced_accuracy":balanced,
            "transition_accuracy":if transitions>0 {Some(transition_correct as f64/transitions as f64)} else {None},
            "correct":correct,"exact":correct,"targets":count,"target_count":count,"examples":count,"example_count":count,
            "transition_count":transitions,"classes":classes});
        close(&metrics, &report["metrics"][split], split)?;
        if split == "train" {
            close(&json!(fit), &report["fit_defect"], "fit_defect")?;
            let ideal = (std::f64::consts::E + n as f64 - 1.).ln() - 1.;
            close(
                &json!(total_ce / count as f64 - ideal),
                &report["ideal_ce_excess"],
                "ideal_ce_excess",
            )?;
        }
        diagnostics.insert(split.clone(),json!({"correct":correct,"examples":count,"accuracy":accuracy,
            "ce":total_ce/count as f64,"minimum_label_margin":minimum_margin,"mean_label_margin":margin_sum/count as f64,
            "mean_true_probability":true_probability/count as f64,"one_hot_defect":fit}));
    }
    let codes: Vec<_> = (0..n * n)
        .map(|i| {
            let phase = std::f64::consts::TAU * (i / n * (i % n)) as f64 / n as f64;
            C::new(phase.cos(), phase.sin())
        })
        .collect();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
    let norms: Vec<f64> = pool.install(|| {
        (1..n)
            .into_par_iter()
            .map(|f| -> Result<f64> {
                let mut moment = vec![C::default(); n * n];
                for (cell, m) in moment.iter_mut().enumerate() {
                    for c in 0..n {
                        *m += model.probabilities[cell * n + c] * codes[f * n + c];
                    }
                }
                Ok(linalg::svd(&moment, n)?.values[0])
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let reported: Vec<f64> = serde_json::from_value(report["operator_norms"].clone())?;
    if norms.len() != reported.len()
        || norms
            .iter()
            .zip(&reported)
            .any(|(a, b)| !b.is_finite() || (a - b).abs() > 1e-8 || *a > model.radius * (1. + 1e-9))
    {
        return Err("direct Fourier norm audit mismatch".into());
    }
    println!(
        "{}",
        json!({"independent_audit":true,"task":name,"arm":report["arm"],"radius":report["radius_name"],
        "source_fingerprint":expected,"diagnostics":diagnostics,"max_operator_norm":norms.into_iter().fold(0.,f64::max)})
    );
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [out, expected] => audit(Path::new(out), expected),
        _ => Err("usage: audit_saved OUTPUT_DIRECTORY EXPECTED_PRODUCER_FINGERPRINT".into()),
    }
}
