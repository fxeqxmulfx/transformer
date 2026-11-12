//! Independent saved-table audit. Do not call the training adapter, CE, FFT,
//! projections or metric implementation. Rebuild the frozen data, read raw
//! coordinates, use plain exp(logit), direct Fourier sums and the shared LAPACK
//! SVD. Ported from bac54d3e for the common model envelope and original residue
//! pooling metrics. No producer prediction, loss, FFT or projection is called.
//! Thus transforms/CE/predictions are independent, but norms are not interval
//! certificates and their SVD implementation is shared with the producer.
use convex_stress_hull::{
    Result, data, hash,
    spectral::{C, linalg},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{fs, path::Path};

#[derive(Deserialize)]
#[serde(tag = "kind", content = "model")]
enum Envelope {
    Moments(Saved),
}

#[derive(Deserialize)]
struct Saved {
    vocab: usize,
    radius: f64,
    nuclear_budget: f64,
    probabilities: Vec<f64>,
}

#[derive(Deserialize)]
struct Spectrum {
    operator: f64,
    nuclear: f64,
}

fn pooled_class(p: &[f64], base: i64) -> usize {
    let maximum = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut pooled = [0.; 97];
    for (c, &value) in p.iter().enumerate() {
        pooled[(c as i64 - p.len() as i64 / 2 + base).rem_euclid(97) as usize] +=
            (value - maximum).exp();
    }
    let mut best = 0;
    for c in 1..97 {
        if pooled[c] > pooled[best] {
            best = c;
        }
    }
    best
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
        || report["status"] != "complete"
        || report["scale"] != 1.
    {
        return Err("producer/model/protocol mismatch".into());
    }
    let name = report["task"].as_str().ok_or("missing task")?;
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
    let Envelope::Moments(model) = serde_json::from_slice(&bytes)?;
    let n = model.vocab;
    if n != task.vocab
        || !(2..=256).contains(&n)
        || model.probabilities.len() != n * n * n
        || !model.radius.is_finite()
        || model.radius <= 0.
        || !model.nuclear_budget.is_finite()
        || model.nuclear_budget <= 0.
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
    let expected_budgets = match report["candidate"].as_str() {
        Some("full_rank") => ((n as f64).sqrt(), n as f64 * (n as f64).sqrt()),
        Some("rank_one") => (n as f64, n as f64),
        _ => return Err("invalid radius".into()),
    };
    if (model.radius, model.nuclear_budget) != expected_budgets {
        return Err("wrong saved budgets".into());
    }
    let mut diagnostics = serde_json::Map::new();
    for (split, examples) in &task.splits {
        let (mut total_ce, mut correct, mut transitions, mut transition_correct) = (0., 0, 0, 0);
        let mut decoded_correct = 0;
        let field = examples[0].field.is_some();
        let (mut minimum_margin, mut margin_sum, mut true_probability, mut fit) =
            (f64::INFINITY, 0., 0., 0_f64);
        let (mut counts, mut hits) = (vec![0; n], vec![0; n]);
        for e in examples {
            if e.tokens.len() != 2
                || e.targets.len() != 2
                || e.targets[0] != -100
                || e.targets[1] < 0
                || !e.prompt.is_empty()
                || e.field.is_some() != field
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
            if let Some((base, future)) = e.field {
                // Preserve the benchmark's residue-pooling and numerical shift.
                // Native-label hits and class statistics remain distinct below.
                decoded_correct += usize::from(pooled_class(p, base) as i64 == future);
            } else {
                decoded_correct += hit;
            }
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
        let accuracy = decoded_correct as f64 / count as f64;
        let classes: serde_json::Map<String,Value> = counts.iter().enumerate().filter(|(_,n)| **n>0).map(|(c,&n)|
            (c.to_string(),json!({"name":c.to_string(),"count":n,"correct":hits[c],"accuracy":hits[c] as f64/n as f64}))).collect();
        let balanced = counts
            .iter()
            .zip(&hits)
            .filter(|(n, _)| **n > 0)
            .map(|(&n, &h)| h as f64 / n as f64)
            .sum::<f64>()
            / classes.len() as f64;
        let mut metrics = json!({"loss":total_ce/count as f64,"example_loss":total_ce/count as f64,
            "sequence_accuracy":accuracy,"token_accuracy":accuracy,"balanced_accuracy":balanced,
            "transition_accuracy":if transitions>0 {Some(transition_correct as f64/transitions as f64)} else {None},
            "correct":decoded_correct,"exact":decoded_correct,"targets":count,"target_count":count,"examples":count,"example_count":count,
            "transition_count":transitions,"classes":classes});
        if field {
            metrics["decoded_accuracy"] = accuracy.into();
            metrics["native_token_accuracy"] = (correct as f64 / count as f64).into();
            metrics["native_correct"] = correct.into();
            metrics["field_correct"] = decoded_correct.into();
        }
        close(&metrics, &report["metrics"][split], split)?;
        if split == "train" {
            close(
                &json!(total_ce / count as f64),
                &report["training_ce"],
                "training_ce",
            )?;
            close(&json!(fit), &report["fit_defect"], "fit_defect")?;
            let ideal = (std::f64::consts::E + n as f64 - 1.).ln() - 1.;
            close(
                &json!(total_ce / count as f64 - ideal),
                &report["ideal_ce_excess"],
                "ideal_ce_excess",
            )?;
        }
        diagnostics.insert(split.clone(),json!({"correct":decoded_correct,"native_correct":correct,"examples":count,"accuracy":accuracy,
            "ce":total_ce/count as f64,"minimum_label_margin":minimum_margin,"mean_label_margin":margin_sum/count as f64,
            "mean_label_coordinate":true_probability/count as f64,"one_hot_defect":fit}));
    }
    let codes: Vec<_> = (0..n * n)
        .map(|i| {
            let phase = std::f64::consts::TAU * (i / n * (i % n)) as f64 / n as f64;
            C::new(phase.cos(), phase.sin())
        })
        .collect();
    let probabilities = &model.probabilities;
    let norms: Vec<Spectrum> = std::thread::scope(|scope| -> Result<Vec<Spectrum>> {
        let handles: Vec<_> = (0..8)
            .map(|worker| {
                let codes = &codes;
                scope.spawn(move || -> Result<Vec<(usize, Spectrum)>> {
                    let mut out = vec![];
                    for f in (1 + worker..n).step_by(8) {
                        let mut moment = vec![C::default(); n * n];
                        for (cell, m) in moment.iter_mut().enumerate() {
                            for c in 0..n {
                                *m += probabilities[cell * n + c] * codes[f * n + c];
                            }
                        }
                        let values = linalg::svd(&moment, n)?.values;
                        out.push((
                            f,
                            Spectrum {
                                operator: values[0],
                                nuclear: values.iter().sum(),
                            },
                        ));
                    }
                    Ok(out)
                })
            })
            .collect();
        let mut pairs = vec![];
        for handle in handles {
            pairs.extend(
                handle
                    .join()
                    .map_err(|_| "direct DFT audit worker panicked")??,
            );
        }
        pairs.sort_by_key(|(f, _)| *f);
        Ok(pairs.into_iter().map(|(_, s)| s).collect())
    })?;
    let reported: Vec<Spectrum> =
        serde_json::from_value(report["feasibility"]["moment_norms"].clone())?;
    if norms.len() != reported.len()
        || norms.iter().zip(&reported).any(|(a, b)| {
            !b.operator.is_finite()
                || !b.nuclear.is_finite()
                || (a.operator - b.operator).abs() > 1e-8
                || (a.nuclear - b.nuclear).abs() > 1e-7
                || a.operator > model.radius * (1. + 1e-9)
                || a.nuclear > model.nuclear_budget * (1. + 1e-9)
        })
    {
        return Err("direct Fourier norm audit mismatch".into());
    }
    println!(
        "{}",
        json!({"independent_audit":true,"task":name,"arm":report["arm"],"candidate":report["candidate"],
        "source_fingerprint":expected,"diagnostics":diagnostics,"max_operator_norm":norms.iter().map(|s|s.operator).fold(0.,f64::max),
        "max_nuclear_norm":norms.iter().map(|s|s.nuclear).fold(0.,f64::max)})
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
        _ => Err("usage: audit_moments CANDIDATE_DIRECTORY PRODUCER_FINGERPRINT".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::pooled_class;
    #[test]
    fn residue_pooling_preserves_alias_counts_and_base_shift() {
        let mut p = vec![0.; 256];
        assert_eq!(pooled_class(&p, 0), 0);
        p[200] = 1.;
        assert_eq!(pooled_class(&p, 0), 72);
        assert_eq!(pooled_class(&p, 7), 79);
        for x in &mut p {
            *x += 0.25;
        }
        assert_eq!(pooled_class(&p, 7), 79);
    }
}
