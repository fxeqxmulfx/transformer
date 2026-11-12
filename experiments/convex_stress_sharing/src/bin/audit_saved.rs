//! Independent saved-table audit. Do not call the training adapter, CE, FFT,
//! projections or metric implementation. Rebuild the frozen data, read raw
//! coordinates, use plain exp(logit), direct Fourier sums and the shared LAPACK
//! SVD. Ported from 2df45bb9 to audit raw fixed-group coordinates under CM10.
//! Saved-map discovery is checked by producer replay; this audit independently
//! checks compact map shape, observed consistency, multiplicities and predictions.
//! Thus transforms/CE/predictions are independent, but norms are not interval
//! certificates and their SVD implementation is shared with the producer.
use convex_stress_sharing::{C, Result, data, hash, linalg};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{fs, path::Path};

#[derive(Deserialize)]
struct Saved {
    vocab: usize,
    radius: f64,
    nuclear_budget: f64,
    probabilities: Vec<f64>,
}

#[derive(Deserialize)]
struct Maps {
    rows: Vec<usize>,
    columns: Vec<usize>,
    row_count: usize,
    column_count: usize,
}

#[derive(Deserialize)]
struct Envelope {
    groups: Maps,
    table: Saved,
}

#[derive(Deserialize)]
struct Spectrum {
    operator: f64,
    nuclear: f64,
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

fn audit(out: &Path, expected: &str, include_spectrum: bool) -> Result<()> {
    let report: Value = serde_json::from_slice(&fs::read(out.join("result.json"))?)?;
    let bytes = fs::read(out.join("model.json"))?;
    let protocol: Value = serde_json::from_str(include_str!("../../protocol.json"))?;
    if report["source_fingerprint"] != expected
        || report["model_sha256"] != hash(&bytes)
        || report["protocol"] != protocol
        || report["status"] != "complete"
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
    let saved: Envelope = serde_json::from_slice(&bytes)?;
    let (model, maps) = (&saved.table, &saved.groups);
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
    for (map, count) in [
        (&maps.rows, maps.row_count),
        (&maps.columns, maps.column_count),
    ] {
        if map.len() != n
            || count == 0
            || count > n
            || map.iter().any(|&g| g >= count)
            || (0..count).any(|g| !map.contains(&g))
        {
            return Err("invalid raw group map".into());
        }
    }
    for col in 0..n {
        for row in 0..n {
            if (row >= maps.row_count || col >= maps.column_count)
                && model.probabilities[(row + n * col) * n..(row + n * col + 1) * n]
                    .iter()
                    .any(|p| (p - 1. / n as f64).abs() > 1e-10)
            {
                return Err("nonuniform inactive padding".into());
            }
        }
    }
    let mut seen = vec![None; n * n];
    let mut multiplicities = vec![0usize; n * n];
    for e in &task.splits["train"] {
        let cell = maps.rows[e.tokens[1] as usize] + n * maps.columns[e.tokens[0] as usize];
        let label = e.targets[1] as usize;
        if seen[cell].is_some_and(|y| y != label) {
            return Err("conflicting observed group".into());
        }
        seen[cell] = Some(label);
        multiplicities[cell] += 1;
    }
    let maximum_multiplicity = *multiplicities.iter().max().unwrap();
    if report["maximum_cell_multiplicity"] != maximum_multiplicity
        || report["step_size"].as_f64() != Some(1. / maximum_multiplicity as f64)
        || report["train_prefixes"] != task.splits["train"].len()
    {
        return Err("incorrect original-observation weights".into());
    }
    let (rows, columns) = (maps.row_count as f64, maps.column_count as f64);
    let expected_budgets = match report["budget_name"].as_str() {
        Some("full_rank") => ((n as f64).sqrt(), rows.min(columns) * (n as f64).sqrt()),
        Some("rank_one") => ((rows * columns).sqrt(), (rows * columns).sqrt()),
        _ => return Err("invalid radius".into()),
    };
    if (model.radius, model.nuclear_budget) != expected_budgets
        || report["radius"].as_f64() != Some(model.radius)
        || report["nuclear_budget"].as_f64() != Some(model.nuclear_budget)
    {
        return Err("wrong saved budgets".into());
    }
    let mut diagnostics = serde_json::Map::new();
    for (split, examples) in &task.splits {
        let (mut covered, mut covered_correct, mut conflicts, mut uncovered_correct) = (0, 0, 0, 0);
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
            let offset = (maps.rows[b] + n * maps.columns[a]) * n;
            let p = &model.probabilities[offset..offset + n];
            let mut winner = 0;
            for c in 1..n {
                if p[c].total_cmp(&p[winner]).is_gt() {
                    winner = c;
                }
            }
            let hit = usize::from(winner == y);
            if let Some(observed_label) = seen[offset / n] {
                covered += 1;
                covered_correct += hit;
                conflicts += usize::from(observed_label != y);
            } else {
                uncovered_correct += hit;
            }
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
            "covered_examples":covered,"covered_correct":covered_correct,"covered_ground_truth_conflicts":conflicts,
            "uncovered_examples":count-covered,"uncovered_correct":uncovered_correct,
            "ce":total_ce/count as f64,"minimum_label_margin":minimum_margin,"mean_label_margin":margin_sum/count as f64,
            "mean_label_coordinate":true_probability/count as f64,"one_hot_defect":fit}));
    }
    if !include_spectrum {
        // Post-training attribution only. No held-out label enters the producer.
        println!(
            "{}",
            json!({"coverage_analysis":true,"spectrum_rechecked":false,
            "task":name,"arm":report["arm"],"budget":report["budget_name"],"source_fingerprint":expected,
            "observed_unique_groups":seen.iter().filter(|label|label.is_some()).count(),
            "group_grid_cells":maps.row_count*maps.column_count,"diagnostics":diagnostics})
        );
        return Ok(());
    }
    let codes: Vec<_> = (0..n * n)
        .map(|i| {
            let phase = std::f64::consts::TAU * (i / n * (i % n)) as f64 / n as f64;
            C::new(phase.cos(), phase.sin())
        })
        .collect();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
    let norms: Vec<Spectrum> = pool.install(|| {
        (1..n)
            .into_par_iter()
            .map(|f| -> Result<Spectrum> {
                let mut moment = vec![C::default(); n * n];
                for (cell, m) in moment.iter_mut().enumerate() {
                    for c in 0..n {
                        *m += model.probabilities[cell * n + c] * codes[f * n + c];
                    }
                }
                let values = linalg::svd(&moment, n)?.values;
                Ok(Spectrum {
                    operator: values[0],
                    nuclear: values.iter().sum(),
                })
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let reported: Vec<Spectrum> = serde_json::from_value(report["moment_norms"].clone())?;
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
        json!({"independent_audit":true,"task":name,"arm":report["arm"],"budget":report["budget_name"],
        "row_groups":maps.row_count,"column_groups":maps.column_count,"maximum_cell_multiplicity":maximum_multiplicity,
        "source_fingerprint":expected,"diagnostics":diagnostics,"max_operator_norm":norms.iter().map(|s|s.operator).fold(0.,f64::max),
        "max_nuclear_norm":norms.iter().map(|s|s.nuclear).fold(0.,f64::max)})
    );
    Ok(())
}
fn compare(old: &Path, new: &Path, expected: &str) -> Result<()> {
    #[derive(Deserialize)]
    struct Probabilities {
        vocab: usize,
        probabilities: Vec<f64>,
    }
    let old_report: Value = serde_json::from_slice(&fs::read(old.join("result.json"))?)?;
    let new_report: Value = serde_json::from_slice(&fs::read(new.join("result.json"))?)?;
    let old_bytes = fs::read(old.join("model.json"))?;
    let new_bytes = fs::read(new.join("model.json"))?;
    if old_report["source_fingerprint"]
        != "913357a9c11a6f700148dd73bb60d23e8e3f00e601bfb6b24aea5825c62ce742"
        || new_report["source_fingerprint"] != expected
        || old_report["budget_name"] != new_report["budget_name"]
        || !matches!(
            new_report["budget_name"].as_str(),
            Some("rank_one" | "full_rank")
        )
        || old_report["task"] != new_report["task"]
        || old_report["arm"] != new_report["arm"]
        || old_report["model_sha256"] != hash(&old_bytes)
        || new_report["model_sha256"] != hash(&new_bytes)
    {
        return Err("invalid regression comparison identity".into());
    }
    let a: Probabilities = serde_json::from_slice(&old_bytes)?;
    let b: Envelope = serde_json::from_slice(&new_bytes)?;
    if a.vocab != b.table.vocab
        || a.probabilities != b.table.probabilities
        || b.groups.rows != (0..a.vocab).collect::<Vec<_>>()
        || b.groups.columns != (0..a.vocab).collect::<Vec<_>>()
    {
        return Err("endpoint-budget control differs from original coordinates".into());
    }
    println!(
        "{}",
        json!({"original_coordinates_equal":true,"task":new_report["task"],"arm":new_report["arm"]})
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
        [out, expected] => audit(Path::new(out), expected, true),
        ["coverage", out, expected] => audit(Path::new(out), expected, false),
        ["compare", old, new, expected] => compare(Path::new(old), Path::new(new), expected),
        _ => Err("usage: audit_saved [coverage] OUTPUT_DIRECTORY EXPECTED_PRODUCER_FINGERPRINT | compare OLD NEW FINGERPRINT".into()),
    }
}
