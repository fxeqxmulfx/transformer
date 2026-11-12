//! Independent ordinary-CE gradient and unrestricted CC8 oracle on a saved model.
//! No centered gradients, gradient memory, optimizer oracle or stored loss enters
//! the calculation. This is an f64 cross-check, not an interval certificate.
//! Port of the adaptive saved-model auditor at f3e27323 to the count alphabet.
use convex_stress_counts::{AdaptiveModel, Result, data, hash, metrics::Predictor};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};

type QueryGradient = Vec<BTreeMap<usize, Vec<f64>>>;

fn audit(directory: &Path) -> Result<Value> {
    let result: Value = serde_json::from_slice(&fs::read(directory.join("result.json"))?)?;
    if !["b06514ec0c08f583d74fe431b3e3a97ea898366c849c2c71f0f3742ae03de210"].contains(
        &result["source_fingerprint"]
            .as_str()
            .ok_or("missing source")?,
    ) {
        return Err("unrecognized frozen model source".into());
    }
    let bytes = fs::read(directory.join("model.json"))?;
    if hash(&bytes) != result["model_sha256"] {
        return Err("model hash mismatch".into());
    }
    let model: AdaptiveModel = serde_json::from_slice(&bytes)?;
    model.feasibility_error()?;
    let suite: Value = serde_json::from_str(include_str!("../../../convex_stress/suite.json"))?;
    let row = suite["tasks"]
        .as_array()
        .ok_or("missing suite")?
        .iter()
        .find(|row| row["name"] == result["task"])
        .ok_or("unknown task")?;
    if row["data_fingerprints"] != result["data_fingerprints"] {
        return Err("data fingerprint mismatch".into());
    }
    let task = data::build(row)?;
    let adapter = model.features.storage();
    if adapter.vocab != task.vocab || model.scale <= 0. {
        return Err("bad model dimensions".into());
    }
    let mut gradients = BTreeMap::<usize, QueryGradient>::new();
    let (mut ce, mut dot, mut count, mut mass_error) = (0., 0., 0usize, 0f64);
    for example in &task.splits["train"] {
        for (position, &label) in example.targets.iter().enumerate() {
            if label < 0 {
                continue;
            }
            let prefix = &example.tokens[..=position];
            let logits = model.logits(prefix);
            mass_error = mass_error.max((logits.iter().sum::<f64>() / model.scale - 1.).abs());
            if logits.iter().any(|&z| z < -1e-12 || !z.is_finite()) {
                return Err("invalid observed logits".into());
            }
            let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<_> = logits.iter().map(|z| (z - maximum).exp()).collect();
            let normalizer: f64 = exps.iter().sum();
            ce += maximum + normalizer.ln() - logits[label as usize];
            let residual: Vec<_> = exps
                .iter()
                .enumerate()
                .map(|(c, e)| model.scale * (e / normalizer - f64::from(c == label as usize)))
                .collect();
            dot += residual
                .iter()
                .zip(&logits)
                .map(|(r, z)| r * z / model.scale)
                .sum::<f64>();
            let query = *prefix.last().unwrap() as usize;
            let query_gradient = gradients
                .entry(query)
                .or_insert_with(|| vec![BTreeMap::new(); adapter.keys()]);
            let mut values = vec![adapter.pad(); adapter.keys()];
            for (key, value) in model.features.encode(prefix) {
                values[key] = value;
            }
            for (key, &value) in values.iter().enumerate() {
                let row = query_gradient[key]
                    .entry(value)
                    .or_insert_with(|| vec![0.; adapter.vocab]);
                for (target, r) in row.iter_mut().zip(&residual) {
                    *target += r;
                }
            }
            count += 1;
        }
    }
    if count == 0 || result["train_prefixes"] != count || mass_error > 1e-10 {
        return Err("prefix count or logit mass mismatch".into());
    }
    let minimum: f64 = gradients
        .values()
        .map(|query| {
            query
                .iter()
                .map(|key| {
                    key.values()
                        .map(|row| row.iter().copied().fold(f64::INFINITY, f64::min))
                        .sum::<f64>()
                })
                .fold(f64::INFINITY, f64::min)
        })
        .sum();
    let mean_ce = ce / count as f64;
    let mean_gap = (dot - minimum) / count as f64;
    let ce_error = (mean_ce - result["training"]["mean_ce"].as_f64().ok_or("missing CE")?).abs();
    let gap_error = (mean_gap
        - result["training"]["mean_fw_gap"]
            .as_f64()
            .ok_or("missing gap")?)
    .abs();
    if ce_error > 1e-8 || gap_error > 1e-8 || mean_gap < -1e-8 {
        return Err(format!(
            "dense audit differs: CE {ce_error}, gap {gap_error}, value {mean_gap}"
        )
        .into());
    }
    Ok(
        json!({"task":result["task"],"arm":result["arm"],"scale":model.scale,
        "model_sha256":result["model_sha256"],"prefixes":count,
        "independent_mean_ce":mean_ce,"independent_fresh_gap":mean_gap,
        "ce_error":ce_error,"gap_error":gap_error,"observed_mass_error":mass_error,
        "method":"Uncentered full-vocabulary gradient and independent exhaustive query/key/value/class oracle, f64."}),
    )
}
fn main() -> Result<()> {
    let directories: Vec<_> = std::env::args().skip(1).collect();
    if directories.is_empty() {
        return Err("usage: audit_saved CELL [CELL ...]".into());
    }
    for directory in directories {
        println!("{}", audit(Path::new(&directory))?);
    }
    Ok(())
}
