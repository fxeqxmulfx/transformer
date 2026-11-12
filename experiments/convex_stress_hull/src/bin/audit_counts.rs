//! Independent ordinary-CE gradient and unrestricted CC8 oracle on a saved model.
//! No centered gradients, gradient memory, optimizer oracle or stored loss enters
//! the calculation. This is an f64 cross-check, not an interval certificate.
//! Port of the counts saved-model auditor at 981a7eca to the hull envelope.
//! The count forward map and causal adapter are shared with the producer;
//! uncentered derivatives and the exhaustive oracle below are independent.
//! Process one query and sixteen keys at a time. The previous all-query dense
//! gradient exhausted host memory on MQAR. Blocking preserves each row's exact
//! addition order and the exhaustive minima, without changing the producer.
use convex_stress_hull::{Expert, Result, data, hash, meta, metrics::Predictor};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};

const BLOCK_KEYS: usize = 16;

struct OracleObservation {
    items: Vec<(usize, usize)>,
    residual: Vec<f64>,
}

fn query_minimum(
    keys: usize,
    vocab: usize,
    pad: usize,
    records: &[OracleObservation],
    block: usize,
) -> Result<(f64, usize)> {
    if keys == 0
        || vocab == 0
        || block == 0
        || records.is_empty()
        || records.iter().any(|r| {
            r.residual.len() != vocab
                || r.residual.iter().any(|r| !r.is_finite())
                || r.items.iter().any(|(k, _)| *k >= keys)
                || r.items.windows(2).any(|w| w[0].0 >= w[1].0)
        })
    {
        return Err("invalid blocked oracle observations".into());
    }
    let (mut minimum, mut peak_rows) = (f64::INFINITY, 0);
    for first in (0..keys).step_by(block) {
        let end = (first + block).min(keys);
        let mut gradients = vec![BTreeMap::<usize, Vec<f64>>::new(); end - first];
        for record in records {
            let mut cursor = record.items.partition_point(|(k, _)| *k < first);
            for (offset, gradient) in gradients.iter_mut().enumerate() {
                let key = first + offset;
                let value = if record.items.get(cursor).is_some_and(|(k, _)| *k == key) {
                    let value = record.items[cursor].1;
                    cursor += 1;
                    value
                } else {
                    pad
                };
                let row = gradient.entry(value).or_insert_with(|| vec![0.; vocab]);
                for (target, residual) in row.iter_mut().zip(&record.residual) {
                    *target += residual;
                }
            }
        }
        peak_rows = peak_rows.max(gradients.iter().map(BTreeMap::len).sum());
        for gradient in gradients {
            let cost = gradient
                .values()
                .map(|row| row.iter().copied().fold(f64::INFINITY, f64::min))
                .sum::<f64>();
            minimum = minimum.min(cost);
        }
    }
    Ok((minimum, peak_rows))
}

fn uncentered(logits: &[f64], label: usize, scale: f64) -> (f64, Vec<f64>) {
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<_> = logits.iter().map(|z| (z - maximum).exp()).collect();
    let normalizer: f64 = exps.iter().sum();
    (
        maximum + normalizer.ln() - logits[label],
        exps.iter()
            .enumerate()
            .map(|(c, e)| scale * (e / normalizer - f64::from(c == label)))
            .collect(),
    )
}

fn audit(directory: &Path, expected: &str) -> Result<Value> {
    let result: Value = serde_json::from_slice(&fs::read(directory.join("result.json"))?)?;
    if result["source_fingerprint"] != expected
        || result["protocol"] != meta::protocol()?
        || result["status"] != "complete"
        || result["candidate"] != "counts"
    {
        return Err("unrecognized frozen model source".into());
    }
    let bytes = fs::read(directory.join("model.json"))?;
    if hash(&bytes) != result["model_sha256"] {
        return Err("model hash mismatch".into());
    }
    let Expert::Counts(model) = serde_json::from_slice(&bytes)? else {
        return Err("expected count model".into());
    };
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
    if adapter.vocab != task.vocab || model.scale <= 0. || result["scale"] != model.scale {
        return Err("bad model dimensions".into());
    }
    let mut queries = BTreeMap::<usize, Vec<(usize, usize)>>::new();
    let (mut ce, mut dot, mut count, mut mass_error) = (0., 0., 0usize, 0f64);
    for (example_index, example) in task.splits["train"].iter().enumerate() {
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
            let (loss, residual) = uncentered(&logits, label as usize, model.scale);
            ce += loss;
            dot += residual
                .iter()
                .zip(&logits)
                .map(|(r, z)| r * z / model.scale)
                .sum::<f64>();
            let query = *prefix.last().unwrap() as usize;
            queries
                .entry(query)
                .or_default()
                .push((example_index, position));
            count += 1;
        }
    }
    if count == 0 || result["train_prefixes"] != count || mass_error > 1e-10 {
        return Err("prefix count or logit mass mismatch".into());
    }
    let (mut minimum, mut peak_gradient_rows, mut peak_query_observations) = (0., 0, 0);
    for observations in queries.values() {
        let records: Vec<_> = observations
            .iter()
            .map(|&(example_index, position)| {
                let example = &task.splits["train"][example_index];
                let prefix = &example.tokens[..=position];
                let (_, residual) = uncentered(
                    &model.logits(prefix),
                    example.targets[position] as usize,
                    model.scale,
                );
                OracleObservation {
                    items: model.features.encode(prefix),
                    residual,
                }
            })
            .collect();
        let (cost, rows) = query_minimum(
            adapter.keys(),
            adapter.vocab,
            adapter.pad(),
            &records,
            BLOCK_KEYS,
        )?;
        minimum += cost;
        peak_gradient_rows = peak_gradient_rows.max(rows);
        peak_query_observations = peak_query_observations.max(records.len());
    }
    let mean_ce = ce / count as f64;
    let mean_gap = (dot - minimum) / count as f64;
    let ce_error = (mean_ce - result["training_ce"].as_f64().ok_or("missing CE")?).abs();
    let gap_error = result["fresh_fw_gap"]
        .as_f64()
        .map(|gap| (mean_gap - gap).abs());
    if result["arm"] != "forward_only" && gap_error.is_none() {
        return Err("missing reported count gap".into());
    }
    if ce_error > 1e-8 || gap_error.is_some_and(|e| e > 1e-8) || mean_gap < -1e-8 {
        return Err(format!(
            "dense audit differs: CE {ce_error}, gap {gap_error:?}, value {mean_gap}"
        )
        .into());
    }
    Ok(
        json!({"independent_audit":true,"task":result["task"],"arm":result["arm"],"candidate":"counts","scale":model.scale,
        "model_sha256":result["model_sha256"],"prefixes":count,
        "independent_mean_ce":mean_ce,"independent_fresh_gap":mean_gap,
        "ce_error":ce_error,"gap_error":gap_error,"observed_mass_error":mass_error,
        "block_keys":BLOCK_KEYS,"peak_gradient_rows":peak_gradient_rows,"peak_query_observations":peak_query_observations,
        "method":"Uncentered full-vocabulary gradient and exhaustive query/key/value/class oracle; one query and sixteen keys at a time, f64."}),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [out, expected] => {
            println!("{}", audit(Path::new(out), expected)?);
            Ok(())
        }
        _ => Err("usage: audit_counts CANDIDATE_DIRECTORY PRODUCER_FINGERPRINT".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deliberately retain the previous dense procedure as the small-data oracle.
    fn dense(keys: usize, vocab: usize, pad: usize, records: &[OracleObservation]) -> f64 {
        let mut gradients = vec![BTreeMap::<usize, Vec<f64>>::new(); keys];
        for r in records {
            let mut values = vec![pad; keys];
            for &(k, v) in &r.items {
                values[k] = v;
            }
            for (k, &v) in values.iter().enumerate() {
                let row = gradients[k].entry(v).or_insert_with(|| vec![0.; vocab]);
                for (a, b) in row.iter_mut().zip(&r.residual) {
                    *a += b;
                }
            }
        }
        gradients
            .iter()
            .map(|g| {
                g.values()
                    .map(|row| row.iter().copied().fold(f64::INFINITY, f64::min))
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn blocked_oracle_matches_dense_with_padding_unused_classes_and_key_ties() -> Result<()> {
        let (keys, vocab, pad) = (37, 7, 7);
        let mut records = vec![];
        for s in 0..53 {
            let logits: Vec<_> = (0..vocab)
                .map(|c| ((s * 13 + c * 7) % 17) as f64 / 17.)
                .collect();
            let (_, residual) = uncentered(&logits, s % 3, 2.);
            records.push(OracleObservation {
                items: (0..keys)
                    .filter(|k| (k + s) % 5 == 0)
                    .map(|k| (k, (s + 3 * k) % 11))
                    .collect(),
                residual,
            });
        }
        let reference = dense(keys, vocab, pad, &records);
        for width in [1, 2, 3, 16, 37, 50] {
            let (actual, rows) = query_minimum(keys, vocab, pad, &records, width)?;
            assert_eq!(actual.to_bits(), reference.to_bits());
            assert!(rows <= width.min(keys) * records.len());
        }
        let uniform = vec![OracleObservation {
            items: vec![],
            residual: vec![-1., 0.5, 0.5],
        }];
        assert_eq!(query_minimum(19, 3, 3, &uniform, 4)?.0, -1.);
        Ok(())
    }

    #[test]
    fn invalid_streaming_inputs_are_rejected() {
        let r = vec![OracleObservation {
            items: vec![(1, 0), (1, 1)],
            residual: vec![0., 1.],
        }];
        assert!(query_minimum(2, 2, 2, &r, 1).is_err());
        assert!(query_minimum(2, 2, 2, &r, 0).is_err());
        assert!(query_minimum(2, 2, 2, &[], 1).is_err());
    }
}
