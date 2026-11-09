//! Independently reread frozen bridge buffers, query reductions and input hashes.
#[path = "../../../signed_transport/src/algebra.rs"]
pub mod algebra;
#[path = "../../../flat_approx/src/model.rs"]
pub mod model;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn near(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-10 * (1. + a.abs().max(b.abs()))
}
fn word(bytes: &[u8], index: usize) -> u64 {
    u64::from_le_bytes(bytes[index * 8..(index + 1) * 8].try_into().unwrap())
}
fn scalar(bytes: &[u8], index: usize) -> f64 {
    f64::from_le_bytes(bytes[index * 8..(index + 1) * 8].try_into().unwrap())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    require(
        args.len() == 2,
        "usage: verify OUTPUT_DIRECTORY TASK197_REPORT",
    )?;
    let root = Path::new(&args[0]);
    let baseline = read(Path::new(&args[1]))?;
    algebra::threads();
    let report = read(&root.join("report.json"))?;
    require(
        report["status"] == "pass" && report["failed"] == 0,
        "cohort failed",
    )?;
    let inputs = report["inputs"]["files"]
        .as_object()
        .ok_or("missing input ledger")?;
    for (path, digest) in inputs {
        require(
            hash(&fs::read(path)?) == digest.as_str().unwrap(),
            "input hash changed",
        )?;
    }
    let cases = report["cases"].as_array().ok_or("missing cases")?;
    require(cases.len() == 26, "missing declared slots")?;
    let replay = report["replay_hashes"]
        .as_array()
        .ok_or("missing replay ledger")?;
    require(
        replay.len() == 6 && replay.iter().all(|r| r["matched"] == true),
        "checkpoint replay did not match",
    )?;
    let mut total = 0usize;
    let mut independent_pairs = 0usize;
    let mut attribution_queries = 0usize;
    for row in cases {
        require(row["status"] == "pass", "nonpassing slot")?;
        let seed = row["seed"].as_u64().unwrap();
        let step = row["step"].as_u64().unwrap();
        let path = root.join(format!("plain-seed{seed}-step{step}"));
        let saved = read(&path.join("result.json"))?;
        require(saved == *row, "case/report discrepancy")?;
        let a = &row["audit"];
        let previous = baseline["cases"]
            .as_array()
            .ok_or("missing baseline cases")?
            .iter()
            .find(|r| r["seed"] == seed && r["step"] == step)
            .ok_or("missing baseline slot")?;
        require(
            a["next_state_sha256"] == previous["audit"]["next_state_sha256"],
            "source update differs from task #197",
        )?;
        require(
            a["instrumented_step_matches"] == true,
            "instrumented step mismatch",
        )?;
        for (name, digest) in [
            ("forecast.bin", &a["forecast_binary"]["sha256"]),
            ("next-state.bin", &a["next_state_sha256"]),
            ("old-state.bin", &a["old_state_sha256"]),
            ("attribution.json", &a["attribution_sha256"]),
        ] {
            require(
                hash(&fs::read(path.join(name))?) == digest.as_str().unwrap(),
                "frozen binary hash mismatch",
            )?;
        }
        attribution_queries += verify_attribution(&path, a)?;
        for family in ["hidden", "readout"] {
            let bytes = fs::read(path.join(format!("{family}-bridge.bin")))?;
            require(
                hash(&bytes) == a["bridge_metadata"][family]["binary"]["sha256"],
                "bridge binary hash mismatch",
            )?;
            if family == "hidden" {
                require(
                    hash(&bytes) == previous["audit"]["bridge_binary"]["sha256"],
                    "hidden baseline bounds changed",
                )?;
            } else {
                let hidden = fs::read(path.join("hidden-bridge.bin"))?;
                let rank_zero_bytes = 8 * (4 + 9409 + 3 * 9409 * 97);
                require(
                    bytes[..rank_zero_bytes] == hidden[..rank_zero_bytes],
                    "rank-zero control changed between basis families",
                )?;
            }
            require(bytes.len() >= 32, "short bridge header")?;
            require(
                [
                    word(&bytes, 0),
                    word(&bytes, 1),
                    word(&bytes, 2),
                    word(&bytes, 3),
                ] == [1, 9409, 97, 9],
                "unexpected bridge format",
            )?;
            let n = 9409;
            let base = 4 + n;
            require(
                bytes.len() == 8 * (base + 9 * n * 97),
                "wrong bridge length",
            )?;
            let records = read(&path.join(format!("{family}-bridge-queries.json")))?;
            // This independently implemented source evaluator never calls the
            // forecast graph. Only the already frozen next-state bytes are used.
            let state = fs::read(path.join("next-state.bin"))?;
            let count = model::Arm::Plain.count();
            require(state.len() == 8 + 12 * count, "invalid next-state length")?;
            let parameters: Vec<_> = state[8..8 + 4 * count]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b) as f64)
                .collect();
            let inputs: Vec<u32> = (0..97)
                .flat_map(|a| (0..97).flat_map(move |b| [a, b, 0]))
                .collect();
            let independent = model::forward(model::Arm::Plain, &parameters, &inputs);
            let records = records.as_array().ok_or("invalid bridge query table")?;
            require(records.len() == 9 * n, "missing query records")?;
            let source = read(&path.join("queries.json"))?;
            let source = source["rows"]
                .as_array()
                .ok_or("missing independent source query table")?;
            require(source.len() == n, "wrong independent query count")?;
            let mut seen = vec![false; 9 * n];
            let mut counts = [[0usize; 9]; 2];
            let mut correct = [[0usize; 9]; 2];
            let mut certified = [[0usize; 9]; 2];
            for rec in records {
                let q = rec["query"].as_u64().unwrap() as usize;
                let t = rec["label"].as_u64().unwrap() as usize;
                let rank = match rec["rank"].as_u64().unwrap() {
                    0 => 0,
                    32 => 1,
                    128 => 2,
                    _ => return Err("unexpected rank".into()),
                };
                let method = match rec["method"].as_str().unwrap() {
                    "old_geometry" => 0,
                    "combined_bridge" => 1,
                    "directional_bridge" => 2,
                    _ => return Err("unexpected method".into()),
                };
                let m = rank * 3 + method;
                require(
                    q < n && t < 97 && !seen[m * n + q],
                    "invalid or duplicate record",
                )?;
                seen[m * n + q] = true;
                require(
                    word(&bytes, 4 + q) == t as u64,
                    "record label differs from frozen candidate",
                )?;
                let u: Vec<_> = (0..97)
                    .map(|c| scalar(&bytes, base + m * n * 97 + q * 97 + c))
                    .collect();
                require(
                    u.iter().all(|x| x.is_finite()) && u[t] == 0.,
                    "nonfinite endpoint or nonzero target gap",
                )?;
                let margin = -(0..97)
                    .filter(|&c| c != t)
                    .map(|c| u[c])
                    .fold(f64::NEG_INFINITY, f64::max);
                if method != 0 {
                    for (c, &endpoint) in u.iter().enumerate() {
                        let gap = independent.logits[q * 97 + c] - independent.logits[q * 97 + t];
                        require(
                            gap <= endpoint + 1e-8 * (1. + gap.abs().max(endpoint.abs())),
                            "independent FP64 class gap exceeds frozen bridge",
                        )?;
                        independent_pairs += 1;
                    }
                }
                let max = u.iter().copied().fold(0., f64::max);
                let ce = max + u.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
                let pass = margin > 1e-8 * (1. + margin.abs());
                require(
                    near(margin, rec["margin_lower"].as_f64().unwrap())
                        && near(ce, rec["ce_upper"].as_f64().unwrap())
                        && near((-ce).exp(), rec["probability_lower"].as_f64().unwrap()),
                    "frozen endpoint reduction mismatch",
                )?;
                require(rec["certified"] == pass, "certificate flag mismatch")?;
                let s = &source[q];
                require(s[0] == q && s[1] == t, "independent target/query mismatch")?;
                let actual_margin = s[if method == 0 { 3 } else { 4 }].as_f64().unwrap();
                let actual_ce = s[if method == 0 { 5 } else { 6 }].as_f64().unwrap();
                require(
                    margin <= actual_margin + 1e-8 * (1. + margin.abs().max(actual_margin.abs()))
                        && actual_ce <= ce + 1e-8 * (1. + actual_ce.abs().max(ce.abs())),
                    "independent source outcome violates frozen bound",
                )?;
                if method != 0 {
                    require(
                        rec["correct"] == (s[1] == s[2]),
                        "future correctness mismatch",
                    )?;
                }
                if method == 0 && actual_margin != 0. {
                    require(
                        rec["correct"] == (actual_margin > 0.),
                        "old correctness mismatch",
                    )?;
                }
                require(!pass || rec["correct"] == true, "false certificate")?;
                let split = match rec["split"].as_str().unwrap() {
                    "train" => 0,
                    "validation" => 1,
                    _ => return Err("unknown split".into()),
                };
                counts[split][m] += 1;
                correct[split][m] += usize::from(rec["correct"] == true);
                certified[split][m] += usize::from(pass);
                total += 1;
            }
            require(seen.iter().all(|x| *x), "missing frozen query/method")?;
            for (split, name) in ["train", "validation"].into_iter().enumerate() {
                for m in 0..9 {
                    let s = &a["bridges"][family][name][m];
                    require(
                        s["count"] == counts[split][m]
                            && s["correct"] == correct[split][m]
                            && s["certified"] == certified[split][m],
                        "reported totals disagree",
                    )?;
                }
            }
        }
    }
    println!(
        "{}",
        json!({"passed":true,"cases":cases.len(),"replayed_original_hashes":replay.len(),"input_files":inputs.len(),"frozen_query_method_checks":total,
            "independent_fp64_class_gap_checks":independent_pairs,
            "independent_attribution_query_checks":attribution_queries,"unchanged_hidden_baselines":cases.len(),
            "task197_report_sha256":hash(&fs::read(&args[1])?),
            "independent_model_sha256":hash(include_bytes!("../../../flat_approx/src/model.rs"))})
    );
    Ok(())
}

/// Independently recover the predicted logit changes from the frozen feature
/// center, new readout and an old-state forward pass. This never reruns a tangent.
fn verify_attribution(path: &Path, audit: &Value) -> Result<usize> {
    let table = read(&path.join("attribution.json"))?;
    for name in [
        "parameter_closure_max_error",
        "tangent_closure_scaled_error",
        "logit_closure_scaled_error",
    ] {
        require(
            table[name] == audit["attribution"][name]
                && table[name].as_f64().ok_or("missing closure")? < 1e-8,
            "attribution closure audit failed",
        )?;
    }
    let parameters = |name: &str| -> Result<Vec<f64>> {
        let state = fs::read(path.join(name))?;
        let count = model::Arm::Plain.count();
        require(state.len() == 8 + 12 * count, "invalid state length")?;
        Ok(state[8..8 + 4 * count]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b) as f64)
            .collect())
    };
    let old = parameters("old-state.bin")?;
    let new = parameters("next-state.bin")?;
    let inputs: Vec<u32> = (0..97)
        .flat_map(|a| (0..97).flat_map(move |b| [a, b, 0]))
        .collect();
    let old = model::forward(model::Arm::Plain, &old, &inputs);
    let bytes = fs::read(path.join("forecast.bin"))?;
    let width = word(&bytes, 2) as usize;
    require(
        word(&bytes, 0) == 1 && word(&bytes, 1) == 9409 && width == 128,
        "unexpected forecast format",
    )?;
    let center: Vec<_> = (0..9409 * width).map(|i| scalar(&bytes, 4 + i)).collect();
    let predicted = algebra::linear(&center, 9409, width, &new[..97 * width], 97);
    let rows = table["queries"]
        .as_array()
        .ok_or("missing attribution rows")?;
    let mut seen = vec![false; 9409];
    let labels = fs::read(path.join("hidden-bridge.bin"))?;
    let mut splits = vec![0; 9409];
    for group in audit["bridge_metadata"]["hidden"]["groups"]
        .as_array()
        .ok_or("missing groups")?
    {
        for fold in group["folds"].as_array().ok_or("missing folds")? {
            if fold["fold"] == 0 {
                for q in fold["queries"]
                    .as_array()
                    .ok_or("missing validation queries")?
                {
                    splits[q.as_u64().unwrap() as usize] = 1;
                }
            }
        }
    }
    require(
        splits.iter().filter(|&&s| s == 1).count() == 4705,
        "invalid validation fold",
    )?;
    let mut counts = [0; 2];
    let mut sums = [[0.; 8]; 2];
    let mut helps = [[0usize; 8]; 2];
    let mut positive = [0; 2];
    let mut agreement = [0; 2];
    for row in rows {
        let q = row[0].as_u64().ok_or("missing query")? as usize;
        let t = row[1].as_u64().ok_or("missing target")? as usize;
        let c = row[2].as_u64().ok_or("missing competitor")? as usize;
        require(
            q < 9409 && t < 97 && c < 97 && c != t && !seen[q],
            "invalid attribution row",
        )?;
        seen[q] = true;
        require(
            word(&labels, 4 + q) == t as u64,
            "attribution label differs from frozen candidate",
        )?;
        let actual_worst = (0..97)
            .filter(|&v| v != t)
            .max_by(|&a, &b| old.logits[q * 97 + a].total_cmp(&old.logits[q * 97 + b]))
            .unwrap();
        require(
            old.logits[q * 97 + c] >= old.logits[q * 97 + actual_worst] - 1e-10,
            "attribution competitor was not old worst",
        )?;
        let terms = row[3].as_array().ok_or("missing attribution terms")?;
        require(terms.len() == 8, "missing attribution component")?;
        let sum: f64 = terms.iter().map(|v| v.as_f64().unwrap()).sum();
        let change = predicted[q * 97 + t] - predicted[q * 97 + c] - old.logits[q * 97 + t]
            + old.logits[q * 97 + c];
        require(
            near(sum, row[4].as_f64().ok_or("missing sum")?) && near(sum, change),
            "independent attribution sum mismatch",
        )?;
        let split = splits[q];
        counts[split] += 1;
        positive[split] += usize::from(sum > 0.);
        agreement[split] += usize::from((terms[1].as_f64().unwrap() > 0.) == (sum > 0.));
        for (i, term) in terms.iter().enumerate() {
            let term = term.as_f64().unwrap();
            sums[split][i] += term;
            helps[split][i] += usize::from(term > 0.);
        }
    }
    require(seen.iter().all(|&x| x), "missing attribution query")?;
    for (s, name) in ["train", "validation"].into_iter().enumerate() {
        let summary = &table[name];
        require(
            *summary == audit["attribution"][name] && summary["count"] == counts[s],
            "attribution summary differs from report",
        )?;
        let n = counts[s] as f64;
        for i in 0..8 {
            require(
                near(
                    sums[s][i] / n,
                    summary["mean_margin_contributions"][i].as_f64().unwrap(),
                ) && near(
                    helps[s][i] as f64 / n,
                    summary["positive_fraction"][i].as_f64().unwrap(),
                ),
                "attribution component summary differs from query table",
            )?;
        }
        require(
            near(
                positive[s] as f64 / n,
                summary["total_positive_fraction"].as_f64().unwrap(),
            ) && near(
                agreement[s] as f64 / n,
                summary["readout_credit_sign_agreement"].as_f64().unwrap(),
            ),
            "attribution sign summary differs from query table",
        )?;
    }
    Ok(rows.len())
}
