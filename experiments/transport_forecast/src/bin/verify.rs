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
    require(args.len() == 1, "usage: verify OUTPUT_DIRECTORY")?;
    let root = Path::new(&args[0]);
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
    for row in cases {
        require(row["status"] == "pass", "nonpassing slot")?;
        let seed = row["seed"].as_u64().unwrap();
        let step = row["step"].as_u64().unwrap();
        let path = root.join(format!("plain-seed{seed}-step{step}"));
        let saved = read(&path.join("result.json"))?;
        require(saved == *row, "case/report discrepancy")?;
        let a = &row["audit"];
        for (name, digest) in [
            ("bridge.bin", &a["bridge_binary"]["sha256"]),
            ("forecast.bin", &a["forecast_binary"]["sha256"]),
            ("next-state.bin", &a["next_state_sha256"]),
        ] {
            require(
                hash(&fs::read(path.join(name))?) == digest.as_str().unwrap(),
                "frozen binary hash mismatch",
            )?;
        }
        let bytes = fs::read(path.join("bridge.bin"))?;
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
        let records = read(&path.join("bridge-queries.json"))?;
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
                let s = &a["bridge"][name][m];
                require(
                    s["count"] == counts[split][m]
                        && s["correct"] == correct[split][m]
                        && s["certified"] == certified[split][m],
                    "reported totals disagree",
                )?;
            }
        }
    }
    println!(
        "{}",
        json!({"passed":true,"cases":cases.len(),"replayed_original_hashes":replay.len(),"input_files":inputs.len(),"frozen_query_method_checks":total,
            "independent_fp64_class_gap_checks":independent_pairs,
            "independent_model_sha256":hash(include_bytes!("../../../flat_approx/src/model.rs"))})
    );
    Ok(())
}
