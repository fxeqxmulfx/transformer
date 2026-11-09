//! Independent trace energies, cumulative identities and FP64 margin checks.
#[path = "../../../signed_transport/src/algebra.rs"]
pub mod algebra;
#[path = "../../../flat_approx/src/model.rs"]
pub mod model;
#[path = "../pairing.rs"]
mod pairing;
#[path = "../schedule.rs"]
mod schedule;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::{self, File},
    io::{BufRead, BufReader, Read},
    path::Path,
};
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
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-8 * (1. + a.abs().max(b.abs()))
}
fn number(v: &Value) -> Result<f64> {
    v.as_f64()
        .filter(|x| x.is_finite())
        .ok_or("invalid finite number".into())
}
fn word(bytes: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap())
}
fn scalar(bytes: &[u8], i: usize) -> f64 {
    f64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap())
}
/// Direct sums and group means, independent of the producer's projectors.
fn energies(e: &[f64], active: &[bool]) -> [f64; 3] {
    let mut sums = [[0.; 128]; 2];
    let mut sq = [0.; 2];
    let mut n = [0.; 2];
    for (c, row) in e.as_chunks::<128>().0.iter().enumerate() {
        let s = usize::from(!active[c]);
        n[s] += 1.;
        for (k, &v) in row.iter().enumerate() {
            sums[s][k] += v;
            sq[s] += v * v;
        }
    }
    let mut out = [0.; 3];
    for s in 0..2 {
        out[s] = sq[s] - sums[s].iter().map(|x| x * x / n[s]).sum::<f64>();
    }
    out[2] = n[0] * n[1] / (n[0] + n[1])
        * (0..128)
            .map(|k| (sums[0][k] / n[0] - sums[1][k] / n[1]).powi(2))
            .sum::<f64>();
    out
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    require(
        args.len() == 2,
        "usage: verify OUTPUT_DIRECTORY TASK199_REPORT",
    )?;
    let root = Path::new(&args[0]);
    let report = read(&root.join("report.json"))?;
    // Original task #200 metadata predates explicit peak recording.
    require(
        report["peak_lr"].as_f64().unwrap_or(0.001) == schedule::PEAK,
        "verifier compiled for another learning-rate arm",
    )?;
    let baseline = read(Path::new(&args[1]))?;
    require(
        report["status"] == "pass" && report["failed"] == 0,
        "cohort did not pass",
    )?;
    for (path, digest) in report["inputs"]["files"]
        .as_object()
        .ok_or("missing input ledger")?
    {
        require(
            hash(&fs::read(path)?) == digest.as_str().unwrap(),
            "input hash changed",
        )?;
    }
    require(
        report["replay_hashes"]
            .as_array()
            .ok_or("missing replays")?
            .len()
            == 2,
        "missing original replays",
    )?;
    algebra::threads();
    let mut update_checks = 0;
    let mut geometry_checks = 0;
    let mut query_checks = 0;
    let mut baseline_checks = 0;
    let seeds = report["seeds"].as_array().ok_or("missing seeds")?;
    require(seeds.len() == 8, "missing declared seeds")?;
    for (row, &expected_seed) in seeds.iter().zip(&[7u64, 11, 1, 3, 5, 13, 17, 19]) {
        require(
            row["seed"] == expected_seed && row["status"] == "pass" && row["completed"] == 10000,
            "incomplete seed",
        )?;
        let path = root.join(format!("seed{expected_seed}"));
        let paired = baseline["seeds"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["seed"] == expected_seed)
            .ok_or("missing paired baseline")?;
        for key in [
            "initial_sha256",
            "train_sha256",
            "validation_sha256",
            "order_sha256",
        ] {
            require(row[key] == paired[key], "paired inputs changed")?;
        }
        require(
            read(&path.join("result.json"))? == *row,
            "seed report mismatch",
        )?;
        let mut active = vec![false; 97];
        for c in row["partition"]["active"]
            .as_array()
            .ok_or("missing support")?
        {
            active[c.as_u64().unwrap() as usize] = true;
        }
        let mut expected = vec![false; 97];
        for s in 0u32..97 {
            expected[(s.pow(3) % 97) as usize] = true;
        }
        require(active == expected, "wrong task label support")?;
        let geometry = row["geometry"].as_array().ok_or("missing geometry")?;
        require(geometry.len() == 31, "missing geometry slots")?;
        let updates_bytes = fs::read(path.join("updates.jsonl"))?;
        require(
            hash(&updates_bytes) == row["updates_sha256"],
            "update table hash mismatch",
        )?;
        let mut updates = BufReader::new(updates_bytes.as_slice()).lines();
        let mut trace = BufReader::new(File::open(path.join("embeddings.bin"))?);
        let mut header = [0u8; 24];
        trace.read_exact(&mut header)?;
        require(
            [word(&header, 0), word(&header, 1), word(&header, 2)] == [1, 10001, 12416],
            "wrong embedding trace header",
        )?;
        let mut digest = Sha256::new();
        digest.update(header);
        let mut previous = vec![];
        let mut first = [0.; 3];
        let mut cumulative = [[0.; 6]; 3];
        let mut quad = [0.; 3];
        let mut increases = [0usize; 3];
        for step in 0..=10000 {
            let mut bytes = vec![0u8; 4 * 12416];
            trace.read_exact(&mut bytes)?;
            digest.update(&bytes);
            let e: Vec<_> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v) as f64)
                .collect();
            let now = energies(&e, &active);
            if step == 0 {
                first = now;
            } else {
                let r: Value = serde_json::from_str(&updates.next().ok_or("missing update row")??)?;
                require(r["step"] == step, "wrong update cadence")?;
                let before = energies(&previous, &active);
                let delta: Vec<_> = e.iter().zip(&previous).map(|(a, b)| a - b).collect();
                let delta_energy = energies(&delta, &active);
                let eta = number(&r["eta"])?;
                let alpha = number(&r["alpha"])?;
                require(
                    alpha > 0. && eta == schedule::Schedule::new(10000)?.rate(step - 1) as f64,
                    "invalid source scales",
                )?;
                let norm = number(&r["gradient_norm"])?;
                require(
                    norm >= 0. && near(number(&r["clip_scale"])?, 1. / norm.max(1.)),
                    "invalid clipping scale",
                )?;
                require(
                    number(&r["unused_ce_damping"])? >= -number(&r["damping_tolerance"])?,
                    "CE damping allowance failed",
                )?;
                require(
                    number(&r["rounding_max"])? <= number(&r["rounding_allowance"])?,
                    "parameter rounding allowance failed",
                )?;
                for i in 0..3 {
                    require(
                        near(before[i], number(&r["before"][i])?)
                            && near(now[i], number(&r["after"][i])?)
                            && near(delta_energy[i], number(&r["quadratic"][i])?),
                        "independent class energy mismatch",
                    )?;
                    let terms = r["radial"][i].as_array().ok_or("missing radial terms")?;
                    require(terms.len() == 6, "missing component")?;
                    let mut sum = 0.;
                    for (j, term) in terms.iter().enumerate() {
                        let value = number(term)?;
                        sum += value;
                        cumulative[i][j] += value;
                    }
                    quad[i] += delta_energy[i];
                    require(
                        near(sum + delta_energy[i], now[i] - before[i]),
                        "finite energy identity mismatch",
                    )?;
                    require(
                        near(
                            cumulative[i].iter().sum::<f64>() + quad[i],
                            now[i] - first[i],
                        ),
                        "prefix energy identity mismatch",
                    )?;
                    require(
                        near(number(&terms[0])?, -2. * eta * 0.1 * before[i]),
                        "decay attribution mismatch",
                    )?;
                    increases[i] += usize::from(
                        number(&r["after"][i])?
                            > number(&r["before"][i])?
                                + 1e-8
                                    * (1. + number(&r["after"][i])?.max(number(&r["before"][i])?)),
                    );
                }
                require(
                    near(
                        number(&r["radial"][1][1])?,
                        -2. * eta * alpha * number(&r["unused_ce_damping"])?,
                    ),
                    "isotropic damping attribution mismatch",
                )?;
                update_checks += 1;
            }
            if let Some(probe) = geometry.iter().find(|g| g["step"] == step) {
                query_checks += verify_probe(&path, probe, &bytes, &now)?;
                geometry_checks += 1;
                if step == 0 {
                    let old = &paired["geometry"][0];
                    require(
                        probe["state_sha256"] == old["state_sha256"]
                            && pairing::coverage_counts_equal(&probe["coverage"], &old["coverage"]),
                        "paired initial probe changed",
                    )?;
                    baseline_checks += 1;
                }
            }
            previous = e;
        }
        let mut extra = [0u8; 1];
        require(
            trace.read(&mut extra)? == 0 && updates.next().is_none(),
            "extra trace rows",
        )?;
        let trace_hash = digest
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        require(
            trace_hash == row["trace_sha256"],
            "embedding trace hash mismatch",
        )?;
        let final_energy = energies(&previous, &active);
        let summary = &row["dynamics"];
        require(
            summary["updates"] == 10000
                && summary["damping_failures"] == 0
                && row["instrumented_matches"] == 30,
            "incomplete source checks",
        )?;
        let checks = row["optimizer_checks"]
            .as_array()
            .ok_or("missing optimizer checks")?;
        require(
            checks.len() == 30 && row["paired_initial_matches"] == 1,
            "incomplete optimizer or pairing checks",
        )?;
        for (check, probe) in checks.iter().zip(geometry) {
            require(
                check["passed"] == true
                    && check["completed"] == probe["step"]
                    && number(&check["max_ulps"])? <= 4.
                    && number(&check["norm_relative_error"])? <= 2e-6,
                "optimizer reference allowance failed",
            )?;
        }
        for i in 0..3 {
            require(
                near(first[i], number(&summary["start"][i])?)
                    && near(final_energy[i], number(&summary["end"][i])?)
                    && near(quad[i], number(&summary["cumulative_quadratic"][i])?)
                    && summary["energy_increase_updates"][i] == increases[i],
                "cumulative summary mismatch",
            )?;
            for j in 0..6 {
                require(
                    near(
                        cumulative[i][j],
                        number(&summary["cumulative_radial"][i][j])?,
                    ),
                    "component total mismatch",
                )?;
            }
        }
        eprintln!("verified seed {expected_seed}");
    }
    require(baseline_checks == 8, "missing paired initial comparisons")?;
    println!(
        "{}",
        json!({"passed":true,"seeds":8,"updates":update_checks,"geometry_probes":geometry_checks,"margin_checks":query_checks,"paired_initial_matches":baseline_checks,"independent_model_sha256":hash(include_bytes!("../../../flat_approx/src/model.rs"))})
    );
    Ok(())
}
fn verify_probe(
    root: &Path,
    probe: &Value,
    embedding: &[u8],
    energies: &[f64; 3],
) -> Result<usize> {
    let step = probe["step"].as_u64().unwrap();
    let path = root.join(format!("step-{step:06}"));
    require(probe["status"] == "pass", "geometry failed")?;
    let state = fs::read(path.join("state.bin"))?;
    require(
        hash(&state) == probe["state_sha256"] && word(&state, 0) == step,
        "probe state hash mismatch",
    )?;
    require(
        state[8..8 + embedding.len()] == *embedding,
        "trace and probe embeddings differ",
    )?;
    let count = model::Arm::Plain.count();
    require(state.len() == 8 + 12 * count, "invalid state length")?;
    let parameters: Vec<_> = state[8..8 + 4 * count]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v) as f64)
        .collect();
    let rows: Vec<u32> = (0..97)
        .flat_map(|a| (0..97).flat_map(move |b| [a, b, 0]))
        .collect();
    let independent = model::forward(model::Arm::Plain, &parameters, &rows);
    let frozen = read(&path.join("frozen.json"))?;
    let bytes = fs::read(path.join("margins.bin"))?;
    require(
        hash(&bytes) == probe["margins_sha256"] && hash(&bytes) == frozen["margins_sha256"],
        "margin binary hash mismatch",
    )?;
    require(
        bytes.len() == 8 * (4 + 9409 + 8 * 9409)
            && [
                word(&bytes, 0),
                word(&bytes, 1),
                word(&bytes, 2),
                word(&bytes, 3),
            ] == [1, 9409, 2, 4],
        "invalid margin format",
    )?;
    for (i, e) in energies.iter().enumerate() {
        require(
            near(*e, number(&probe["structural"]["energies"][i])?),
            "probe/trace structural energy mismatch",
        )?;
    }
    let records = read(&path.join("queries.json"))?;
    let records = records.as_array().ok_or("missing queries")?;
    require(records.len() == 9409, "missing query rows")?;
    let mut counts = [[0usize; 8]; 2];
    let mut correct = [[0usize; 8]; 2];
    let mut certified = [[0usize; 8]; 2];
    for (q, row) in records.iter().enumerate() {
        let target = (((q / 97 + q % 97) as u32).pow(3) % 97) as usize;
        require(
            row[0] == q && row[1] == target && word(&bytes, 4 + q) == target as u64,
            "candidate route mismatch",
        )?;
        let split = row[2].as_u64().ok_or("missing split")? as usize;
        require(split < 2, "invalid split")?;
        let exact = independent.logits[q * 97 + target]
            - (0..97)
                .filter(|&c| c != target)
                .map(|c| independent.logits[q * 97 + c])
                .fold(f64::NEG_INFINITY, f64::max);
        let observed = number(&row[4])?;
        let is_correct = row[3] == target;
        for m in 0..8 {
            let lower = scalar(&bytes, 4 + 9409 + m * 9409 + q);
            require(
                lower.is_finite()
                    && lower <= exact + 1e-8 * (1. + lower.abs().max(exact.abs()))
                    && lower <= observed + 1e-8 * (1. + lower.abs().max(observed.abs())),
                "independent margin enclosure failed",
            )?;
            let pass = lower > 1e-8 * (1. + lower.abs());
            require(!pass || is_correct, "false certificate")?;
            counts[split][m] += 1;
            correct[split][m] += usize::from(is_correct);
            certified[split][m] += usize::from(pass);
        }
        require(
            scalar(&bytes, 4 + 9409 + q) == scalar(&bytes, 4 + 9409 + 4 * 9409 + q),
            "rank-zero controls differ",
        )?;
    }
    for (s, name) in ["train", "validation"].into_iter().enumerate() {
        for m in 0..8 {
            let r = &probe["coverage"][name][m];
            require(
                r["count"] == counts[s][m]
                    && r["correct"] == correct[s][m]
                    && r["certified"] == certified[s][m],
                "coverage summary mismatch",
            )?;
        }
    }
    Ok(8 * 9409)
}
