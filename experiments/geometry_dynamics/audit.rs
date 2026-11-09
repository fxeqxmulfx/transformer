//! Read the emitted evidence afresh and independently recompute FP64 margins.
//! This verifier does not call the basis constructor or use its coefficients.
use crate::{
    Result, SEEDS, algebra,
    evidence::{hash, require, scalar, word},
    model, steps,
};
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
pub fn run(output: &Path) -> Result<()> {
    let started = Instant::now();
    let report = read(&output.join("report.json"))?;
    require(
        report["status"] == "pass" && report["failed"] == 0 && report["declared_cases"] == 744,
        "incomplete support-complete cohort",
    )?;
    for (path, digest) in report["inputs"]["files"]
        .as_object()
        .ok_or("missing source ledger")?
    {
        require(
            hash(&fs::read(path)?) == digest.as_str().unwrap(),
            "source ledger changed",
        )?;
    }
    let cases = report["cases"].as_array().ok_or("missing cases")?;
    require(cases.len() == 744, "missing declared cases")?;
    algebra::threads();
    let rows: Vec<_> = (0..97)
        .flat_map(|a| (0..97).flat_map(move |b| [a, b, 0]))
        .collect();
    let mut slot = 0;
    let mut margin_checks = 0;
    for (ci, name) in ["lr001", "lr01", "settle"].into_iter().enumerate() {
        require(
            report["cohorts"][ci]["name"] == name,
            "cohort order changed",
        )?;
        let source = Path::new(
            report["cohorts"][ci]["path"]
                .as_str()
                .ok_or("missing source path")?,
        );
        for seed in SEEDS {
            for step in steps() {
                let case = &cases[slot];
                require(
                    case["cohort"] == name
                        && case["seed"] == seed
                        && case["step"] == step
                        && case["status"] == "pass",
                    "case order or status changed",
                )?;
                let path = output.join(format!("{name}-seed{seed}-step{step:06}"));
                let src = source.join(format!("seed{seed}/step-{step:06}"));
                let state = fs::read(src.join("state.bin"))?;
                let count = model::Arm::Plain.count();
                require(
                    state.len() == 8 + 12 * count
                        && word(&state, 0) == step
                        && hash(&state) == case["state_sha256"],
                    "state changed",
                )?;
                let p: Vec<_> = state[8..8 + 4 * count]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v) as f64)
                    .collect();
                let direct = model::forward(model::Arm::Plain, &p, &rows);
                let bounds = fs::read(path.join("margins.bin"))?;
                let legacy = fs::read(src.join("margins.bin"))?;
                let active = fs::read(
                    Path::new(
                        report["active_reference"]
                            .as_str()
                            .ok_or("missing active source")?,
                    )
                    .join(format!("{name}-seed{seed}-step{step:06}/margins.bin")),
                )?;
                let frozen = read(&path.join("frozen.json"))?;
                require(
                    bounds.len() == 8 * (4 + 9409 + 8 * 9409)
                        && [
                            word(&bounds, 0),
                            word(&bounds, 1),
                            word(&bounds, 2),
                            word(&bounds, 3),
                        ] == [1, 9409, 2, 4],
                    "wrong emitted bound shape",
                )?;
                require(
                    hash(&bounds) == case["margins_sha256"]
                        && hash(&bounds) == frozen["margins_sha256"]
                        && frozen["state_sha256"] == case["state_sha256"],
                    "frozen bound hash mismatch",
                )?;
                let outcomes = read(&src.join("queries.json"))?;
                let outcomes = outcomes.as_array().ok_or("missing source outcomes")?;
                require(outcomes.len() == 9409, "missing source outcome")?;
                let mut count = 0usize;
                let mut correct = 0usize;
                let mut certified = [0usize; 8];
                let mut gained = [0usize; 8];
                let mut lost = [0usize; 8];
                for (q, r) in outcomes.iter().enumerate() {
                    let t = word(&bounds, 4 + q) as usize;
                    require(t < 97 && r[0] == q && r[1] == t, "target route changed")?;
                    let exact = direct.logits[q * 97 + t]
                        - (0..97)
                            .filter(|&c| c != t)
                            .map(|c| direct.logits[q * 97 + c])
                            .fold(f64::NEG_INFINITY, f64::max);
                    let observed = r[4].as_f64().ok_or("missing GPU margin")?;
                    let is_correct = r[3] == t;
                    if r[2] == 1 {
                        count += 1;
                        correct += usize::from(is_correct);
                    }
                    for (m, cert) in certified.iter_mut().enumerate() {
                        let lower = scalar(&bounds, 4 + 9409 + m * 9409 + q);
                        let previous = scalar(&active, 4 + 9409 + (m % 4) * 9409 + q);
                        if m >= 4 {
                            let support = scalar(&bounds, 4 + 9409 + (m % 4) * 9409 + q);
                            let all = scalar(&legacy, 4 + 9409 + (m % 4 + 4) * 9409 + q);
                            require(
                                lower.to_bits() == support.max(previous).max(all).to_bits(),
                                "envelope does not preserve its constituent bounds",
                            )?;
                        }
                        require(
                            lower.is_finite()
                                && lower <= exact + 1e-8 * (1. + lower.abs().max(exact.abs()))
                                && lower
                                    <= observed + 1e-8 * (1. + lower.abs().max(observed.abs())),
                            "independent emitted margin failed",
                        )?;
                        let pass = lower > 1e-8 * (1. + lower.abs());
                        require(!pass || is_correct, "false emitted certificate")?;
                        if r[2] == 1 {
                            *cert += usize::from(pass);
                            let old_pass = previous > 1e-8 * (1. + previous.abs());
                            gained[m] += usize::from(pass && !old_pass);
                            lost[m] += usize::from(!pass && old_pass);
                        }
                        margin_checks += 1;
                    }
                }
                require(count == 4705, "validation inventory changed")?;
                for (m, cert) in certified.into_iter().enumerate() {
                    let c = &case["coverage"][m];
                    require(
                        c["rank"] == [0, 32, 33, 128][m % 4]
                            && c["family"] == if m < 4 { "support" } else { "envelope" }
                            && c["count"] == count
                            && c["correct"] == correct
                            && c["certified"] == cert
                            && c["newly_certified"] == gained[m]
                            && c["lost_certificates"] == lost[m],
                        "emitted summary mismatch",
                    )?;
                }
                slot += 1;
            }
            eprintln!("verified support cohort={name} seed={seed}");
        }
    }
    let total = report["seconds"].as_f64().ok_or("missing producer clock")?
        + started.elapsed().as_secs_f64();
    require(
        total <= 1800.,
        "postprocessing and verification exceeded frozen cap",
    )?;
    println!(
        "{}",
        json!({"passed":true,"cases":slot,"margin_checks":margin_checks,"false_certificates":0,"seconds":started.elapsed().as_secs_f64(),"combined_postprocessing_seconds":total})
    );
    Ok(())
}
