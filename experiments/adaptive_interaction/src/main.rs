//! AI1-AI5; checkpoint/oracle checks reused from residual_dynamics 420485e3.
#[path = "../../residual_dynamics/src/algebra.rs"]
mod algebra;
#[path = "../../flat_approx/src/data.rs"]
mod data;
mod diagnostic;
#[path = "../../residual_dynamics/src/geometry.rs"]
mod geometry;
#[path = "../../residual_dynamics/src/gradient.rs"]
mod gradient;
mod interaction;
#[path = "../../flat_approx/src/model.rs"]
mod model;
mod moments;
#[path = "../../residual_dynamics/src/svd.rs"]
mod svd;
// The frozen oracle retains its original imports and scalar implementation.
mod domain {
    pub mod model {
        pub use crate::model::{Arm, E, W};
    }
}
#[path = "../../flat_approx/fixtures/reference.rs"]
mod oracle;

use model::{Arm, P};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs,
    path::Path,
    time::{Duration, Instant},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn write(path: &Path, report: &Value) -> Result<()> {
    let mut b = serde_json::to_vec_pretty(report)?;
    b.push(b'\n');
    fs::write(path, b)?;
    Ok(())
}

fn source_hash() -> String {
    let files: &[&[u8]] = &[
        include_bytes!("main.rs"),
        include_bytes!("../../residual_dynamics/src/algebra.rs"),
        include_bytes!("../../flat_approx/src/data.rs"),
        include_bytes!("diagnostic.rs"),
        include_bytes!("../../residual_dynamics/src/geometry.rs"),
        include_bytes!("../../residual_dynamics/src/gradient.rs"),
        include_bytes!("moments.rs"),
        include_bytes!("interaction.rs"),
        include_bytes!("../../residual_dynamics/src/svd.rs"),
        include_bytes!("../../flat_approx/src/model.rs"),
        include_bytes!("../../flat_approx/fixtures/reference.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
    ];
    hash(&files.concat())
}

const ORIGINAL_SOURCE: &str = "14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da";
struct Saved {
    parameters: Vec<f64>,
    first: Vec<f64>,
    second: Vec<f64>,
    identity: Value,
    history: Value,
}

fn checkpoint(
    path: &Path,
    arm: Arm,
    seed: u64,
    step: usize,
    split: &data::Split,
    protocol_hash: &str,
) -> Result<Saved> {
    let meta = json(&path.join("checkpoint.json"))?;
    let bytes = fs::read(path.join("state.bin"))?;
    let history = fs::read(path.join("history.json"))?;
    let require = |condition: bool, what: &str| -> Result<()> {
        if condition {
            Ok(())
        } else {
            Err(format!("{}: {what}", path.display()).into())
        }
    };
    require(meta["state_sha256"] == hash(&bytes), "state hash mismatch")?;
    require(
        meta["history_sha256"] == hash(&history),
        "history hash mismatch",
    )?;
    require(
        meta["identity"]["train_sha256"] == hash(&split.bytes(true)),
        "training split mismatch",
    )?;
    require(
        meta["identity"]["validation_sha256"] == hash(&split.bytes(false)),
        "validation split mismatch",
    )?;
    require(
        meta["identity"]["arm"] == arm.name() && meta["identity"]["seed"] == seed,
        "case identity mismatch",
    )?;
    require(
        meta["step"] == step && bytes.len() == 8 + 12 * arm.count(),
        "step or state length mismatch",
    )?;
    require(
        u64::from_le_bytes(bytes[..8].try_into()?) as usize == step,
        "binary step mismatch",
    )?;
    require(
        meta["identity"]["stored_parameters"] == arm.count(),
        "parameter count mismatch",
    )?;
    require(
        meta["identity"]["source_sha256"] == ORIGINAL_SOURCE
            && meta["identity"]["protocol_sha256"] == protocol_hash,
        "original optimizer source/protocol mismatch",
    )?;
    let all: Vec<_> = bytes[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b) as f64)
        .collect();
    require(all.iter().all(|x| x.is_finite()), "nonfinite checkpoint")?;
    let count = arm.count();
    require(
        all[2 * count..].iter().all(|&v| v >= 0.),
        "negative second moment",
    )?;
    if step == 0 {
        require(
            all[count..].iter().all(|&v| v == 0.),
            "nonzero initial moment",
        )?;
    }
    Ok(Saved {
        parameters: all[..count].to_vec(),
        first: all[count..count + model::E].to_vec(),
        second: all[2 * count..2 * count + model::E].to_vec(),
        identity: meta,
        history: serde_json::from_slice(&history)?,
    })
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: adaptive-interaction-diagnostic CHECKPOINT_ROOT OUTPUT_JSON".into());
    }
    let root = Path::new(&args[0]);
    let output = Path::new(&args[1]);
    let original = json(&root.join("protocol.json"))?;
    if original["source_sha256"] != ORIGINAL_SOURCE
        || original["betas"] != json!([0.9, 0.98])
        || original["epsilon"] != 1e-8
        || original["learning_rate"] != 0.001
        || original["weight_decay"] != 1.0
        || original["warmup_updates"] != 10
        || original["optimizer"] != "coordinate AdamW"
    {
        return Err("original coordinate AdamW recipe mismatch".into());
    }
    let protocol_hash = hash(&serde_json::to_vec(&original)?);
    algebra::threads();
    let started = Instant::now();
    let rows = data::rows();
    let mut cases = Vec::new();
    let mut intervals = Vec::new();
    let mut masks = Vec::new();
    for seed in [7, 11] {
        let split = data::Split::new(seed);
        let (_, cert) = data::certificate(&split, seed);
        masks.push(cert);
        for arm in Arm::ALL {
            let mut previous = None;
            for step in [0, 5000, 10000] {
                if started.elapsed() > Duration::from_secs(300) {
                    return Err(
                        "five-minute diagnostic cap reached; completed cases remain in output"
                            .into(),
                    );
                }
                let path = root.join(format!("{}-seed{seed}/step-{step:06}", arm.name()));
                let saved_state = checkpoint(&path, arm, seed, step, &split, &protocol_hash)?;
                let p = &saved_state.parameters;
                let out = model::forward(arm, p, &rows);
                let normalization_error = out
                    .raw
                    .iter()
                    .zip(&out.logits)
                    .enumerate()
                    .map(|(i, (raw, logit))| (raw / out.denominators[i / P] - logit).abs())
                    .fold(0., f64::max);
                let normalization_scale = out.logits.iter().map(|v| v.abs()).fold(0., f64::max);
                if normalization_error > 1e-9 * (1. + normalization_scale) {
                    return Err("final RMS readout normalization mismatch".into());
                }
                let sample: Vec<usize> = (0..32).map(|i| i * (P * P - 1) / 31).collect();
                let sample_rows: Vec<_> = sample
                    .iter()
                    .flat_map(|&q| rows[q * 3..q * 3 + 3].iter().copied())
                    .collect();
                let reference = oracle::logits(arm, p, &sample_rows);
                let actual: Vec<_> = sample
                    .iter()
                    .flat_map(|&q| out.logits[q * P..(q + 1) * P].iter().copied())
                    .collect();
                let error = reference
                    .iter()
                    .zip(&actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f64::max);
                let tolerance = 1e-9 * (1. + reference.iter().map(|v| v.abs()).fold(0., f64::max));
                if error > tolerance {
                    return Err(format!(
                        "scalar oracle disagreement {} {seed} {step}: {error}",
                        arm.name()
                    )
                    .into());
                }
                let (snapshot, analysis) = diagnostic::analyze(&out, &split, &saved_state, step);
                if let Some((old_step, old)) = &previous {
                    intervals.push(json!({"arm":arm.name(),"seed":seed,"from_step":old_step,
                        "to_step":step,"diagnostic":interaction::compare(old,&snapshot)}));
                }
                previous = Some((step, snapshot));
                let saved = saved_state
                    .history
                    .as_array()
                    .ok_or("invalid history")?
                    .last()
                    .ok_or("empty history")?;
                let counts_match = analysis["train"]["source_correct"] == saved["train"]["correct"]
                    && analysis["validation"]["source_correct"] == saved["validation"]["correct"];
                if !counts_match {
                    return Err(
                        format!("stored correctness counts differ: {}", path.display()).into(),
                    );
                }
                eprintln!(
                    "{} seed={seed} step={step} validation={} training_ce={} gradient_energy={} elapsed={:.2}s",
                    arm.name(),
                    analysis["validation"]["source_correct"],
                    analysis["train"]["mean_ce"],
                    analysis["readout_gradient_energy"],
                    started.elapsed().as_secs_f64()
                );
                cases.push(json!({"arm":arm.name(),"seed":seed,"step":step,"checkpoint":saved_state.identity,
                "normalization_max_error":normalization_error,
                "scalar_oracle_max_error":error,"scalar_oracle_loss":oracle::loss(arm,p,&sample_rows),
                "stored_counts_match":counts_match,"stored_metrics":saved,"diagnostic":analysis}));
                write(
                    output,
                    &json!({"protocol":"PROTOCOL.md fixed 2026-11-08",
                    "protocol_sha256":hash(include_bytes!("../PROTOCOL.md")),"source_sha256":source_hash(),
                    "complete":false,"masks":masks,"cases":cases,"intervals":intervals}),
                )?;
            }
        }
    }
    if started.elapsed() > Duration::from_secs(300) {
        return Err("five-minute diagnostic cap reached before completion".into());
    }
    write(
        output,
        &json!({"protocol":"PROTOCOL.md fixed 2026-11-08",
        "protocol_sha256":hash(include_bytes!("../PROTOCOL.md")),"source_sha256":source_hash(),
        "complete":true,"masks":masks,
        "seconds":started.elapsed().as_secs_f64(),"cases":cases,"intervals":intervals}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_source_arms_agree_with_frozen_scalar_oracle() {
        algebra::threads();
        let rows = [0, 0, 0, 1, 2, 27, 96, 96, 89];
        for arm in Arm::ALL {
            let p: Vec<_> = (0..arm.count())
                .map(|i| ((i as f64 + 1.) * 0.173).sin() * 0.03)
                .collect();
            let out = model::forward(arm, &p, &rows);
            let expected = oracle::logits(arm, &p, &rows);
            assert!(
                out.logits
                    .iter()
                    .zip(&expected)
                    .all(|(x, y)| (x - y).abs() < 1e-10)
            );
            for (r, row) in out.raw.as_chunks::<P>().0.iter().enumerate() {
                for (c, x) in row.iter().enumerate() {
                    assert!((x / out.denominators[r] - out.logits[r * P + c]).abs() < 1e-10);
                }
            }
            assert!(oracle::loss(arm, &p, &rows).is_finite());
        }
    }
    #[test]
    fn frozen_splits_and_mask_certificates() {
        let expected = [(
            7,
            "7110dd2a7dcc4fe48a00d50587ce8b3518c48a51adbe9a144f5967f421219296",
            "95f17a8e9e169c44e1ebc5e3a49053c56c490a78a2fa250f26f735f9e056a513",
        )];
        for (seed, train, validation) in expected {
            let split = data::Split::new(seed);
            assert_eq!(hash(&split.bytes(true)), train);
            assert_eq!(hash(&split.bytes(false)), validation);
            let (slack, _) = data::certificate(&split, seed);
            assert!(slack > 0. && slack < 1.);
        }
    }
}
