//! Capture actual GPU states and full-training gradients without changing parameters.
use crate::{
    domain::{
        Result,
        data::{Data, Random},
        model::{Arm, E, W},
    },
    geometry, gradient,
    infrastructure::engine::Engine,
    inputs::require,
    oracle,
};
use serde_json::{Value, json};

pub struct Snapshot {
    pub features: Vec<f64>,
    pub logits: Vec<f64>,
    pub readout: Vec<f64>,
    pub full: Vec<f64>,
    pub metrics: Value,
    pub rms_error: f64,
}
pub fn training(data: &Data) -> Vec<(usize, usize)> {
    data.train
        .as_chunks::<3>()
        .0
        .iter()
        .map(|r| ((97 * r[0] + r[1]) as usize, r[2] as usize))
        .collect()
}
pub fn snapshot(engine: &Engine, data: &Data, gradients: bool) -> Result<Snapshot> {
    let mut s = Snapshot {
        features: vec![0.; 9409 * W],
        logits: vec![0.; 9409 * 97],
        readout: vec![0.; E],
        full: vec![0.; E],
        metrics: json!({}),
        rms_error: 0.,
    };
    for (validation, rows, name) in [
        (false, &data.train, "train"),
        (true, &data.validation, "validation"),
    ] {
        let n = rows.len() / 3;
        let (mut loss, mut correct) = (0., 0);
        for offset in (0..n).step_by(512) {
            let batch = (n - offset).min(512);
            engine.forward(validation, offset, batch, false)?;
            let (raw, x) = engine.audit_features(batch)?;
            let logits = engine.logits.read()?;
            let metrics = engine.metrics(batch)?;
            loss += metrics.loss * batch as f64;
            correct += metrics.correct;
            for i in 0..batch {
                let row = &rows[(offset + i) * 3..][..3];
                let at = (97 * row[0] + row[1]) as usize;
                s.features[at * W..(at + 1) * W].copy_from_slice(&x[i * W..(i + 1) * W]);
                let d = (gradient::energy(&raw[i * W..(i + 1) * W]) / W as f64
                    + f32::EPSILON as f64)
                    .sqrt();
                for k in 0..W {
                    s.rms_error = s.rms_error.max((raw[i * W + k] / d - x[i * W + k]).abs());
                }
                for c in 0..97 {
                    s.logits[at * 97 + c] = logits[i * 97 + c] as f64;
                }
            }
            if gradients && !validation {
                let (readout, full) = engine.audit_gradient(batch)?;
                for k in 0..E {
                    s.readout[k] += readout[k] * batch as f64 / n as f64;
                    s.full[k] += full[k] * batch as f64 / n as f64;
                }
            }
        }
        s.metrics[name] = json!({"count":n,"correct":correct,"loss":loss/n as f64,
            "accuracy":correct as f64/n as f64});
    }
    require(
        s.features
            .iter()
            .chain(&s.logits)
            .chain(&s.full)
            .all(|x| x.is_finite()),
        "nonfinite GPU snapshot",
    )?;
    require(s.rms_error < 5e-5, "final RMS mismatch")?;
    Ok(s)
}

pub fn validate(engine: &Engine, arm: Arm, parameters: &[f32], data: &Data) -> Result<Value> {
    let p: Vec<_> = parameters.iter().map(|&x| x as f64).collect();
    let rows = &data.train[..24];
    engine.forward(false, 0, 8, false)?;
    let (_, x) = engine.audit_features(8)?;
    let logits = engine.logits.read()?;
    let reference = oracle::logits(arm, &p, rows);
    let forward_error = logits
        .iter()
        .zip(&reference)
        .map(|(&a, b)| (a as f64 - b).abs())
        .fold(0., f64::max);
    let forward_scale = reference.iter().map(|v| v.abs()).fold(0., f64::max);
    require(
        forward_error <= 5e-5 * (1. + forward_scale),
        "scalar forward oracle mismatch",
    )?;
    let (readout, full) = engine.audit_gradient(8)?;
    let labels: Vec<_> = rows
        .as_chunks::<3>()
        .0
        .iter()
        .map(|r| r[2] as usize)
        .collect();
    let (_, g) = gradient::loss_gradient(&x, &labels, &p[..E], W, 97);
    let readout_error = readout
        .iter()
        .zip(&g)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let readout_scale = g.iter().map(|v| v.abs()).fold(0., f64::max);
    require(
        readout_error <= 2e-6 + 2e-4 * readout_scale,
        "readout derivative mismatch",
    )?;
    let mut checks = Vec::new();
    for seed in [313, 911] {
        let mut rng = Random(seed);
        let mut direction: Vec<_> = (0..E).map(|_| 2. * rng.unit() - 1.).collect();
        let norm = gradient::energy(&direction).sqrt();
        direction.iter_mut().for_each(|d| *d /= norm);
        let actual: f64 = full.iter().zip(&direction).map(|(g, d)| g * d).sum();
        let mut plus = p.clone();
        let mut minus = p.clone();
        for k in 0..E {
            plus[k] += 1e-5 * direction[k];
            minus[k] -= 1e-5 * direction[k];
        }
        let finite = (oracle::loss(arm, &plus, rows) - oracle::loss(arm, &minus, rows)) / 2e-5;
        let tolerance = 2e-5 + 2e-4 * finite.abs();
        require(
            (actual - finite).abs() <= tolerance,
            "full tied embedding derivative mismatch",
        )?;
        checks.push(
            json!({"direction_seed":seed,"gpu":actual,"finite_difference":finite,
            "error":(actual-finite).abs(),"tolerance":tolerance}),
        );
    }
    Ok(
        json!({"scalar_forward_max_error":forward_error,"readout_max_error":readout_error,
        "full_embedding_directions":checks,"passed":true}),
    )
}

pub fn groups(snapshot: &Snapshot, data: &Data) -> [Vec<geometry::Group>; 2] {
    let training = training(data);
    [
        geometry::groups(&snapshot.features, &training, W, false),
        geometry::groups(&snapshot.features, &training, W, true),
    ]
}
