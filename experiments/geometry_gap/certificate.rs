//! The ST/OG3 arbitrary-basis bound, with training-selected active readout rows.
//! The execution allowances and all-class maximum are copied from task #199.
use crate::{
    Result,
    evidence::{Ledger, hash, require, scalar, word, write},
    groups, model,
    reconstruction::{self, Family, RANKS},
};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::Path};
const N: usize = 9409;
const P: usize = 97;
const W: usize = 128;
fn margins(g: &groups::Geometry, embedding: &[f64]) -> Vec<Vec<f64>> {
    let norms: Vec<_> = (0..P)
        .flat_map(|t| {
            (0..P).map(move |c| {
                (0..W)
                    .map(|k| (embedding[c * W + k] - embedding[t * W + k]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
        })
        .collect();
    g.upper
        .iter()
        .map(|bounds| {
            (0..N)
                .map(|q| {
                    let t = g.labels[q].expect("missing training route");
                    -(0..P)
                        .filter(|&c| c != t)
                        .map(|c| bounds[q * P + c] + norms[t * P + c] * 1e-3 + 2e-3)
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .collect()
        })
        .collect()
}
pub fn run(
    root: &Path,
    probe: &Value,
    training: &[(usize, usize)],
    out: &Path,
    ledger: &mut Ledger,
) -> Result<Value> {
    let step = probe["step"].as_u64().ok_or("missing step")?;
    let source = root.join(format!("step-{step:06}"));
    let bytes = ledger.read(&source.join("state.bin"))?;
    let count = model::Arm::Plain.count();
    require(
        bytes.len() == 8 + 12 * count
            && word(&bytes, 0) == step
            && hash(&bytes) == probe["state_sha256"],
        "state source mismatch",
    )?;
    let parameters: Vec<_> = bytes[8..8 + 4 * count]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v) as f64)
        .collect();
    let rows: Vec<_> = (0..97)
        .flat_map(|a| (0..97).flat_map(move |b| [a, b, 0]))
        .collect();
    let actual = model::forward(model::Arm::Plain, &parameters, &rows);
    let features: Vec<_> = actual
        .states
        .as_chunks::<W>()
        .0
        .iter()
        .zip(&actual.denominators)
        .flat_map(|(r, d)| r.iter().map(move |v| v / d))
        .collect();
    let inputs: Vec<_> = (0..N).map(|q| (q / P, q % P)).collect();
    let active: Vec<_> = training
        .iter()
        .map(|&(_, t)| t)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    require(active.len() == 33, "observed class support changed")?;
    let old = groups::construct(
        &inputs,
        &features,
        &parameters[..model::E],
        training,
        W,
        P,
        Family::Readout,
    );
    let saved = ledger.read(&source.join("margins.bin"))?;
    require(
        saved.len() == 8 * (4 + N + 8 * N)
            && hash(&saved) == probe["margins_sha256"]
            && [
                word(&saved, 0),
                word(&saved, 1),
                word(&saved, 2),
                word(&saved, 3),
            ] == [1, N as u64, 2, 4],
        "saved margins changed",
    )?;
    let legacy = margins(&old, &parameters[..model::E]);
    for (m, track) in legacy.iter().enumerate() {
        for (q, &v) in track.iter().enumerate() {
            require(
                v.to_bits() == scalar(&saved, 4 + N + (m + 4) * N + q).to_bits(),
                "original readout margin replay is not byte exact",
            )?;
            require(
                old.labels[q].map(|t| t as u64) == Some(word(&saved, 4 + q)),
                "original training route changed",
            )?;
        }
    }
    let selected = groups::construct_with(
        &inputs,
        &features,
        &parameters[..model::E],
        training,
        W,
        P,
        Family::Readout,
        |p| reconstruction::evaluate_selected(p, &RANKS, &active),
    );
    require(
        selected.labels == old.labels,
        "active basis changed the candidate label",
    )?;
    let bounds = margins(&selected, &parameters[..model::E]);
    require(
        bounds.iter().flatten().all(|v| v.is_finite()),
        "nonfinite certificate",
    )?;
    fs::create_dir(out)?;
    let mut frozen = Vec::with_capacity(8 * (4 + N + 4 * N));
    for x in [1u64, N as u64, 1, 4] {
        frozen.extend(x.to_le_bytes());
    }
    for t in &selected.labels {
        frozen.extend((t.unwrap() as u64).to_le_bytes());
    }
    for track in &bounds {
        for v in track {
            frozen.extend(v.to_le_bytes());
        }
    }
    fs::write(out.join("margins.bin"), &frozen)?;
    write(
        &out.join("frozen.json"),
        &json!({"state_sha256":probe["state_sha256"],"margins_sha256":hash(&frozen),"active_classes":active,"ranks":RANKS,"geometry":selected.groups}),
    )?;
    // Bounds are now immutable. Only this stage reads query outcomes.
    let queries = ledger.json(&source.join("queries.json"))?;
    let queries = queries.as_array().ok_or("missing outcomes")?;
    require(queries.len() == N, "outcome inventory changed")?;
    let mut counts = [[0usize; 4]; 2];
    let mut correct = [[0usize; 4]; 2];
    let mut certified = [[0usize; 4]; 2];
    let mut violations = 0;
    let mut false_certificates = 0;
    let mut newly_certified = [0usize; 4];
    let mut lost_certificates = [0usize; 4];
    let mut training_mask = vec![false; N];
    for &(q, _) in training {
        training_mask[q] = true;
    }
    for (q, row) in queries.iter().enumerate() {
        let t = selected.labels[q].unwrap();
        require(row[0] == q && row[1] == t, "route/observed target mismatch")?;
        let split = row[2].as_u64().ok_or("missing split")? as usize;
        require(split < 2, "invalid split")?;
        require((split == 0) == training_mask[q], "training split changed")?;
        let exact = actual.logits[q * P + t]
            - (0..P)
                .filter(|&c| c != t)
                .map(|c| actual.logits[q * P + c])
                .fold(f64::NEG_INFINITY, f64::max);
        let observed = row[4].as_f64().ok_or("missing GPU margin")?;
        let is_correct = row[3] == t;
        for m in 0..4 {
            let lower = bounds[m][q];
            violations += usize::from(lower > exact + 1e-8 * (1. + lower.abs().max(exact.abs())))
                + usize::from(lower > observed + 1e-8 * (1. + lower.abs().max(observed.abs())));
            let pass = lower > 1e-8 * (1. + lower.abs());
            let old_pass = legacy[m][q] > 1e-8 * (1. + legacy[m][q].abs());
            false_certificates += usize::from(pass && !is_correct);
            counts[split][m] += 1;
            correct[split][m] += usize::from(is_correct);
            certified[split][m] += usize::from(pass);
            if split == 1 {
                newly_certified[m] += usize::from(pass && !old_pass);
                lost_certificates[m] += usize::from(!pass && old_pass);
            }
        }
    }
    let coverage:Vec<_>=(0..4).map(|m|json!({"rank":RANKS[m],"count":counts[1][m],"correct":correct[1][m],"certified":certified[1][m],"coverage":certified[1][m] as f64/counts[1][m] as f64,"newly_certified":newly_certified[m],"lost_certificates":lost_certificates[m]})).collect();
    let conditioning: Vec<_> = selected
        .groups
        .iter()
        .flat_map(|g| g["folds"].as_array().unwrap())
        .filter_map(|f| f["geometry"].as_object().map(|_| &f["geometry"]))
        .collect();
    let ranks: Vec<_> = conditioning
        .iter()
        .flat_map(|g| g["ranks"].as_array().unwrap())
        .filter(|r| r["requested_rank"] == 32)
        .collect();
    let result = json!({"status":if violations==0 && false_certificates==0{"pass"}else{"fail"},"coverage":coverage,"legacy_coverage":probe["coverage"]["validation"],"accuracy":probe["metrics"]["validation"]["accuracy"],"margin_checks":4*N,"legacy_exact_matches":4*N,"false_certificates":false_certificates,"violations":violations,"margins_sha256":hash(&frozen),"state_sha256":probe["state_sha256"],"conditioning32":{"min_rank":ranks.iter().map(|r|r["achieved_rank"].as_u64().unwrap()).min(),"max_rank":ranks.iter().map(|r|r["achieved_rank"].as_u64().unwrap()).max(),"max_coefficient_norm":ranks.iter().map(|r|r["max_coefficient_norm"].as_f64().unwrap()).fold(0.,f64::max),"max_orthogonality_penalty":ranks.iter().map(|r|r["max_orthogonality_penalty"].as_f64().unwrap()).fold(0.,f64::max),"smallest_training_singular":conditioning.iter().map(|g|g["smallest_retained_training_singular_value"].as_f64().unwrap()).fold(f64::INFINITY,f64::min)}});
    write(&out.join("result.json"), &result)?;
    Ok(result)
}
