//! Current-state geometry, frozen without query labels, then source audit.
use crate::{
    algebra::linear,
    application::State,
    basis_svd,
    classes::{Partition, energy},
    domain::{
        Result,
        data::Data,
        model::{Arm, E, W},
    },
    groups,
    infrastructure::{engine::Engine, storage},
    inputs::require,
    measure, model,
    reconstruction::{Family, RANKS},
};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};
pub fn run(
    engine: &Engine,
    state: &State,
    data: &Data,
    part: &Partition,
    step: usize,
    path: &Path,
) -> Result<Value> {
    fs::create_dir(path)?;
    let state_bytes = storage::state_bytes(step, state);
    fs::write(path.join("state.bin"), &state_bytes)?;
    let p: Vec<_> = state.parameters.iter().map(|&v| v as f64).collect();
    let rows: Vec<u32> = (0..97)
        .flat_map(|a| (0..97).flat_map(move |b| [a, b, 0]))
        .collect();
    let actual = model::forward(model::Arm::Plain, &p, &rows);
    let inputs: Vec<_> = rows
        .as_chunks::<3>()
        .0
        .iter()
        .map(|r| (r[0] as usize, r[1] as usize))
        .collect();
    let training = measure::training(data);
    let parts = part.project(&p[..E]);
    let energies = parts.each_ref().map(|v| energy(v));
    let centered: Vec<_> = (0..E).map(|k| parts.iter().map(|v| v[k]).sum()).collect();
    let spectrum = basis_svd::thin(&centered, 97, W);
    let active = basis_svd::thin(&parts[0], 97, W);
    let rank = active
        .values
        .iter()
        .filter(|&&s| s > 1e-10 * active.values[0])
        .count()
        .min(32);
    let overlap = linear(
        &spectrum.right[..rank * W],
        rank,
        W,
        &active.right[..rank * W],
        rank,
    );
    let total = energy(&centered);
    require(
        (total - energies.iter().sum::<f64>()).abs() < 1e-8 * (1. + total),
        "class energy decomposition failed",
    )?;
    let structural = json!({"energies":energies,"centered_energy":total,"active_fraction":energies[0]/total,
        "unused_fraction":energies[1]/total,"between_fraction":energies[2]/total,
        "tail32":spectrum.values.iter().skip(32).map(|s|s*s).sum::<f64>()/total,
        "tail33":spectrum.values.iter().skip(33).map(|s|s*s).sum::<f64>()/total,
        "active_full_overlap":energy(&overlap)/rank as f64,"active_basis_rank":rank});
    let mut labels = vec![];
    let mut margins = vec![];
    let mut geometry = json!({});
    let contrasts: Vec<_> = (0..97)
        .flat_map(|t| {
            let p = &p;
            (0..97).map(move |c| {
                (0..W)
                    .map(|k| (p[c * W + k] - p[t * W + k]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
        })
        .collect();
    for family in Family::ALL {
        let g = groups::construct(&inputs, &actual.states, &p[..E], &training, W, 97, family);
        if labels.is_empty() {
            labels = g.labels.clone();
        } else {
            require(labels == g.labels, "family changed candidate labels")?;
        }
        require(
            g.labels.iter().all(Option::is_some),
            "missing training route",
        )?;
        for bounds in &g.upper {
            let mut margin = vec![0.; 9409];
            for q in 0..9409 {
                let t = g.labels[q].unwrap();
                let mut upper = f64::NEG_INFINITY;
                for c in 0..97 {
                    if c != t {
                        let norm = contrasts[t * 97 + c];
                        upper = upper.max(bounds[q * 97 + c] + norm * 1e-3 + 2e-3);
                    }
                }
                margin[q] = -upper;
            }
            margins.push(margin);
        }
        geometry[family.name()] = json!(g.groups);
    }
    let file = File::create_new(path.join("margins.bin"))?;
    let mut writer = BufWriter::new(file);
    for v in [1u64, 9409, 2, 4] {
        writer.write_all(&v.to_le_bytes())?;
    }
    for t in &labels {
        writer.write_all(&(t.unwrap() as u64).to_le_bytes())?;
    }
    for track in &margins {
        for x in track {
            writer.write_all(&x.to_le_bytes())?;
        }
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    let bounds_hash = storage::hash(&fs::read(path.join("margins.bin"))?);
    storage::json_write(
        &path.join("frozen.json"),
        &json!({"state_sha256":storage::hash(&state_bytes),"margins_sha256":bounds_hash,"ranks":RANKS,"geometry":geometry,"structural":structural}),
    )?;
    // First use of query labels/outcomes in this constructor is after freezing.
    let gpu = measure::snapshot(engine, data, false)?;
    let executed = linear(&gpu.features, 9409, W, &p[..E], 97);
    let mut true_labels = vec![0usize; 9409];
    let mut split = vec![1usize; 9409];
    for (name, source) in [(0, &data.train), (1, &data.validation)] {
        for r in source.as_chunks::<3>().0 {
            let q = (97 * r[0] + r[1]) as usize;
            split[q] = name;
            true_labels[q] = r[2] as usize;
        }
    }
    let mut feature_error = 0f64;
    let mut logit_error = 0f64;
    let mut violations = 0;
    let mut false_certificates = 0;
    let mut outcomes = Vec::with_capacity(9409);
    let mut count = [[0usize; 8]; 2];
    let mut correct = [[0usize; 8]; 2];
    let mut certified = [[0usize; 8]; 2];
    for q in 0..9409 {
        let t = labels[q].unwrap();
        require(t == true_labels[q], "training route has wrong candidate")?;
        feature_error = feature_error.max(
            (0..W)
                .map(|k| (actual.states[q * W + k] - gpu.features[q * W + k]).powi(2))
                .sum::<f64>()
                .sqrt(),
        );
        for (c, &v) in executed[q * 97..(q + 1) * 97].iter().enumerate() {
            logit_error = logit_error.max((v - gpu.logits[q * 97 + c]).abs());
        }
        let winner = (1..97).fold(0, |best, c| {
            if gpu.logits[q * 97 + c] > gpu.logits[q * 97 + best] {
                c
            } else {
                best
            }
        });
        let margin = gpu.logits[q * 97 + t]
            - (0..97)
                .filter(|&c| c != t)
                .map(|c| gpu.logits[q * 97 + c])
                .fold(f64::NEG_INFINITY, f64::max);
        let exact_margin = actual.logits[q * 97 + t]
            - (0..97)
                .filter(|&c| c != t)
                .map(|c| actual.logits[q * 97 + c])
                .fold(f64::NEG_INFINITY, f64::max);
        for (m, track) in margins.iter().enumerate() {
            let lower = track[q];
            let pass = lower > 1e-8 * (1. + lower.abs());
            violations += usize::from(lower > margin + 1e-8 * (1. + margin.abs().max(lower.abs())))
                + usize::from(
                    lower > exact_margin + 1e-8 * (1. + exact_margin.abs().max(lower.abs())),
                );
            false_certificates += usize::from(pass && winner != t);
            count[split[q]][m] += 1;
            correct[split[q]][m] += usize::from(winner == t);
            certified[split[q]][m] += usize::from(pass);
        }
        outcomes.push(json!([q, t, split[q], winner, margin]));
    }
    storage::json_write(&path.join("queries.json"), &json!(outcomes))?;
    let mut coverage = json!({});
    for (s, name) in ["train", "validation"].into_iter().enumerate() {
        coverage[name]=json!((0..8).map(|m|json!({"family":Family::ALL[m/4].name(),"rank":RANKS[m%4],"count":count[s][m],"correct":correct[s][m],"certified":certified[s][m],"coverage":certified[s][m] as f64/count[s][m] as f64})).collect::<Vec<_>>());
    }
    require(
        feature_error <= 1e-3 && logit_error <= 1e-3,
        "source numerical allowance failed",
    )?;
    Ok(
        json!({"step":step,"passed":violations==0&&false_certificates==0,"state_sha256":storage::hash(&state_bytes),"margins_sha256":bounds_hash,"metrics":gpu.metrics,"structural":structural,"coverage":coverage,"violations":violations,"false_certificates":false_certificates,"feature_error":feature_error,"logit_error":logit_error,"arm":Arm::Plain.name()}),
    )
}
