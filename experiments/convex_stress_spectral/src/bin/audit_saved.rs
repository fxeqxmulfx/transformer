//! Independent direct-logit CE and phase-fit audit of saved pair models.
//! Training is already complete: held-out labels are used only in diagnostics.
use convex_stress_spectral::{C, Model, Result, data, hash, linalg};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn audit(out: &Path, expected_fingerprint: &str) -> Result<Value> {
    let report: Value = serde_json::from_slice(&fs::read(out.join("result.json"))?)?;
    if report["source_fingerprint"] != expected_fingerprint {
        return Err("unexpected producer fingerprint".into());
    }
    let suite: Value = serde_json::from_str(include_str!("../../../convex_stress/suite.json"))?;
    let row = suite["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == report["task"])
        .ok_or("unknown task")?;
    let task = data::build(row)?;
    let bytes = fs::read(out.join("model.json"))?;
    if report["model_sha256"] != hash(&bytes)
        || report["data_fingerprints"] != row["data_fingerprints"]
    {
        return Err("model or data fingerprint mismatch".into());
    }
    let model: Model = serde_json::from_slice(&bytes)?;
    let n = model.vocab;
    if n != task.vocab || model.entries.len() != n * n {
        return Err("dimension mismatch".into());
    }
    let mut target = vec![C::default(); n * n];
    let mut cells = std::collections::BTreeSet::new();
    let mut reports = serde_json::Map::new();
    for (name, examples) in &task.splits {
        let (mut ce, mut error, mut modulus, mut count, mut correct) =
            (0_f64, 0_f64, 0_f64, 0_usize, 0_usize);
        for e in examples {
            if e.tokens.len() != 2 || e.targets.len() != 2 || e.targets[0] >= 0 || e.targets[1] < 0
            {
                return Err("audit requires frozen pair data".into());
            }
            let i = e.tokens[1] as usize;
            let j = e.tokens[0] as usize;
            let y = e.targets[1] as usize;
            if i >= n || j >= n || y >= n || !cells.insert(i + n * j) {
                return Err("invalid or repeated cell".into());
            }
            let z = model.entries[i + n * j];
            let mut scores = Vec::with_capacity(n);
            for c in 0..n {
                let angle = std::f64::consts::TAU * c as f64 / n as f64;
                // Deliberately avoid Model::logits, codes(), loss() and Model::cell.
                scores.push(angle.cos() * z.re + angle.sin() * z.im);
            }
            let predicted = (0..n)
                .max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(b.cmp(&a)))
                .unwrap();
            ce += scores.iter().map(|x| x.exp()).sum::<f64>().ln() - scores[y];
            correct += usize::from(predicted == y);
            count += 1;
            let angle = std::f64::consts::TAU * y as f64 / n as f64;
            let truth = C::new(angle.cos(), angle.sin());
            error = error.max((z - truth).norm());
            modulus += z.norm();
            target[i + n * j] = truth;
        }
        ce /= count as f64;
        let saved = &report["metrics"][name];
        if (ce - saved["loss"].as_f64().ok_or("no CE")?).abs() > 1e-11
            || saved["correct"] != correct
            || saved["targets"] != count
        {
            return Err(format!("independent CE/prediction mismatch on {name}").into());
        }
        reports.insert(
            name.clone(),
            json!({"ce":ce,"correct":correct,"count":count,
            "max_target_phase_error":error,"mean_entry_modulus":modulus/count as f64}),
        );
    }
    let (entry, op) = model.validate()?;
    let target_opnorm = linalg::svd(&target, n)?.values[0];
    Ok(
        json!({"audited":true,"task":report["task"],"arm":report["arm"],"radius":report["radius_name"],
        "producer_fingerprint":expected_fingerprint,"metrics":reports,"max_entry_norm":entry,"operator_norm":op,
        "postrun_target_operator_norm":target_opnorm,
        "target_scope":"Known train+validation labels only; unrepresented input cells set to zero. Used after training, never to choose a radius."}),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: audit_saved OUT EXPECTED_PRODUCER_FINGERPRINT".into());
    }
    println!("{}", audit(Path::new(&args[0]), &args[1])?);
    Ok(())
}
