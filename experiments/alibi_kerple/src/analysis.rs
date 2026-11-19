//! `analyze ROOT CONTROL_ROOT`: the comparisons of `PROTOCOL.md` § Analysis,
//! as JSON on standard output, from the cells' `result.json` files alone.
//! CONTROL_ROOT is the recorded control arm (gpt's `phase-score-lerope`).
//!
//! 1. `tiers`: per arm and tier, the cells that passed, ran out of budget,
//!    failed to run or are missing.
//! 2. `tasks`: per task and arm, the epochs to first99 per seed and their
//!    median (a cell that did not pass counts as infinite), and its ratio to
//!    the control's median.
//! 3. `splits`: per task with length or position-shift splits, per arm, the
//!    exact sequences summed over the seeds that finished.
//! 4. `kerple`: per KERPLE arm, every head's final (r1, r2), and the α of the
//!    least-squares fit b(r) ≈ −α ln(1 + r) over the trained distances with
//!    the fraction of b's energy the fit leaves.
use crate::{Result, runner};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn result(dir: &Path) -> Option<Value> {
    serde_json::from_slice(&fs::read(dir.join("result.json")).ok()?).ok()
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    match values.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => values[n / 2],
        n => (values[n / 2 - 1] + values[n / 2]) / 2.,
    }
}
/// A JSON number, or null for an infinite or undefined value.
fn finite(v: f64) -> Value {
    if v.is_finite() { json!(v) } else { Value::Null }
}

pub fn run(root: &Path, control: &Path) -> Result<()> {
    let manifest = runner::suite();
    let tasks = manifest["tasks"].as_array().ok_or("suite tasks")?;
    let mut arms = vec![("control", control.to_path_buf())];
    arms.extend(runner::ARMS.iter().map(|&a| (a, root.join(a))));
    let cell = |arm: &Path, task: &str, seed: u32| arm.join(task).join(format!("seed{seed}"));

    let mut tiers = serde_json::Map::new();
    for (name, dir) in &arms {
        let mut row = serde_json::Map::new();
        for tier in ["short", "long"] {
            let (mut passed, mut exhausted, mut errors, mut missing, mut other) = (0, 0, 0, 0, 0);
            for task in tasks.iter().filter(|t| t["tier"] == tier) {
                for seed in 0..3 {
                    let at = cell(dir, task["name"].as_str().unwrap_or(""), seed);
                    match result(&at) {
                        Some(r) if r["status"] == "passed" => passed += 1,
                        Some(r) if r["status"] == "budget_exhausted" => exhausted += 1,
                        Some(_) => other += 1,
                        None if at.join("error.json").exists() => errors += 1,
                        None => missing += 1,
                    }
                }
            }
            row.insert(
                tier.into(),
                json!({"passed": passed, "budget_exhausted": exhausted, "execution_errors": errors,
                    "other": other, "missing": missing}),
            );
        }
        tiers.insert((*name).into(), row.into());
    }

    let mut task_rows = Vec::new();
    let mut split_rows = Vec::new();
    for task in tasks {
        let name = task["name"].as_str().unwrap_or("");
        let mut per_arm = serde_json::Map::new();
        let mut control_median = f64::NAN;
        let mut splits = serde_json::Map::new();
        for (arm, dir) in &arms {
            let results: Vec<Option<Value>> = (0..3).map(|s| result(&cell(dir, name, s))).collect();
            let epochs: Vec<Value> = results
                .iter()
                .map(|r| {
                    r.as_ref()
                        .map_or(Value::Null, |r| r["first99_epochs"].clone())
                })
                .collect();
            let finished = results.iter().filter(|r| r.is_some()).count();
            let m = if finished == 3 {
                median(
                    epochs
                        .iter()
                        .map(|e| e.as_f64().unwrap_or(f64::INFINITY))
                        .collect(),
                )
            } else {
                f64::NAN
            };
            if *arm == "control" {
                control_median = m;
            }
            per_arm.insert(
                (*arm).into(),
                json!({"first99_epochs": epochs, "median": finite(m),
                    "median_is_infinite": m == f64::INFINITY,
                    "ratio_to_control": finite(m / control_median)}),
            );
            let mut sums = serde_json::Map::new();
            for r in results.iter().flatten() {
                if let Some(Value::Object(final_splits)) = r.get("final_splits") {
                    for (split, v) in final_splits {
                        if !(split.contains("length") || split.contains("shift")) {
                            continue;
                        }
                        let entry = sums.entry(split.clone()).or_insert(json!([0, 0, 0]));
                        entry[0] = json!(
                            entry[0].as_u64().unwrap_or(0) + v["exact"].as_u64().unwrap_or(0)
                        );
                        entry[1] = json!(
                            entry[1].as_u64().unwrap_or(0) + v["examples"].as_u64().unwrap_or(0)
                        );
                        entry[2] = json!(entry[2].as_u64().unwrap_or(0) + 1);
                    }
                }
            }
            if !sums.is_empty() {
                splits.insert((*arm).into(), sums.into());
            }
        }
        task_rows.push(json!({"task": name, "tier": task["tier"], "arms": per_arm}));
        if !splits.is_empty() {
            split_rows.push(json!({"task": name,
                "columns": ["exact", "examples", "seeds"], "arms": splits}));
        }
    }

    let mut kerple = serde_json::Map::new();
    for arm in ["kerple-log", "kerple-power"] {
        let power = arm == "kerple-power";
        let mut heads = Vec::new();
        let (mut r1s, mut r2s, mut alphas) = (Vec::new(), Vec::new(), Vec::new());
        for task in tasks {
            let name = task["name"].as_str().unwrap_or("");
            for seed in 0..3 {
                let Some(r) = result(&cell(&root.join(arm), name, seed)) else {
                    continue;
                };
                let length = r["metadata"]["max_length"].as_u64().unwrap_or(1) as usize;
                let values: Vec<f64> = r["position_final_values"]
                    .as_array()
                    .map(|a| a.iter().filter_map(Value::as_f64).collect())
                    .unwrap_or_default();
                for (h, pair) in values.chunks(2).enumerate() {
                    let (r1, r2) = (pair[0], pair[1]);
                    let b = |x: f64| {
                        if power {
                            -r1 * x.powf(r2)
                        } else {
                            -r1 * (1. + r2 * x).ln()
                        }
                    };
                    let (mut bl, mut ll, mut bb) = (0., 0., 0.);
                    for x in (1..length).map(|x| x as f64) {
                        let l = (1. + x).ln();
                        bl += b(x) * l;
                        ll += l * l;
                        bb += b(x) * b(x);
                    }
                    let alpha = -bl / ll;
                    let residual = if bb > 0. {
                        1. - bl * bl / (ll * bb)
                    } else {
                        0.
                    };
                    r1s.push(r1);
                    r2s.push(r2);
                    alphas.push(alpha);
                    heads.push(
                        json!({"task": name, "seed": seed, "head": h, "status": r["status"],
                        "r1": r1, "r2": r2, "trained_length": length,
                        "log_fit_alpha": alpha, "log_fit_residual_fraction": residual}),
                    );
                }
            }
        }
        let quartiles = |v: &Vec<f64>| {
            let mut s = v.clone();
            s.sort_by(f64::total_cmp);
            let q = |p: f64| {
                if s.is_empty() {
                    Value::Null
                } else {
                    json!(s[((s.len() - 1) as f64 * p).round() as usize])
                }
            };
            json!([q(0.25), q(0.5), q(0.75)])
        };
        kerple.insert(
            arm.into(),
            json!({"heads": heads.len(), "r1_quartiles": quartiles(&r1s),
                "r2_quartiles": quartiles(&r2s), "log_fit_alpha_quartiles": quartiles(&alphas),
                "per_head": heads}),
        );
    }

    let out = json!({"root": root, "control": control, "tiers": tiers, "tasks": task_rows,
        "splits": split_rows, "kerple": kerple});
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
