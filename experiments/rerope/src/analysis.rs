//! `analyze ROOT CONTROL_ROOT`: the comparisons of `PROTOCOL.md` § Analysis,
//! as JSON on standard output, from the cells' files alone. CONTROL_ROOT is
//! the recorded control arm (gpt's `phase-score-lerope`).
//!
//! 1. `tiers`: per arm and tier, the cells that passed, ran out of budget,
//!    failed to run or are missing.
//! 2. `tasks`: per task and arm, the epochs to first99 per seed and their
//!    median (a cell that did not pass counts as infinite), its ratio to the
//!    control's median, and the median seconds per update with its ratio to
//!    the control's. A task is `far` when one of its sequences is longer than
//!    the window, so that the arms differ from the control on it.
//! 3. `splits`: per task with length or position-shift splits, per arm, the
//!    exact sequences summed over the seeds that finished.
//! 4. `near`: every arm cell of a task that is not far against the recorded
//!    control cell, as `validate`'s replays compare them. The arms run the
//!    control's kernels there, so they must agree bit for bit.
//! 5. `cost`: per arm, the geometric mean over the far tasks and over the
//!    near ones of the ratio of seconds per update to the control's. The
//!    near tasks do the control's work, so their ratio is the drift of the
//!    machine between the cohorts.
use crate::{Result, runner, validation};
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
fn seconds_per_update(r: &Value) -> Option<f64> {
    Some(r["seconds"].as_f64()? / r["completed"].as_f64().filter(|&c| c > 0.)?)
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
    let mut near_cells = Vec::new();
    // Per arm: the logarithms of the far and near tasks' cost ratios.
    let mut costs = vec![(Vec::new(), Vec::new()); runner::ARMS.len()];
    for task in tasks {
        let name = task["name"].as_str().unwrap_or("");
        let max_length = result(&cell(control, name, 0))
            .and_then(|r| r["metadata"]["max_length"].as_u64())
            .ok_or_else(|| format!("{name}: no recorded control cell"))?
            as usize;
        let far = max_length > runner::WINDOW;
        let mut per_arm = serde_json::Map::new();
        let (mut control_median, mut control_cost) = (f64::NAN, f64::NAN);
        let mut splits = serde_json::Map::new();
        for (index, (arm, dir)) in arms.iter().enumerate() {
            let results: Vec<Option<Value>> = (0..3).map(|s| result(&cell(dir, name, s))).collect();
            let epochs: Vec<Value> = results
                .iter()
                .map(|r| {
                    r.as_ref()
                        .map_or(Value::Null, |r| r["first99_epochs"].clone())
                })
                .collect();
            let finished = results.iter().filter(|r| r.is_some()).count();
            let (m, cost) = if finished == 3 {
                (
                    median(
                        epochs
                            .iter()
                            .map(|e| e.as_f64().unwrap_or(f64::INFINITY))
                            .collect(),
                    ),
                    median(
                        results
                            .iter()
                            .flatten()
                            .map(|r| seconds_per_update(r).unwrap_or(f64::NAN))
                            .collect(),
                    ),
                )
            } else {
                (f64::NAN, f64::NAN)
            };
            if index == 0 {
                (control_median, control_cost) = (m, cost);
            } else if (cost / control_cost).is_finite() {
                let (far_costs, near_costs) = &mut costs[index - 1];
                if far { far_costs } else { near_costs }.push((cost / control_cost).ln());
            }
            per_arm.insert(
                (*arm).into(),
                json!({"first99_epochs": epochs, "median": finite(m),
                    "median_is_infinite": m == f64::INFINITY,
                    "ratio_to_control": finite(m / control_median),
                    "seconds_per_update": finite(cost),
                    "cost_ratio_to_control": finite(cost / control_cost)}),
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
            if index > 0 && !far {
                for seed in 0..3 {
                    let (ours, theirs) = (cell(dir, name, seed), cell(control, name, seed));
                    if !ours.join("result.json").exists() {
                        continue;
                    }
                    let mut row = validation::same_cell(&ours, &theirs)?;
                    row["task"] = name.into();
                    row["arm"] = (*arm).into();
                    row["seed"] = seed.into();
                    near_cells.push(row);
                }
            }
        }
        task_rows.push(
            json!({"task": name, "tier": task["tier"], "max_length": max_length,
            "far": far, "arms": per_arm}),
        );
        if !splits.is_empty() {
            split_rows.push(json!({"task": name,
                "columns": ["exact", "examples", "seeds"], "arms": splits}));
        }
    }

    let identical = near_cells.iter().filter(|r| r["pass"] == true).count();
    let differing: Vec<Value> = near_cells
        .iter()
        .filter(|r| r["pass"] != true)
        .cloned()
        .collect();
    let geometric = |logs: &Vec<f64>| finite((logs.iter().sum::<f64>() / logs.len() as f64).exp());
    let mut cost = serde_json::Map::new();
    for (arm, (far_costs, near_costs)) in runner::ARMS.iter().zip(&costs) {
        cost.insert(
            (*arm).into(),
            json!({"far_tasks": far_costs.len(), "far_ratio": geometric(far_costs),
                "near_tasks": near_costs.len(), "near_ratio": geometric(near_costs)}),
        );
    }
    let out = json!({"root": root, "control": control, "window": runner::WINDOW,
        "tiers": tiers, "tasks": task_rows, "splits": split_rows,
        "near": {"cells": near_cells.len(), "identical": identical, "differing": differing},
        "cost": cost});
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
