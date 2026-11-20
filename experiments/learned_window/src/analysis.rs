//! `analyze ROOT CONTROL_ROOT RERUN_ROOT`: the comparisons of `PROTOCOL.md`
//! § Analysis, as JSON on standard output, from the cells' files alone. ROOT
//! holds this cohort's arm; CONTROL_ROOT is the recorded control arm (gpt's
//! `phase-score-lerope`); RERUN_ROOT is the recorded cohort of `../rerope`,
//! with its arms `rerope` and `leaky-rerope`.
//!
//! 1. `tiers`: per arm and tier, the cells that passed, ran out of budget,
//!    failed to run or are missing.
//! 2. `tasks`: per task and arm, the epochs to first99 per seed and their
//!    median (a cell that did not pass counts as infinite), its ratios to the
//!    control's and to Leaky ReRoPE's medians, and the median seconds per
//!    update with their ratios. A task is `far` when one of its sequences is
//!    longer than the initial window, so that the arms differ from the
//!    control on it.
//! 3. `splits`: per task with length or position-shift splits, per arm, the
//!    exact sequences summed over the seeds that finished.
//! 4. `windows`: the final window and slope of every head in the far cells of
//!    this cohort's arm, and over all of them how many heads opened over the
//!    longest training distance (no far pair in any training batch, so no
//!    gradient), narrowed below the initial window, widened short of that, or
//!    never moved, and where the slopes went.
//! 5. `near`: every cell of this cohort's arm on a task that is not far
//!    against the recorded control cell, as `validate`'s replays compare
//!    them. The arm runs the control's kernels there, so they must agree bit
//!    for bit.
//! 6. `cost`: per arm, the geometric mean over the far tasks and over the
//!    near ones of the ratio of seconds per update to the control's, and over
//!    the far tasks of the ratio to Leaky ReRoPE's. The near tasks do the
//!    control's work, so their ratio is the drift of the machine between the
//!    cohorts.
use crate::{Result, data, runner, validation};
use serde_json::{Value, json};
use std::{fs, path::Path};

/// The recorded arms of `../rerope`.
const RERUN_ARMS: [&str; 2] = ["rerope", "leaky-rerope"];
/// The arm the learned one starts as: the ratios of items 2 and 6 take it too.
const LEAKY: &str = "leaky-rerope";

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

/// A head's final window w and slope s, in f64, and whether its ω and σ
/// never left zero.
struct Shape {
    window: f64,
    slope: f64,
    unmoved: bool,
}

/// The layers, the heads and every head's final shape, layer-major, of a
/// cell of the learned arm: ω and σ at the offset its protocol records, in
/// the parameters that open its last checkpoint, written when it finished.
fn final_shapes(dir: &Path) -> Result<(usize, usize, Vec<Shape>)> {
    let protocol: Value = serde_json::from_slice(&fs::read(dir.join("protocol.json"))?)?;
    let number = |v: &Value| -> Result<usize> {
        Ok(v.as_u64().ok_or("protocol without learned windows")? as usize)
    };
    let encoding = &protocol["position_encoding"];
    let (offset, window, slope) = (
        number(&encoding["parameter_offset"])?,
        number(&encoding["window"])?,
        number(&encoding["slope"])?,
    );
    let case = &protocol["case"]["case"];
    let (layers, heads) = (number(&case["layers"])?, number(&case["heads"])?);
    let n = layers * heads;
    let bytes = fs::read(dir.join("checkpoint.bin"))?;
    if bytes.len() < 4 * (offset + 2 * n) {
        return Err("checkpoint shorter than its windows".into());
    }
    let value = |i: usize| {
        f32::from_le_bytes([
            bytes[4 * i],
            bytes[4 * i + 1],
            bytes[4 * i + 2],
            bytes[4 * i + 3],
        ])
    };
    let shapes = (0..n)
        .map(|i| {
            let (omega, sigma) = (value(offset + i), value(offset + n + i));
            Shape {
                window: window as f64 * (omega as f64).exp(),
                slope: 1. / (1. + (slope - 1) as f64 * (-(sigma as f64)).exp()),
                unmoved: omega == 0. && sigma == 0.,
            }
        })
        .collect();
    Ok((layers, heads, shapes))
}

/// The smallest, median and largest of `values`.
fn spread(values: impl Iterator<Item = f64>) -> Value {
    let mut v: Vec<f64> = values.collect();
    v.sort_by(f64::total_cmp);
    json!({"min": v.first().copied().map_or(Value::Null, finite),
        "median": finite(median(v.clone())), "max": v.last().copied().map_or(Value::Null, finite)})
}

pub fn run(root: &Path, control: &Path, rerun: &Path) -> Result<()> {
    let manifest = runner::suite();
    let tasks = manifest["tasks"].as_array().ok_or("suite tasks")?;
    let mut arms = vec![("control", control.to_path_buf())];
    arms.extend(RERUN_ARMS.iter().map(|&a| (a, rerun.join(a))));
    arms.extend(runner::ARMS.iter().map(|&a| (a, root.join(a))));
    let leaky = arms
        .iter()
        .position(|(a, _)| *a == LEAKY)
        .ok_or("no Leaky ReRoPE arm")?;
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
    let mut window_cells = Vec::new();
    // Every head of the far cells, with its task's longest training sequence.
    let mut heads_seen: Vec<(Shape, usize)> = Vec::new();
    // Per arm: the logarithms of the far and near tasks' cost ratios to the
    // control, and of the far tasks' to Leaky ReRoPE.
    let mut costs = vec![(Vec::new(), Vec::new(), Vec::new()); arms.len()];
    for task in tasks {
        let name = task["name"].as_str().unwrap_or("");
        let max_length = result(&cell(control, name, 0))
            .and_then(|r| r["metadata"]["max_length"].as_u64())
            .ok_or_else(|| format!("{name}: no recorded control cell"))?
            as usize;
        let far = max_length > runner::WINDOW;
        // Per arm: the seeds' results, the median epochs to first99 and the
        // median seconds per update, once all three seeds finished.
        let stats: Vec<(Vec<Option<Value>>, f64, f64)> = arms
            .iter()
            .map(|(_, dir)| {
                let results: Vec<Option<Value>> =
                    (0..3).map(|s| result(&cell(dir, name, s))).collect();
                if results.iter().all(Option::is_some) {
                    let results_ = results.iter().flatten();
                    let epochs = results_
                        .clone()
                        .map(|r| r["first99_epochs"].as_f64().unwrap_or(f64::INFINITY));
                    let cost = results_.map(|r| seconds_per_update(r).unwrap_or(f64::NAN));
                    let (m, c) = (median(epochs.collect()), median(cost.collect()));
                    (results, m, c)
                } else {
                    (results, f64::NAN, f64::NAN)
                }
            })
            .collect();
        let (control_median, control_cost) = (stats[0].1, stats[0].2);
        let (leaky_median, leaky_cost) = (stats[leaky].1, stats[leaky].2);
        let mut per_arm = serde_json::Map::new();
        let mut splits = serde_json::Map::new();
        for (index, ((arm, dir), (results, m, cost))) in arms.iter().zip(&stats).enumerate() {
            let epochs: Vec<Value> = results
                .iter()
                .map(|r| {
                    r.as_ref()
                        .map_or(Value::Null, |r| r["first99_epochs"].clone())
                })
                .collect();
            if index > 0 {
                let (far_costs, near_costs, leaky_costs) = &mut costs[index];
                if (cost / control_cost).is_finite() {
                    if far { far_costs } else { near_costs }.push((cost / control_cost).ln());
                }
                if far && (cost / leaky_cost).is_finite() {
                    leaky_costs.push((cost / leaky_cost).ln());
                }
            }
            per_arm.insert(
                (*arm).into(),
                json!({"first99_epochs": epochs, "median": finite(*m),
                    "median_is_infinite": *m == f64::INFINITY,
                    "ratio_to_control": finite(m / control_median),
                    "ratio_to_leaky": finite(m / leaky_median),
                    "seconds_per_update": finite(*cost),
                    "cost_ratio_to_control": finite(cost / control_cost),
                    "cost_ratio_to_leaky": finite(cost / leaky_cost)}),
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
            if !runner::ARMS.contains(arm) {
                continue;
            }
            // The longest training sequence: no training distance reaches it.
            let train_length = if far {
                data::build(task)?.splits["train"]
                    .iter()
                    .map(|e| e.tokens.len())
                    .max()
                    .ok_or("empty training split")?
            } else {
                0
            };
            for seed in 0..3 {
                let (ours, theirs) = (cell(dir, name, seed), cell(control, name, seed));
                if !ours.join("result.json").exists() {
                    continue;
                }
                if far {
                    let (layers, heads, shapes) = final_shapes(&ours)?;
                    let by_layer = |f: fn(&Shape) -> f64| -> Vec<Vec<f64>> {
                        shapes
                            .chunks(heads)
                            .map(|l| l.iter().map(f).collect())
                            .collect()
                    };
                    window_cells.push(json!({"task": name, "arm": arm, "seed": seed,
                        "train_length": train_length, "layers": layers, "heads": heads,
                        "windows": by_layer(|s| s.window), "slopes": by_layer(|s| s.slope)}));
                    heads_seen.extend(shapes.into_iter().map(|s| (s, train_length)));
                } else {
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
    let (initial_window, initial_slope) = (runner::WINDOW as f64, 1. / runner::SLOPE as f64);
    let count =
        |f: &dyn Fn(&Shape, usize) -> bool| heads_seen.iter().filter(|(s, l)| f(s, *l)).count();
    let open = |s: &Shape, l: usize| s.window.ceil() >= l as f64;
    let windows = json!({"cells": window_cells.len(), "heads": heads_seen.len(),
        "initial_window": runner::WINDOW, "initial_slope": initial_slope,
        "open_in_training": count(&open),
        "narrowed": count(&|s, _| s.window < initial_window),
        "widened_short_of_open": count(&|s, l| s.window > initial_window && !open(s, l)),
        "unmoved": count(&|s, _| s.unmoved),
        "slope_toward_zero": count(&|s, _| s.slope < initial_slope),
        "slope_toward_one": count(&|s, _| s.slope > initial_slope),
        "window_spread": spread(heads_seen.iter().map(|(s, _)| s.window)),
        "slope_spread": spread(heads_seen.iter().map(|(s, _)| s.slope)),
        "cells_by_task": window_cells});
    let geometric = |logs: &Vec<f64>| finite((logs.iter().sum::<f64>() / logs.len() as f64).exp());
    let mut cost = serde_json::Map::new();
    for ((arm, _), (far_costs, near_costs, leaky_costs)) in arms.iter().zip(&costs).skip(1) {
        cost.insert(
            (*arm).into(),
            json!({"far_tasks": far_costs.len(), "far_ratio": geometric(far_costs),
                "near_tasks": near_costs.len(), "near_ratio": geometric(near_costs),
                "far_ratio_to_leaky": geometric(leaky_costs)}),
        );
    }
    let out = json!({"root": root, "control": control, "rerun": rerun,
        "window": runner::WINDOW, "slope": runner::SLOPE,
        "tiers": tiers, "tasks": task_rows, "splits": split_rows, "windows": windows,
        "near": {"cells": near_cells.len(), "identical": identical, "differing": differing},
        "cost": cost});
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

/// `final_shapes` reads ω, then σ, at the offset the protocol records, layer
/// by layer, from the parameters at the head of a checkpoint.
#[test]
fn final_shapes_read_each_head_after_the_recorded_offset() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("learned-window-shapes-{}", std::process::id()));
    fs::create_dir_all(&dir)?;
    let (layers, heads, offset) = (2, 3, 5);
    let protocol = json!({"case": {"case": {"layers": layers, "heads": heads}},
        "position_encoding": {"parameter_offset": offset, "window": 16, "slope": 16}});
    fs::write(dir.join("protocol.json"), serde_json::to_vec(&protocol)?)?;
    let ln = |x: f32| x.ln();
    let omega = [0., ln(2.), 0., -ln(2.), 0., 0.];
    let sigma = [0., 0., ln(15.), 0., 0., -ln(15.)];
    let parameters: Vec<f32> =
        [vec![7.; offset], omega.to_vec(), sigma.to_vec(), vec![7.]].concat();
    let state: Vec<f32> = [
        parameters.clone(),
        vec![3.; parameters.len()],
        vec![5.; parameters.len()],
    ]
    .concat();
    let bytes: Vec<u8> = state.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(dir.join("checkpoint.bin"), bytes)?;
    let (l, h, shapes) = final_shapes(&dir)?;
    fs::remove_dir_all(&dir)?;
    assert_eq!((l, h), (layers, heads));
    let windows = [16., 32., 16., 8., 16., 16.];
    let slopes = [1. / 16., 1. / 16., 0.5, 1. / 16., 1. / 16., 1. / 226.];
    for (i, s) in shapes.iter().enumerate() {
        assert!(
            (s.window - windows[i]).abs() < 1e-5 * windows[i],
            "window {i}: {}",
            s.window
        );
        assert!((s.slope - slopes[i]).abs() < 1e-6, "slope {i}: {}", s.slope);
        assert_eq!(s.unmoved, i == 0 || i == 4, "head {i}");
    }
    Ok(())
}
