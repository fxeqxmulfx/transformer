//! Task #203: stream every embedding update; retain fixed hold-window budgets.
#[path = "../geometry_gap/evidence.rs"]
pub mod evidence;
pub mod moments;
use evidence::{hash, require};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
    time::Instant,
};
pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
const SEEDS: [u64; 8] = [7, 11, 1, 3, 5, 13, 17, 19];
const E: usize = 97 * 128;
fn read_row(reader: &mut BufReader<File>, digest: &mut Sha256) -> Result<Vec<f64>> {
    let mut bytes = vec![0; 4 * E];
    reader.read_exact(&mut bytes)?;
    digest.update(&bytes);
    let row: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b) as f64)
        .collect();
    require(row.iter().all(|v| v.is_finite()), "nonfinite embedding")?;
    Ok(row)
}
fn hex(h: Sha256) -> String {
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
fn seed(root: &Path, reference: &Value) -> Result<Value> {
    let mut trace = BufReader::new(File::open(root.join("embeddings.bin"))?);
    let mut td = Sha256::new();
    let mut header = [0; 24];
    trace.read_exact(&mut header)?;
    td.update(header);
    require(
        header
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| u64::from_le_bytes(*b))
            .collect::<Vec<_>>()
            == [1, 10001, E as u64],
        "trace shape changed",
    )?;
    let mut old = read_row(&mut trace, &mut td)?;
    let mut updates = BufReader::new(File::open(root.join("updates.jsonl"))?);
    let mut ud = Sha256::new();
    let mut line = String::new();
    let mut anchor: Vec<f64> = vec![];
    let mut radius = 0.;
    let mut budget = 0.;
    let mut rate_sum = 0.;
    let mut rounding_sum = 0.;
    let mut max_distance = 0f64;
    let mut path_length = 0.;
    let mut min_share = 1f64;
    let mut anchor_active = 0.;
    let mut moving = 0;
    let mut hold_updates = 0;
    let mut max_excess = 0f64;
    let mut max_direction = 0f64;
    for step in 1..=10000 {
        line.clear();
        require(updates.read_line(&mut line)? > 0, "missing update row")?;
        ud.update(line.as_bytes());
        let row: Value = serde_json::from_str(&line)?;
        require(row["step"] == step, "update cadence changed")?;
        let eta = row["eta"].as_f64().ok_or("missing learning rate")?;
        let rounding = row["rounding_max"].as_f64().ok_or("missing rounding")?;
        require(
            eta >= 0. && eta * 0.1 <= 1. && rounding >= 0.,
            "invalid update scale",
        )?;
        let new = read_row(&mut trace, &mut td)?;
        let b = moments::direction_bound(step);
        let allowed = eta * b + rounding;
        let mut squared_step = 0.;
        let mut squared_distance = 0.;
        for k in 0..E {
            let forcing = (new[k] - (1. - eta * 0.1) * old[k]).abs();
            max_excess = max_excess.max((forcing - allowed) / allowed.max(1e-30));
            if eta > 0. {
                max_direction = max_direction.max((forcing - rounding).max(0.) / eta);
            }
            squared_step += (new[k] - old[k]).powi(2);
            if (4001..=8000).contains(&step) {
                squared_distance += (new[k] - anchor[k]).powi(2);
            }
        }
        if step == 4000 {
            anchor = new.clone();
            radius = anchor.iter().map(|v| v.abs()).fold(0., f64::max);
            anchor_active = row["after"][0]
                .as_f64()
                .ok_or("missing active energy")?
                .sqrt();
        }
        if (4001..=8000).contains(&step) {
            budget += eta * (0.1 * radius + b) + rounding;
            radius = (1. - eta * 0.1) * radius + eta * b + rounding;
            rate_sum += eta;
            rounding_sum += rounding;
            path_length += squared_step.sqrt();
            max_distance = max_distance.max(squared_distance.sqrt());
            moving += usize::from(squared_step > 0.);
            hold_updates += 1;
            let energies: Vec<_> = row["after"]
                .as_array()
                .ok_or("missing energies")?
                .iter()
                .map(|v| v.as_f64().ok_or("invalid energy"))
                .collect::<std::result::Result<_, _>>()?;
            require(
                energies.len() == 3 && energies.iter().all(|v| *v >= 0.),
                "invalid energy partition",
            )?;
            min_share = min_share.min(energies[0] / energies.iter().sum::<f64>());
        }
        old = new;
    }
    require(
        updates.read_line(&mut String::new())? == 0 && trace.read(&mut [0])? == 0,
        "extra trace entries",
    )?;
    let trace_hash = hex(td);
    let updates_hash = hex(ud);
    require(
        reference["trace_sha256"] == trace_hash && reference["updates_sha256"] == updates_hash,
        "trace hash mismatch",
    )?;
    require(
        max_excess <= 1e-6,
        "direction exceeds fixed relative allowance",
    )?;
    require(hold_updates == 4000, "hold inventory changed")?;
    let frobenius_budget = (E as f64).sqrt() * budget;
    Ok(
        json!({"status":"pass","updates":10000,"coordinate_step_checks":10000*E,
        "max_relative_direction_excess":max_excess,"max_observed_direction_lower_bound":max_direction,
        "trace_sha256":trace_hash,"updates_sha256":updates_hash,
        "hold":{"updates":hold_updates,"moving_updates":moving,"learning_rate_sum":rate_sum,
            "rounding_coordinate_sum":rounding_sum,"analytic_coordinate_motion_budget":budget,
            "analytic_frobenius_motion_budget":frobenius_budget,"actual_max_frobenius_displacement":max_distance,
            "actual_frobenius_path_length":path_length,"initial_active_norm":anchor_active,
            "budget_below_initial_active_norm":frobenius_budget < anchor_active,"minimum_active_share":min_share}}),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    require(
        args.len() == 4,
        "usage: geometry-motion NEW_REPORT BASELINE001 BASELINE01 SETTLE",
    )?;
    let output = Path::new(&args[0]);
    require(!output.exists(), "refusing to overwrite evidence")?;
    let started = Instant::now();
    let mut ledger = evidence::Ledger::default();
    let mut report = json!({"status":"running","source_sha256":hash(&[
        include_bytes!("motion.rs").as_slice(),include_bytes!("moments.rs"),include_bytes!("../geometry_gap/evidence.rs"),
        include_bytes!("PROTOCOL.md"),include_bytes!("../class_geometry_wsd/Cargo.toml"),include_bytes!("../class_geometry_wsd/Cargo.lock")].concat()),"cases":[]});
    for (i, name) in ["lr001", "lr01", "settle"].into_iter().enumerate() {
        let root = Path::new(&args[i + 1]).canonicalize()?;
        let r = ledger.json(&root.join("report.json"))?;
        require(r["status"] == "pass", "incomplete source cohort")?;
        for (si, id) in SEEDS.into_iter().enumerate() {
            let reference = &r["seeds"][si];
            require(
                reference["seed"] == id
                    && reference["completed"] == 10000
                    && reference["status"] == "pass",
                "seed inventory changed",
            )?;
            let mut row = match seed(&root.join(format!("seed{id}")), reference) {
                Ok(v) => v,
                Err(e) => json!({"status":"error","error":e.to_string()}),
            };
            row["cohort"] = json!(name);
            row["seed"] = json!(id);
            eprintln!(
                "motion cohort={name} seed={id} status={} elapsed={:.1}s",
                row["status"],
                started.elapsed().as_secs_f64()
            );
            report["cases"].as_array_mut().unwrap().push(row);
            evidence::write(output, &report)?;
        }
    }
    ledger.verify()?;
    let failed = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["status"] != "pass")
        .count();
    report["failed"] = json!(failed);
    report["seconds"] = json!(started.elapsed().as_secs_f64());
    report["status"] = json!(if failed == 0 { "pass" } else { "fail" });
    report["inputs"] = json!(ledger.0);
    evidence::write(output, &report)?;
    require(failed == 0, "motion audit failed; all cases retained")
}
