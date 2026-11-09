//! All paired seeds and fixed probes; full-recipe comparison, no seed selection.
use serde_json::{Value, json};
use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn load(path: &Path) -> Result<Value> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}
fn n(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
fn coverage(probe: &Value, rank: u64) -> f64 {
    n(&probe["coverage"]["validation"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["family"] == "readout" && c["rank"] == rank)
        .unwrap()["coverage"])
}
fn describe(seed: &Value) -> Value {
    if seed["status"] != "pass" {
        return json!({"status":seed["status"],"error":seed["error"]});
    }
    let rows = seed["geometry"].as_array().unwrap();
    let late: Vec<_> = rows
        .iter()
        .filter(|p| p["step"].as_u64().unwrap() >= 5000)
        .collect();
    let endpoint = rows.last().unwrap();
    let first:Vec<_>=[0.5,0.9,0.99].into_iter().map(|level|json!({"level":level,
        "accuracy":rows.iter().find(|p|n(&p["metrics"]["validation"]["accuracy"])>=level).map(|p|p["step"].clone()),
        "coverage":rows.iter().find(|p|coverage(p,32)>=level).map(|p|p["step"].clone())})).collect();
    let regressions = rows
        .windows(2)
        .filter(|w| coverage(&w[1], 32) < coverage(&w[0], 32))
        .count();
    json!({"status":"pass","accuracy":endpoint["metrics"]["validation"]["accuracy"],"readout32":coverage(endpoint,32),
        "readout33":coverage(endpoint,33),"readout128":coverage(endpoint,128),"active_share":endpoint["structural"]["active_fraction"],
        "late_min_accuracy":late.iter().map(|p|n(&p["metrics"]["validation"]["accuracy"])).fold(f64::INFINITY,f64::min),
        "late_min_readout32":late.iter().map(|p|coverage(p,32)).fold(f64::INFINITY,f64::min),
        "readout32_regressions":regressions,"crossings":first})
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: compare BASELINE_DIRECTORY CANDIDATE_DIRECTORY".into());
    }
    let base = load(&Path::new(&args[0]).join("report.json"))?;
    let new = load(&Path::new(&args[1]).join("report.json"))?;
    let mut pairs = vec![];
    for &seed in &[7u64, 11, 1, 3, 5, 13, 17, 19] {
        let b = base["seeds"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["seed"] == seed)
            .ok_or("baseline seed missing")?;
        let c = new["seeds"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["seed"] == seed)
            .ok_or("candidate seed missing")?;
        let before = describe(b);
        let after = describe(c);
        let mut row = json!({"seed":seed,"baseline":before,"candidate":after});
        if b["status"] == "pass" && c["status"] == "pass" {
            for key in [
                "initial_sha256",
                "train_sha256",
                "validation_sha256",
                "order_sha256",
            ] {
                if b[key] != c[key] {
                    return Err("paired provenance mismatch".into());
                }
            }
            let mut clipped = 0;
            let mut max_norm = 0f64;
            let mut final_eta = 0.;
            let mut count = 0;
            let file = File::open(Path::new(&args[1]).join(format!("seed{seed}/updates.jsonl")))?;
            for line in BufReader::new(file).lines() {
                let update: Value = serde_json::from_str(&line?)?;
                let norm = n(&update["gradient_norm"]);
                clipped += usize::from(norm > 1.);
                max_norm = max_norm.max(norm);
                final_eta = n(&update["eta"]);
                count += 1;
            }
            if count != 10000 {
                return Err("incomplete update table".into());
            }
            let mut differences = json!({});
            for key in [
                "accuracy",
                "readout32",
                "readout33",
                "readout128",
                "active_share",
                "late_min_accuracy",
                "late_min_readout32",
            ] {
                differences[key] = json!(n(&after[key]) - n(&before[key]));
            }
            row["difference"] = differences;
            row["clipped_updates"] = json!(clipped);
            row["max_gradient_norm"] = json!(max_norm);
            row["final_lr"] = json!(final_eta);
        }
        pairs.push(row);
    }
    let thresholds:Vec<_>=[0.9,0.99].into_iter().flat_map(|level| {
        ["baseline","candidate"].into_iter().map({let pairs=&pairs;move |arm|json!({"arm":arm,"level":level,"declared_seeds":8,
            "endpoint_accuracy":pairs.iter().filter(|p|p[arm]["status"]=="pass" && n(&p[arm]["accuracy"])>=level).count(),
            "endpoint_readout32":pairs.iter().filter(|p|p[arm]["status"]=="pass" && n(&p[arm]["readout32"])>=level).count(),
            "all_late_accuracy":pairs.iter().filter(|p|p[arm]["status"]=="pass" && n(&p[arm]["late_min_accuracy"])>=level).count(),
            "all_late_readout32":pairs.iter().filter(|p|p[arm]["status"]=="pass" && n(&p[arm]["late_min_readout32"])>=level).count()})})
    }).collect();
    println!(
        "{}",
        json!({"candidate_status":new["status"],"candidate_seconds":new["seconds"],"candidate_source":new["source_sha256"],"pairs":pairs,"threshold_counts":thresholds})
    );
    Ok(())
}
