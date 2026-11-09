//! #207: constructor reads old checkpoints only; verifier opens #206 next updates.
mod analysis;
mod audit;
pub mod classes;
#[path = "../geometry_gap/evidence.rs"]
pub mod evidence;
#[path = "../ce_lyapunov/analysis.rs"]
pub mod reference;
mod state;
pub mod inputs {
    pub use crate::evidence::require;
}
use evidence::{Ledger, hash, require, write};
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const SEEDS: [u64; 8] = [7, 11, 1, 3, 5, 13, 17, 19];
pub const ANCHORS: [usize; 6] = [0, 1000, 4000, 6000, 8000, 9500];
pub const COHORTS: [&str; 3] = ["lr001", "lr01", "settle"];
fn source_hash() -> String {
    hash(
        &[
            include_bytes!("main.rs").as_slice(),
            include_bytes!("analysis.rs"),
            include_bytes!("audit.rs"),
            include_bytes!("state.rs"),
            include_bytes!("classes.rs"),
            include_bytes!("PROTOCOL.md"),
            include_bytes!("Cargo.toml"),
            include_bytes!("Cargo.lock"),
            include_bytes!("../ce_lyapunov/analysis.rs"),
            include_bytes!("../geometry_gap/evidence.rs"),
        ]
        .concat(),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 5 && args[0] == "verify" {
        return audit::verify(Path::new(&args[1]), &args[2..]);
    }
    require(
        args.len() == 4,
        "usage: ce-metric NEW_OUTPUT ORIGINAL_LR001 ORIGINAL_LR01 ORIGINAL_SETTLE | verify OUTPUT CE_LR001 CE_LR01 CE_SETTLE",
    )?;
    let out = Path::new(&args[0]);
    require(!out.exists(), "refusing overwrite")?;
    let started = Instant::now();
    let mut ledger = Ledger::default();
    let mut sources = vec![];
    for (i, cohort) in COHORTS.into_iter().enumerate() {
        let root = Path::new(&args[i + 1]).canonicalize()?;
        let report = ledger.json(&root.join("report.json"))?;
        require(
            report["status"] == "pass" && report["seeds"].as_array().is_some_and(|s| s.len() == 8),
            "source cohort",
        )?;
        sources.push((cohort, root, report));
    }
    fs::create_dir(out)?;
    let slots:Vec<_>=COHORTS.into_iter().flat_map(|cohort|SEEDS.into_iter().flat_map(move|seed|
        ANCHORS.into_iter().map(move|anchor|json!({"cohort":cohort,"seed":seed,"anchor":anchor,"status":"pending"})))).collect();
    let mut report = json!({"status":"running","cases":slots,"source_sha256":source_hash(),
        "protocol":include_str!("PROTOCOL.md"),"next_state_reads":0});
    let mut slot = 0;
    for (cohort, root, sr) in &sources {
        for (si, seed) in SEEDS.into_iter().enumerate() {
            let seed_report = &sr["seeds"][si];
            require(
                seed_report["seed"] == seed
                    && seed_report["status"] == "pass"
                    && seed_report["completed"] == 10000,
                "seed inventory",
            )?;
            for key in [
                "initial_sha256",
                "train_sha256",
                "validation_sha256",
                "order_sha256",
            ] {
                require(
                    seed_report[key] == sources[0].2["seeds"][si][key],
                    "paired source mismatch",
                )?;
            }
            let root = root.join(format!("seed{seed}"));
            let active = state::support(&ledger.json(&root.join("step-000000/queries.json"))?)?;
            for anchor in ANCHORS {
                let result = (|| -> Result<Value> {
                    require(started.elapsed().as_secs_f64() < 600., "construction cap")?;
                    let p = root.join(format!("step-{anchor:06}/state.bin"));
                    let bytes = ledger.read(&p)?;
                    let probe = seed_report["geometry"]
                        .as_array()
                        .ok_or("probes")?
                        .iter()
                        .find(|p| p["step"] == anchor)
                        .ok_or("anchor")?;
                    require(hash(&bytes) == probe["state_sha256"], "checkpoint hash")?;
                    let old = state::decode(&bytes, anchor)?;
                    let (alpha, r) = state::metric(&old, &active, anchor);
                    let eta = state::rate(cohort, anchor);
                    let s = eta * alpha / (1. - 0.9f64.powi(anchor as i32 + 1));
                    let f = json!({"old_state":p,"old_sha256":hash(&bytes),"active":active,
                        "alpha":alpha,"r":r,"eta":eta,"s":s,"cohort":cohort,"seed":seed,"anchor":anchor});
                    let name = format!("{cohort}-seed{seed}-anchor{anchor}.json");
                    write(&out.join(&name), &f)?;
                    Ok(json!({"file":name,"sha256":hash(&fs::read(out.join(&name))?)}))
                })();
                match result {
                    Ok(v) => {
                        report["cases"][slot]["status"] = json!("frozen");
                        report["cases"][slot]["metric"] = v;
                    }
                    Err(e) => {
                        report["cases"][slot]["status"] = json!("error");
                        report["cases"][slot]["error"] = json!(e.to_string());
                    }
                }
                slot += 1;
            }
            write(&out.join("construction.json"), &report)?;
        }
    }
    ledger.verify()?;
    let failed = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["status"] != "frozen")
        .count();
    report["status"] = json!(if failed == 0 { "frozen" } else { "fail" });
    report["failed"] = json!(failed);
    report["seconds"] = json!(started.elapsed().as_secs_f64());
    report["inputs"] = json!(ledger.0);
    write(&out.join("construction.json"), &report)?;
    require(failed == 0, "construction failed; every slot retained")
}
