//! Task #204: immutable observed trajectories and finite displacement forecasts.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../residual_dynamics/src/algebra.rs"]
pub mod algebra;
mod audit;
#[path = "../class_geometry/src/classes.rs"]
pub mod classes;
mod endpoint;
#[path = "../geometry_gap/evidence.rs"]
pub mod evidence;
#[path = "../single_update/src/forecast_jet.rs"]
pub mod forecast_jet;
#[path = "../single_update/src/forecast_network.rs"]
pub mod forecast_network;
#[path = "../residual_dynamics/src/gradient.rs"]
pub mod gradient;
pub mod inputs {
    pub use crate::evidence::{hash, require};
}
#[path = "../forecast_tightening/src/intervals.rs"]
pub mod intervals;
#[path = "../forecast_tightening/src/jet.rs"]
pub mod jet;
#[path = "../forecast_tightening/src/matrix.rs"]
pub mod matrix;
#[path = "../flat_approx/src/model.rs"]
pub mod model;
#[path = "../forecast_tightening/src/network.rs"]
pub mod network;
#[path = "../flat_approx/fixtures/reference.rs"]
pub mod oracle;
#[path = "../forecast_tightening/src/product.rs"]
pub mod product;
#[path = "../single_update/src/query_step.rs"]
pub mod query_step;
mod trace;
use evidence::{Ledger, hash, require, write};
use serde_json::json;
use std::{fs, path::Path, time::Instant};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SEEDS: [u64; 8] = [7, 11, 1, 3, 5, 13, 17, 19];
const WINDOWS: [(u64, u64); 8] = [
    (4000, 5000),
    (4000, 8000),
    (5000, 5250),
    (5000, 6000),
    (6000, 6250),
    (6000, 7000),
    (7000, 7250),
    (7000, 8000),
];
fn source_hash() -> String {
    hash(
        &[
            include_bytes!("main.rs").as_slice(),
            include_bytes!("trace.rs"),
            include_bytes!("endpoint.rs"),
            include_bytes!("audit.rs"),
            include_bytes!("PROTOCOL.md"),
            include_bytes!("../class_geometry_wsd/Cargo.toml"),
            include_bytes!("../class_geometry_wsd/Cargo.lock"),
            include_bytes!("../class_geometry_wsd/build.rs"),
            include_bytes!("../class_geometry/src/classes.rs"),
            include_bytes!("../geometry_gap/evidence.rs"),
            include_bytes!("../flat_approx/src/model.rs"),
            include_bytes!("../residual_dynamics/src/algebra.rs"),
            include_bytes!("../residual_dynamics/src/gradient.rs"),
            include_bytes!("../single_update/src/forecast_jet.rs"),
            include_bytes!("../single_update/src/forecast_network.rs"),
            include_bytes!("../single_update/src/query_step.rs"),
            include_bytes!("../forecast_tightening/src/intervals.rs"),
            include_bytes!("../forecast_tightening/src/jet.rs"),
            include_bytes!("../forecast_tightening/src/matrix.rs"),
            include_bytes!("../forecast_tightening/src/network.rs"),
            include_bytes!("../forecast_tightening/src/product.rs"),
            env!("TRAINER_SOURCE_SHA256").as_bytes(),
        ]
        .concat(),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 2 && args[0] == "verify" {
        return audit::all(Path::new(&args[1]));
    }
    require(
        args.len() == 5,
        "usage: geometry-persistence NEW_OUTPUT LR001 LR01 SETTLE SUPPORT203",
    )?;
    let out = Path::new(&args[0]);
    require(!out.exists(), "refusing to overwrite evidence")?;
    let mut ledger = Ledger::default();
    let support = Path::new(&args[4]).canonicalize()?;
    let sr = ledger.json(&support.join("report.json"))?;
    require(
        sr["status"] == "pass" && sr["declared_cases"] == 744,
        "incomplete support source",
    )?;
    let mut cohorts = vec![];
    for (i, name) in ["lr001", "lr01", "settle"].into_iter().enumerate() {
        let root = Path::new(&args[i + 1]).canonicalize()?;
        let r = ledger.json(&root.join("report.json"))?;
        require(
            r["status"] == "pass" && r["seeds"].as_array().is_some_and(|v| v.len() == 8),
            "incomplete cohort",
        )?;
        cohorts.push((name, root, r));
    }
    fs::create_dir(out)?;
    algebra::threads();
    let start = Instant::now();
    let mut slots = vec![];
    let mut traces = vec![];
    for (name, _, _) in &cohorts {
        for seed in SEEDS {
            traces.push(json!({"cohort":name,"seed":seed,"status":"pending"}));
            for (anchor, end) in WINDOWS {
                slots.push(json!({"cohort":name,"seed":seed,"anchor":anchor,"end":end,"horizon":end-anchor,"status":"pending"}));
            }
        }
    }
    let mut report = json!({"status":"running","source_sha256":source_hash(),"protocol":include_str!("PROTOCOL.md"),
        "support":support,"cohorts":cohorts.iter().map(|(name,p,_)|json!({"name":name,"path":p})).collect::<Vec<_>>(),
        "declared_traces":24,"declared_cases":192,"traces":traces,"cases":slots});
    write(&out.join("report.json"), &report)?;
    let mut slot = 0;
    for (ci, (name, root, source)) in cohorts.iter().enumerate() {
        for (si, seed) in SEEDS.into_iter().enumerate() {
            let reference = &source["seeds"][si];
            require(
                reference["seed"] == seed
                    && reference["completed"] == 10000
                    && reference["status"] == "pass",
                "seed inventory changed",
            )?;
            for key in [
                "initial_sha256",
                "train_sha256",
                "validation_sha256",
                "order_sha256",
            ] {
                require(
                    reference[key] == cohorts[0].2["seeds"][si][key],
                    "unpaired input",
                )?;
            }
            let seed_root = root.join(format!("seed{seed}"));
            let original = ledger.json(&seed_root.join("step-000000/queries.json"))?;
            let mut active = vec![false; 97];
            let mut training = vec![false; 9409];
            for row in original.as_array().ok_or("missing initial queries")? {
                if row[2] == 0 {
                    active[row[1].as_u64().unwrap() as usize] = true;
                    training[row[0].as_u64().unwrap() as usize] = true;
                }
            }
            require(
                active.iter().filter(|&&x| x).count() == 33
                    && training.iter().filter(|&&x| x).count() == 4704,
                "support or split changed",
            )?;
            let tr = trace::run(
                &seed_root,
                reference,
                &active,
                &out.join(format!("{name}-seed{seed}-trace.jsonl")),
                false,
            );
            report["traces"][ci * 8 + si]["audit"] = match tr {
                Ok(v) => v,
                Err(e) => json!({"status":"error","error":e.to_string()}),
            };
            report["traces"][ci * 8 + si]["status"] =
                report["traces"][ci * 8 + si]["audit"]["status"].clone();
            for (anchor, end) in WINDOWS {
                let case_out = out.join(format!("{name}-seed{seed}-{anchor}-{end}"));
                fs::create_dir(&case_out)?;
                let result = if start.elapsed().as_secs_f64() < 1800. {
                    endpoint::run(
                        endpoint::Case {
                            name,
                            seed,
                            anchor,
                            end,
                            root: &seed_root,
                            reference,
                            support: &support,
                            support_report: &sr,
                            training: &training,
                            out: &case_out,
                        },
                        &mut ledger,
                    )
                } else {
                    Err("1800-second frozen cap".into())
                };
                let (status, value) = match result {
                    Ok(v) => ("pass", v),
                    Err(e) => ("error", json!({"error":e.to_string()})),
                };
                report["cases"][slot]["status"] = json!(status);
                report["cases"][slot]["audit"] = value;
                slot += 1;
                report["seconds"] = json!(start.elapsed().as_secs_f64());
                write(&out.join("report.json"), &report)?;
            }
            eprintln!(
                "persistence cohort={name} seed={seed} cases={slot}/192 elapsed={:.1}s",
                start.elapsed().as_secs_f64()
            );
        }
    }
    ledger.verify()?;
    let failed = ["traces", "cases"]
        .into_iter()
        .flat_map(|k| report[k].as_array().unwrap())
        .filter(|v| v["status"] != "pass")
        .count();
    report["failed"] = json!(failed);
    report["status"] = json!(if failed == 0 { "pass" } else { "fail" });
    report["seconds"] = json!(start.elapsed().as_secs_f64());
    report["inputs"] = json!({"unchanged":true,"files":ledger.0});
    write(&out.join("report.json"), &report)?;
    require(failed == 0, "persistence audit failed; all slots retained")
}
