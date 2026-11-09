//! Task #203, adapted from task #202 at 210efa2a with every declared case retained.
#[path = "../residual_dynamics/src/algebra.rs"]
pub mod algebra;
mod audit;
#[path = "../signed_transport/src/svd.rs"]
pub mod basis_svd;
mod certificate;
#[path = "../geometry_gap/evidence.rs"]
mod evidence;
#[path = "../class_geometry/src/groups.rs"]
pub mod groups;
#[path = "../flat_approx/src/model.rs"]
pub mod model;
mod moments;
#[path = "../class_geometry/src/reconstruction.rs"]
pub mod reconstruction;
use evidence::{Ledger, hash, require, write};
use serde_json::json;
use std::{error::Error, fs, path::Path, time::Instant};
pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
const SEEDS: [u64; 8] = [7, 11, 1, 3, 5, 13, 17, 19];
fn steps() -> Vec<u64> {
    [0, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000]
        .into_iter()
        .chain((5000..=10000).step_by(250))
        .collect()
}
fn source_hash() -> String {
    hash(
        &[
            include_bytes!("support.rs").as_slice(),
            include_bytes!("certificate.rs"),
            include_bytes!("../geometry_gap/evidence.rs"),
            include_bytes!("audit.rs"),
            include_bytes!("moments.rs"),
            include_bytes!("PROTOCOL.md"),
            include_bytes!("../class_geometry_wsd/Cargo.toml"),
            include_bytes!("../class_geometry_wsd/Cargo.lock"),
            include_bytes!("../class_geometry/src/groups.rs"),
            include_bytes!("../class_geometry/src/reconstruction.rs"),
            include_bytes!("../flat_approx/src/model.rs"),
            include_bytes!("../residual_dynamics/src/algebra.rs"),
            include_bytes!("../signed_transport/src/svd.rs"),
        ]
        .concat(),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() == 2 && args[0] == "verify" {
        return audit::run(Path::new(&args[1]));
    }
    require(
        args.len() == 5,
        "usage: geometry-dynamics NEW_OUTPUT_DIRECTORY BASELINE001_DIRECTORY BASELINE01_DIRECTORY SETTLE_DIRECTORY ACTIVE_DIRECTORY",
    )?;
    let output = Path::new(&args[0]);
    require(!output.exists(), "refusing to overwrite evidence")?;
    let mut ledger = Ledger::default();
    let active_root = Path::new(&args[4]).canonicalize()?;
    let active_report = ledger.json(&active_root.join("report.json"))?;
    require(
        active_report["status"] == "pass"
            && active_report["failed"] == 0
            && active_report["cases"]
                .as_array()
                .is_some_and(|s| s.len() == 744),
        "incomplete active-basis reference",
    )?;
    let mut cohorts = vec![];
    for (i, name) in ["lr001", "lr01", "settle"].into_iter().enumerate() {
        let root = Path::new(&args[i + 1]).canonicalize()?;
        let r = ledger.json(&root.join("report.json"))?;
        require(
            r["status"] == "pass" && r["seeds"].as_array().is_some_and(|s| s.len() == 8),
            "incomplete source cohort",
        )?;
        require(
            r["peak_lr"].as_f64().unwrap_or(0.001) == if i == 0 { 0.001 } else { 0.01 },
            "wrong source learning rate",
        )?;
        require(
            r["schedule_name"].as_str().unwrap_or("main_wsd")
                == if i == 2 {
                    "early_settle_1000_4000_hold_1e-4"
                } else {
                    "main_wsd"
                },
            "wrong source schedule",
        )?;
        cohorts.push((name, root, r));
    }
    fs::create_dir(output)?;
    algebra::threads();
    let started = Instant::now();
    let slots: Vec<_> = cohorts
        .iter()
        .flat_map(|(name, _, _)| {
            SEEDS.into_iter().flat_map(move |seed| {
                steps().into_iter().map(
                    move |step| json!({"cohort":name,"seed":seed,"step":step,"status":"pending"}),
                )
            })
        })
        .collect();
    let mut report = json!({"status":"running","source_sha256":source_hash(),"protocol":include_str!("PROTOCOL.md"),"declared_cases":744,"cases":slots});
    report["active_reference"] = json!(active_root);
    report["cohorts"] = json!(
        cohorts
            .iter()
            .map(|(name, root, _)| json!({"name":name,"path":root}))
            .collect::<Vec<_>>()
    );
    write(&output.join("report.json"), &report)?;
    let mut slot = 0;
    for (name, root, source) in &cohorts {
        for (si, seed) in SEEDS.into_iter().enumerate() {
            let row = &source["seeds"][si];
            require(
                row["seed"] == seed && row["status"] == "pass" && row["completed"] == 10000,
                "source seed inventory changed",
            )?;
            for key in [
                "initial_sha256",
                "train_sha256",
                "validation_sha256",
                "order_sha256",
            ] {
                require(
                    row[key] == cohorts[0].2["seeds"][si][key],
                    "paired source inputs differ",
                )?;
            }
            let seed_root = root.join(format!("seed{seed}"));
            // This adapter exposes only observed training labels to the fit.
            let initial = ledger.json(&seed_root.join("step-000000/queries.json"))?;
            let training: Vec<_> = initial
                .as_array()
                .ok_or("missing initial queries")?
                .iter()
                .filter(|r| r[2] == 0)
                .map(|r| {
                    (
                        r[0].as_u64().unwrap() as usize,
                        r[1].as_u64().unwrap() as usize,
                    )
                })
                .collect();
            require(training.len() == 4704, "training inventory changed")?;
            let source_probes = row["geometry"].as_array().ok_or("missing source probes")?;
            require(source_probes.len() == 31, "missing source probes")?;
            for (probe, step) in source_probes.iter().zip(steps()) {
                require(
                    probe["step"] == step && probe["passed"] == true,
                    "source probe cadence changed",
                )?;
                let result = if started.elapsed().as_secs_f64() >= 1800. {
                    json!({"status":"capped"})
                } else {
                    let path = output.join(format!("{name}-seed{seed}-step{step:06}"));
                    let previous = active_root.join(format!("{name}-seed{seed}-step{step:06}"));
                    let old = &active_report["cases"][slot];
                    require(
                        old["cohort"] == *name && old["seed"] == seed && old["step"] == step,
                        "active reference inventory changed",
                    )?;
                    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        certificate::run(
                            &seed_root,
                            probe,
                            &training,
                            &path,
                            &previous,
                            old,
                            &mut ledger,
                        )
                    }));
                    match run {
                        Ok(Ok(v)) => v,
                        Ok(Err(e)) => json!({"status":"error","error":e.to_string()}),
                        Err(_) => json!({"status":"error","error":"panic; see stderr"}),
                    }
                };
                let mut result = result;
                result["cohort"] = json!(name);
                result["seed"] = json!(seed);
                result["step"] = json!(step);
                report["cases"][slot] = result;
                slot += 1;
                write(&output.join("report.json"), &report)?;
                if step == 0 || step == 10000 || report["cases"][slot - 1]["status"] != "pass" {
                    eprintln!(
                        "cohort={name} seed={seed} step={step} status={} elapsed={:.1}s",
                        report["cases"][slot - 1]["status"],
                        started.elapsed().as_secs_f64()
                    );
                }
            }
        }
    }
    ledger.verify()?;
    report["inputs"] = json!({"unchanged":true,"files":ledger.0});
    report["seconds"] = json!(started.elapsed().as_secs_f64());
    let failed = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["status"] != "pass")
        .count();
    report["failed"] = json!(failed);
    report["status"] = json!(if failed == 0 { "pass" } else { "fail" });
    write(&output.join("report.json"), &report)?;
    require(
        failed == 0,
        "support-complete study failed; all slots retained",
    )
}
