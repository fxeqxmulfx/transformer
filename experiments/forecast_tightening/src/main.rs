//! Frozen task #196: tighter mathematical forecasts, unchanged source training.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../../residual_dynamics/src/algebra.rs"]
pub mod algebra;
mod audit;
#[path = "../../single_update/src/forecast_jet.rs"]
pub mod forecast_jet;
#[path = "../../single_update/src/forecast_network.rs"]
pub mod forecast_network;
#[path = "../../residual_dynamics/src/geometry.rs"]
pub mod geometry;
#[path = "../../residual_dynamics/src/gradient.rs"]
pub mod gradient;
#[path = "../../single_update/src/inputs.rs"]
pub mod inputs;
mod intervals;
mod jet;
mod matrix;
#[path = "../../single_update/src/measure.rs"]
pub mod measure;
mod network;
#[path = "../../flat_approx/fixtures/reference.rs"]
pub mod oracle;
#[path = "../../forward_training/src/ownership.rs"]
mod ownership;
mod product;
#[path = "../../single_update/src/query_step.rs"]
pub mod query_step;
mod run;
mod statistics;
#[path = "../../residual_dynamics/src/svd.rs"]
pub mod svd;
use domain::{
    Result,
    data::{Data, Plan},
    model::Arm,
};
use infrastructure::storage;
use inputs::require;
use serde_json::json;
use std::{collections::BTreeMap, fs, path::Path, time::Instant};
fn source_hash() -> String {
    inputs::hash(
        &[
            include_bytes!("main.rs").as_slice(),
            include_bytes!("audit.rs"),
            include_bytes!("intervals.rs"),
            include_bytes!("jet.rs"),
            include_bytes!("matrix.rs"),
            include_bytes!("network.rs"),
            include_bytes!("product.rs"),
            include_bytes!("run.rs"),
            include_bytes!("statistics.rs"),
            include_bytes!("instrumentation.rs"),
            include_bytes!("../Cargo.toml"),
            include_bytes!("../Cargo.lock"),
            include_bytes!("../PROTOCOL.md"),
            include_bytes!("../../forward_training/src/ownership.rs"),
            inputs::source_hash().as_bytes(),
            env!("TRAINER_SOURCE_SHA256").as_bytes(),
        ]
        .concat(),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    require(
        args.len() == 3,
        "usage: rotate90-complex-kernite CHECKPOINT_ROOT TASK188_STATES_TSV NEW_OUTPUT_DIRECTORY",
    )?;
    let root = Path::new(&args[0]).canonicalize()?;
    let reference = Path::new(&args[1]).canonicalize()?;
    let output = Path::new(&args[2]);
    require(!output.exists(), "refusing to overwrite evidence")?;
    let parent = output.parent().unwrap_or(Path::new(".")).canonicalize()?;
    require(
        !parent.starts_with(&root),
        "output must be outside checkpoint inputs",
    )?;
    let mut ledger = inputs::Ledger::default();
    let bytes = ledger.read(&reference)?;
    let text = std::str::from_utf8(&bytes)?;
    let mut expected = BTreeMap::new();
    for line in text.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        require(fields.len() == 4, "invalid reference row")?;
        require(
            expected
                .insert(
                    (
                        fields[0].to_owned(),
                        fields[1].parse::<u64>()?,
                        fields[2].parse::<usize>()?,
                    ),
                    fields[3].to_owned(),
                )
                .is_none(),
            "duplicate reference",
        )?;
    }
    require(expected.len() == 24, "reference must contain all 24 states")?;
    let (original, protocol) = inputs::protocol(&mut ledger, &root)?;
    fs::create_dir(output)?;
    algebra::threads();
    let mut report = json!({"status":"running","protocol":include_str!("../PROTOCOL.md"),"source_sha256":source_hash(),"original_protocol":original,"reference_sha256":storage::hash(&bytes),"cases":[]});
    let started = Instant::now();
    let mut failed = 0;
    for seed in [7, 11] {
        let data = Data::new(seed);
        let plan = Plan::new(seed, 10001);
        for arm in Arm::ALL {
            for step in [0, 5000, 10000] {
                let directory = output.join(format!("{}-seed{seed}-step{step}", arm.name()));
                fs::create_dir(&directory)?;
                let before = ownership::snapshot(&directory, "before");
                let begin = Instant::now();
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
                        require(
                            started.elapsed().as_secs_f64() < 900.,
                            "900-second cohort cap",
                        )?;
                        let case = inputs::Case {
                            arm,
                            seed,
                            step,
                            data: &data,
                            plan: &plan,
                            protocol: &protocol,
                            forecast_output: &directory,
                        };
                        let saved = inputs::checkpoint(&mut ledger, &root, &case)?;
                        let state = expected
                            .get(&(arm.name().to_owned(), seed, step))
                            .ok_or("missing reference state")?;
                        run::run(&case, &saved, &directory, state)
                    }));
                let mut row = json!({"arm":arm.name(),"seed":seed,"step":step,"post_budget_probe":step==10000,"seconds":begin.elapsed().as_secs_f64(),"gpu_before":before});
                match result {
                    Ok(Ok(v)) => {
                        row["status"] = json!(if v["passed"] == true { "pass" } else { "fail" });
                        row["audit"] = v;
                    }
                    Ok(Err(e)) => {
                        row["status"] = json!("error");
                        row["error"] = json!(e.to_string());
                    }
                    Err(_) => {
                        row["status"] = json!("error");
                        row["error"] = json!("panic; see stderr");
                    }
                }
                if row["status"] != "pass" {
                    failed += 1;
                }
                row["gpu_after"] = ownership::snapshot(&directory, "after");
                eprintln!(
                    "{} seed={seed} step={step} {} {:.2}s",
                    arm.name(),
                    row["status"],
                    begin.elapsed().as_secs_f64()
                );
                storage::json_write(&directory.join("result.json"), &row)?;
                report["cases"].as_array_mut().unwrap().push(row);
                report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
                storage::json_write(&output.join("report.json"), &report)?;
            }
        }
    }
    match ledger.verify() {
        Ok(v) => report["inputs"] = v,
        Err(e) => {
            failed += 1;
            report["inputs"] = json!({"unchanged":false,"error":e.to_string()});
        }
    }
    report["failed"] = json!(failed);
    report["status"] = json!(if failed == 0 { "pass" } else { "fail" });
    storage::json_write(&output.join("report.json"), &report)?;
    require(failed == 0, "audit failed; all outcomes retained")
}
