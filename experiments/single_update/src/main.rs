//! Actual source AdamW probes; see PROTOCOL.md, task #186 and trainer b70811e.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../../residual_dynamics/src/algebra.rs"]
pub mod algebra;
mod drift;
mod forcing;
mod forecast_jet;
mod forecast_network;
mod forecast_report;
#[path = "../../residual_dynamics/src/geometry.rs"]
pub mod geometry;
#[path = "../../residual_dynamics/src/gradient.rs"]
pub mod gradient;
mod inputs;
mod margins;
mod measure;
#[path = "../../flat_approx/fixtures/reference.rs"]
pub mod oracle;
mod probe;
mod query_step;
mod query_summary;
#[path = "../../residual_dynamics/src/svd.rs"]
pub mod svd;

use domain::{
    Result,
    data::{Data, Plan},
    model::Arm,
};
use serde_json::json;
use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    inputs::require(
        args.len() == 2,
        "usage: single-update-audit CHECKPOINT_ROOT OUTPUT_JSON",
    )?;
    let root = Path::new(&args[0]).canonicalize()?;
    let output = Path::new(&args[1]);
    inputs::require(
        !output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .canonicalize()?
            .starts_with(&root),
        "output must be outside original checkpoint root",
    )?;
    inputs::require(!output.exists(), "refusing to overwrite existing report")?;
    let forecast_output = output.with_extension("features.bin");
    let _forecast_file = fs::File::create_new(&forecast_output)?;
    let mut ledger = inputs::Ledger::default();
    let (original, protocol_hash) = inputs::protocol(&mut ledger, &root)?;
    let mut report = json!({"status":"running","protocol":include_str!("../PROTOCOL.md"),
        "margin_protocol":include_str!("../MARGIN_PROTOCOL.md"),
        "forecast_protocol":include_str!("../FORECAST_PROTOCOL.md"),
        "source_sha256":inputs::source_hash(),"original_protocol":original,"cases":[],"elapsed_seconds":0.});
    let save = |report: &serde_json::Value| -> Result<()> {
        let mut bytes = serde_json::to_vec_pretty(report)?;
        bytes.push(b'\n');
        fs::write(output, bytes)?;
        Ok(())
    };
    save(&report)?;
    algebra::threads();
    let started = Instant::now();
    let mut failed = 0;
    for seed in [7, 11] {
        let data = Data::new(seed);
        let plan = Plan::new(seed, 10001);
        for arm in Arm::ALL {
            for step in [0, 5000, 10000] {
                let begin = Instant::now();
                let case = inputs::Case {
                    arm,
                    seed,
                    step,
                    data: &data,
                    plan: &plan,
                    protocol: &protocol_hash,
                    forecast_output: &forecast_output,
                };
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
                        inputs::require(
                            started.elapsed() < Duration::from_secs(600),
                            "600-second audit cap reached",
                        )?;
                        let saved = inputs::checkpoint(&mut ledger, &root, &case)?;
                        probe::run(&case, &saved)
                    }));
                let mut record = json!({"arm":arm.name(),"seed":seed,"step":step,"next_step":step+1,
                    "post_budget_probe":step==10000,"seconds":begin.elapsed().as_secs_f64()});
                match result {
                    Ok(Ok(value)) => {
                        record["status"] = json!("pass");
                        record["audit"] = value;
                    }
                    Ok(Err(error)) => {
                        failed += 1;
                        record["status"] = json!("error");
                        record["error"] = json!(error.to_string());
                    }
                    Err(_) => {
                        failed += 1;
                        record["status"] = json!("error");
                        record["error"] = json!("panic; see stderr");
                    }
                }
                eprintln!(
                    "{} seed={} step={} {} {:.2}s",
                    arm.name(),
                    seed,
                    step,
                    record["status"],
                    begin.elapsed().as_secs_f64()
                );
                report["cases"].as_array_mut().unwrap().push(record);
                report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
                save(&report)?;
            }
        }
    }
    match ledger.verify() {
        Ok(value) => report["input_verification"] = value,
        Err(error) => {
            failed += 1;
            report["input_verification"] = json!({"unchanged":false,"error":error.to_string()});
        }
    }
    report["status"] = json!(if failed == 0 { "pass" } else { "fail" });
    report["failed"] = json!(failed);
    report["forecast_file"] = json!({"name":forecast_output.file_name().unwrap().to_str(),
        "sha256":inputs::hash(&fs::read(&forecast_output)?),"bytes":fs::metadata(&forecast_output)?.len()});
    save(&report)?;
    inputs::require(
        failed == 0,
        "audit failed; all case outcomes retained in output",
    )
}
