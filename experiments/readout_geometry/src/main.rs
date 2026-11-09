//! Task #198: observable spectral geometry and actual source-step attribution.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../../residual_dynamics/src/algebra.rs"]
pub mod algebra;
mod attribution;
#[path = "../../transport_forecast/src/audit.rs"]
mod audit;
#[path = "../../signed_transport/src/svd.rs"]
pub mod basis_svd;
#[path = "../../transport_forecast/src/bridge.rs"]
mod bridge;
#[path = "../../single_update/src/forcing.rs"]
pub mod forcing;
#[path = "../../single_update/src/forecast_jet.rs"]
pub mod forecast_jet;
#[path = "../../single_update/src/forecast_network.rs"]
pub mod forecast_network;
#[path = "../../residual_dynamics/src/geometry.rs"]
pub mod geometry;
#[path = "../../residual_dynamics/src/gradient.rs"]
pub mod gradient;
mod groups;
#[path = "../../single_update/src/inputs.rs"]
pub mod inputs;
#[path = "../../forecast_tightening/src/intervals.rs"]
pub mod intervals;
#[path = "../../forecast_tightening/src/jet.rs"]
pub mod jet;
#[path = "../../forecast_tightening/src/matrix.rs"]
pub mod matrix;
#[path = "../../single_update/src/measure.rs"]
pub mod measure;
#[path = "../../forecast_tightening/src/network.rs"]
pub mod network;
#[path = "../../flat_approx/fixtures/reference.rs"]
pub mod oracle;
#[path = "../../forecast_tightening/src/product.rs"]
pub mod product;
#[path = "../../single_update/src/query_step.rs"]
pub mod query_step;
mod reconstruction;
mod run;
#[path = "../../forecast_tightening/src/statistics.rs"]
pub mod statistics;
#[path = "../../residual_dynamics/src/svd.rs"]
pub mod svd;
#[path = "../../forecast_tightening/src/audit.rs"]
pub mod tight_audit;
use application::Backend;
use domain::{
    Result,
    data::{Data, Plan},
    model::Arm,
};
use infrastructure::{engine::Engine, storage};
use inputs::require;
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};
pub const STEPS: [usize; 13] = [
    0, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000,
];
fn source_hash() -> String {
    inputs::hash(
        &[
            include_bytes!("main.rs").as_slice(),
            include_bytes!("../../transport_forecast/src/audit.rs"),
            include_bytes!("../../transport_forecast/src/bridge.rs"),
            include_bytes!("../../single_update/src/forcing.rs"),
            include_bytes!("attribution.rs"),
            include_bytes!("groups.rs"),
            include_bytes!("reconstruction.rs"),
            include_bytes!("run.rs"),
            include_bytes!("instrumentation.rs"),
            include_bytes!("../Cargo.toml"),
            include_bytes!("../Cargo.lock"),
            include_bytes!("../PROTOCOL.md"),
            include_bytes!("../../signed_transport/src/svd.rs"),
            include_bytes!("../../forecast_tightening/src/audit.rs"),
            include_bytes!("../../forecast_tightening/src/intervals.rs"),
            include_bytes!("../../forecast_tightening/src/network.rs"),
            include_bytes!("../../forecast_tightening/src/matrix.rs"),
            include_bytes!("../../forecast_tightening/src/jet.rs"),
            include_bytes!("../../forecast_tightening/src/product.rs"),
            include_bytes!("../../forecast_tightening/src/statistics.rs"),
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
        "usage: rotate90-complex-kernite CHECKPOINT_ROOT TASK196_REPORT NEW_OUTPUT_DIRECTORY",
    )?;
    let root = Path::new(&args[0]).canonicalize()?;
    let output = Path::new(&args[2]);
    require(!output.exists(), "refusing to overwrite evidence")?;
    require(
        !output
            .parent()
            .unwrap_or(Path::new("."))
            .canonicalize()?
            .starts_with(&root),
        "output overlaps inputs",
    )?;
    let mut ledger = inputs::Ledger::default();
    let (original, protocol) = inputs::protocol(&mut ledger, &root)?;
    let reference = ledger.json(&Path::new(&args[1]).canonicalize()?)?;
    require(reference["status"] == "pass", "task #196 reference failed")?;
    fs::create_dir(output)?;
    algebra::threads();
    let started = Instant::now();
    let slots: Vec<_> = [7, 11]
        .into_iter()
        .flat_map(|seed| {
            STEPS
                .into_iter()
                .map(move |step| json!({"seed":seed,"step":step,"status":"pending"}))
        })
        .collect();
    let mut report = json!({"status":"running","source_sha256":source_hash(),"protocol":include_str!("../PROTOCOL.md"),"original_protocol":original,"cases":slots,"replay_hashes":[]});
    for (si, seed) in [7, 11].into_iter().enumerate() {
        let data = Data::new(seed);
        let plan = Plan::new(seed, 10001);
        let mut saved = vec![];
        for step in [0, 5000, 10000] {
            saved.push(inputs::checkpoint(
                &mut ledger,
                &root,
                &inputs::Case {
                    arm: Arm::Plain,
                    seed,
                    step,
                    data: &data,
                    plan: &plan,
                    protocol: &protocol,
                    forecast_output: output,
                },
            )?);
        }
        let history: Value = serde_json::from_slice(&ledger.read(&root.join(format!(
            "{}-seed{seed}/step-010000/history.json",
            Arm::Plain.name()
        )))?)?;
        let mut engine = Engine::new(Arm::Plain, &saved[0].state.parameters, &data, &plan)?;
        engine.restore(&saved[0].state)?;
        let mut completed = 0;
        let mut failed = false;
        for (i, step) in STEPS.into_iter().enumerate() {
            let slot = si * STEPS.len() + i;
            if failed || started.elapsed().as_secs_f64() >= 1200. {
                report["cases"][slot]["status"] = json!(if failed {
                    "skipped_after_error"
                } else {
                    "capped"
                });
                continue;
            }
            let directory = output.join(format!("plain-seed{seed}-step{step}"));
            fs::create_dir(&directory)?;
            let begin = Instant::now();
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Value> {
                    let replay = Instant::now();
                    while completed < step {
                        let at = plan.boundaries[completed];
                        engine.train_step(
                            at,
                            plan.boundaries[completed + 1] - at,
                            completed + 1,
                        )?;
                        completed += 1;
                    }
                    let state = engine.state()?;
                    let replay_seconds = replay.elapsed().as_secs_f64();
                    if let Some(index) = [0, 5000, 10000].iter().position(|&s| s == step) {
                        let bytes = storage::state_bytes(step, &state);
                        let digest = storage::hash(&bytes);
                        require(
                            saved[index].metadata["state_sha256"] == digest,
                            "replay changed original checkpoint",
                        )?;
                        report["replay_hashes"]
                            .as_array_mut()
                            .unwrap()
                            .push(json!({"seed":seed,"step":step,"sha256":digest,"matched":true}));
                    }
                    let expected = reference["cases"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|c| {
                            c["arm"] == Arm::Plain.name() && c["seed"] == seed && c["step"] == step
                        })
                        .and_then(|c| c["audit"]["state_sha256"].as_str());
                    let mut result = run::probe(
                        &mut engine,
                        &inputs::Case {
                            arm: Arm::Plain,
                            seed,
                            step,
                            data: &data,
                            plan: &plan,
                            protocol: &protocol,
                            forecast_output: &directory,
                        },
                        &state,
                        &directory,
                        &history,
                        expected,
                    )?;
                    completed = step + 1;
                    result["seconds"]["replay"] = json!(replay_seconds);
                    Ok(result)
                }));
            let mut row = json!({"seed":seed,"step":step,"post_budget_probe":step==10000,"seconds":begin.elapsed().as_secs_f64()});
            match result {
                Ok(Ok(v)) => {
                    row["status"] = json!(if v["passed"] == true { "pass" } else { "fail" });
                    row["audit"] = v;
                }
                Ok(Err(e)) => {
                    row["status"] = json!("error");
                    row["error"] = json!(e.to_string());
                    failed = true;
                }
                Err(_) => {
                    row["status"] = json!("error");
                    row["error"] = json!("panic; see stderr");
                    failed = true;
                }
            }
            eprintln!(
                "seed={seed} step={step} {} {:.2}s",
                row["status"],
                begin.elapsed().as_secs_f64()
            );
            storage::json_write(&directory.join("result.json"), &row)?;
            report["cases"][slot] = row;
            storage::json_write(&output.join("report.json"), &report)?;
        }
    }
    report["inputs"] = ledger.verify()?;
    let failed = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["status"] != "pass")
        .count();
    report["failed"] = json!(failed);
    report["status"] = json!(if failed == 0 { "pass" } else { "fail" });
    report["seconds"] = json!(started.elapsed().as_secs_f64());
    storage::json_write(&output.join("report.json"), &report)?;
    require(failed == 0, "bridge audit failed; all slots retained")
}
