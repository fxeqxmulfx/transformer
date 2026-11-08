//! Source-preserving local-loss diagnostic. Run with checkpoint root and JSON path.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../../residual_dynamics/src/algebra.rs"]
pub mod algebra;
mod audit;
#[path = "../../single_update/src/inputs.rs"]
pub mod inputs;
pub mod kernel;
pub mod loss;

use domain::{
    Result,
    data::{Data, Plan},
    model::Arm,
};
use inputs::require;
use serde_json::json;
use std::{
    path::Path,
    time::{Duration, Instant},
};

fn source_hash() -> String {
    inputs::hash(
        &[
            include_bytes!("main.rs").as_slice(),
            include_bytes!("audit.rs"),
            include_bytes!("loss.rs"),
            include_bytes!("kernel.rs"),
            include_bytes!("instrumentation.rs"),
            include_bytes!("../PROTOCOL.md"),
            include_bytes!("../Cargo.toml"),
            include_bytes!("../Cargo.lock"),
            inputs::source_hash().as_bytes(),
        ]
        .concat(),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    require(
        args.len() == 2,
        "usage: forward-forward-loss CHECKPOINT_ROOT OUTPUT_JSON",
    )?;
    let root = Path::new(&args[0]).canonicalize()?;
    let output = Path::new(&args[1]);
    require(!output.exists(), "refusing to overwrite prior evidence")?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    require(
        !parent.starts_with(&root),
        "output must be outside checkpoint root",
    )?;
    let mut ledger = inputs::Ledger::default();
    let (original, protocol) = inputs::protocol(&mut ledger, &root)?;
    let cuda = kernite_cuda::Cuda::new()?;
    let kernels = kernel::kernels::Kernels::load(&cuda)?;
    algebra::threads();
    let edges = audit::kernel_edges(&cuda, &kernels)?;
    let mut failures = usize::from(edges["violations"] != 0);
    let mut report = json!({"status":"running","protocol":include_str!("../PROTOCOL.md"),
        "source_sha256":source_hash(),"original_protocol":original,
        "loss_kernel_ptx_sha256":inputs::hash(kernel::kernels::PTX.as_bytes()),"kernel_edges":edges,
        "cases":[],"elapsed_seconds":0.});
    infrastructure::storage::json_write(output, &report)?;
    let started = Instant::now();
    for seed in [7, 11] {
        let data = Data::new(seed);
        let plan = Plan::new(seed, 10001);
        for arm in Arm::ALL {
            for step in [0, 5000, 10000] {
                let began = Instant::now();
                let case = inputs::Case {
                    arm,
                    seed,
                    step,
                    data: &data,
                    plan: &plan,
                    protocol: &protocol,
                    forecast_output: output,
                };
                let outcome =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
                        require(
                            started.elapsed() < Duration::from_secs(600),
                            "600-second cohort cap reached",
                        )?;
                        let saved = inputs::checkpoint(&mut ledger, &root, &case)?;
                        audit::run(&case, &saved, &kernels)
                    }));
                let mut row = json!({"arm":arm.name(),"seed":seed,"step":step});
                match outcome {
                    Ok(Ok(value)) => {
                        let passed = value["passed"] == true;
                        failures += usize::from(!passed);
                        row["status"] = json!(if passed { "pass" } else { "fail" });
                        row["audit"] = value;
                    }
                    Ok(Err(error)) => {
                        failures += 1;
                        row["status"] = json!("error");
                        row["error"] = json!(error.to_string());
                    }
                    Err(_) => {
                        failures += 1;
                        row["status"] = json!("error");
                        row["error"] = json!("panic; see diagnostic log");
                    }
                }
                row["seconds"] = json!(began.elapsed().as_secs_f64());
                eprintln!(
                    "{} seed={} step={} {} {:.2}s",
                    arm.name(),
                    seed,
                    step,
                    row["status"],
                    began.elapsed().as_secs_f64()
                );
                report["cases"].as_array_mut().unwrap().push(row);
                report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
                infrastructure::storage::json_write(output, &report)?;
            }
        }
    }
    match ledger.verify() {
        Ok(v) => report["input_verification"] = v,
        Err(e) => {
            failures += 1;
            report["input_verification"] = json!({"unchanged":false,"error":e.to_string()});
        }
    }
    report["failed"] = json!(failures);
    report["status"] = json!(if failures == 0 { "pass" } else { "fail" });
    infrastructure::storage::json_write(output, &report)?;
    require(failures == 0, "diagnostic failed; every case retained")
}
