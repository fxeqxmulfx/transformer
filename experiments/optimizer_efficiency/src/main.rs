//! Full-backpropagation optimizer efficiency; no local-learning variants.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../../residual_dynamics/src/algebra.rs"]
pub mod algebra;
mod benchmark;
mod fast_kernel;
#[path = "../../single_update/src/inputs.rs"]
pub mod inputs;
mod layout;
#[path = "../../forward_training/src/ownership.rs"]
mod ownership;
mod preflight;
mod profile;
mod quality;
mod state;
mod timer;
mod trainer;
mod training;
mod worker;
use domain::Result;
use infrastructure::storage;
use inputs::require;
use serde_json::json;
use std::{fs, path::Path};
fn source_hash() -> String {
    inputs::hash(
        &[
            include_bytes!("main.rs").as_slice(),
            include_bytes!("profile.rs"),
            include_bytes!("benchmark.rs"),
            include_bytes!("fast_kernel.rs"),
            include_bytes!("layout.rs"),
            include_bytes!("preflight.rs"),
            include_bytes!("quality.rs"),
            include_bytes!("trainer.rs"),
            include_bytes!("training.rs"),
            include_bytes!("worker.rs"),
            include_bytes!("timer.rs"),
            include_bytes!("state.rs"),
            include_bytes!("instrumentation.rs"),
            include_bytes!("../Cargo.toml"),
            include_bytes!("../Cargo.lock"),
            include_bytes!("../PROFILE_PROTOCOL.md"),
            include_bytes!("../PROTOCOL.md"),
            include_bytes!("../../single_update/build.rs"),
            include_bytes!("../../single_update/src/inputs.rs"),
            include_bytes!("../../forward_training/src/ownership.rs"),
            env!("TRAINER_SOURCE_SHA256").as_bytes(),
        ]
        .concat(),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    require(
        args.len() == 2 || (args.len() == 3 && args[0] == "compare"),
        "usage: rotate90-complex-kernite [compare] ORIGINAL_CHECKPOINT_ROOT NEW_OUTPUT_DIRECTORY",
    )?;
    let compare = args.len() == 3;
    let offset = usize::from(compare);
    let root = Path::new(&args[offset]).canonicalize()?;
    let output = Path::new(&args[offset + 1]);
    require(!output.exists(), "refusing to overwrite evidence")?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    require(
        !parent.starts_with(&root),
        "output must be outside checkpoint inputs",
    )?;
    let mut ledger = inputs::Ledger::default();
    let (original, protocol) = inputs::protocol(&mut ledger, &root)?;
    fs::create_dir(output)?;
    let mut report = json!({"status":"running","mode":if compare{"compare"}else{"profile"},"protocol":if compare{include_str!("../PROTOCOL.md")}else{include_str!("../PROFILE_PROTOCOL.md")},"source_sha256":source_hash(),"original_protocol":original,"candidate_ptx_sha256":inputs::hash(fast_kernel::kernels::PTX.as_bytes())});
    let path = output.join("report.json");
    storage::json_write(&path, &report)?;
    let outcome = (|| -> Result<_> {
        let cuda = kernite_cuda::Cuda::new()?;
        if compare {
            report["preflight"] = preflight::run(&root, output, &mut ledger, &protocol, &cuda)?;
            storage::json_write(&path, &report)?;
            require(
                report["preflight"]["passed"] == true,
                "bitwise preflight failed; no timing/training started",
            )?;
            report["benchmark"] = benchmark::run(&root, output, &mut ledger, &protocol, &cuda)?;
            storage::json_write(&path, &report)?;
            require(
                report["benchmark"]["passed"] == true,
                "paired workload check failed; training not started",
            )?;
            report["training"] = training::run(output, &cuda)?;
            Ok(report["training"]["valid_completed_cohort"] == true)
        } else {
            report["profile"] = profile::run(&root, output, &mut ledger, &protocol, &cuda)?;
            Ok(report["profile"]["passed"] == true)
        }
    })();
    let mut passed = false;
    match outcome {
        Ok(v) => {
            passed = v;
        }
        Err(e) => report["error"] = json!(e.to_string()),
    }
    match ledger.verify() {
        Ok(v) => report["inputs"] = v,
        Err(e) => {
            passed = false;
            report["inputs"] = json!({"unchanged":false,"error":e.to_string()});
        }
    }
    report["status"] = json!(if passed { "pass" } else { "fail" });
    storage::json_write(&path, &report)?;
    require(passed, "profile failed; available evidence retained")
}
