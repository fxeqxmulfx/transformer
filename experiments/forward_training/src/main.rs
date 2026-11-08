//! Full native local-learning pilot, from the frozen protocol at 34da0951.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../../residual_dynamics/src/algebra.rs"]
pub mod algebra;
mod audit;
mod config;
#[path = "../../single_update/src/inputs.rs"]
pub mod inputs;
#[path = "../../forward_forward/src/kernel.rs"]
pub mod kernel;
mod local_norm;
#[path = "../../forward_forward/src/loss.rs"]
pub mod loss;
#[path = "../../flat_approx/fixtures/reference.rs"]
pub mod oracle;
mod ownership;
mod reference;
mod runner;
mod summary;
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
            include_bytes!("audit.rs"),
            include_bytes!("config.rs"),
            include_bytes!("instrumentation.rs"),
            include_bytes!("local_norm.rs"),
            include_bytes!("reference.rs"),
            include_bytes!("runner.rs"),
            include_bytes!("worker.rs"),
            include_bytes!("summary.rs"),
            include_bytes!("ownership.rs"),
            include_bytes!("../../forward_forward/src/loss.rs"),
            include_bytes!("../../forward_forward/src/kernel.rs"),
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
        "usage: rotate90-complex-kernite CHECKPOINT_ROOT NEW_OUTPUT_DIRECTORY",
    )?;
    let root = Path::new(&args[0]).canonicalize()?;
    let output = Path::new(&args[1]);
    require(
        !output.exists(),
        "refusing to overwrite evidence; use a new output directory",
    )?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    require(
        !parent.starts_with(&root),
        "output must be outside original checkpoint directory",
    )?;
    let mut ledger = inputs::Ledger::default();
    let (original, protocol) = inputs::protocol(&mut ledger, &root)?;
    let pair_ptx = inputs::hash(kernel::kernels::PTX.as_bytes());
    require(
        pair_ptx == "47d42ba5da6bdaa57adaae76cce739d382d75134102a8af294018b92f8556ab3",
        "pair kernel changed from task #192",
    )?;
    fs::create_dir(output)?;
    let mut report = json!({"status":"preflight","source_sha256":source_hash(),"protocol":include_str!("../PROTOCOL.md"),
        "original_protocol":original,"pair_kernel_ptx_sha256":pair_ptx,
        "auxiliary_norm_ptx_sha256":inputs::hash(local_norm::kernels::PTX.as_bytes()),
        "timing_exclusions":["compilation and JIT warmup","numerical preflight","waiting for other methods",
            "GPU ownership polling","report serialization and checkpoint I/O"],
        "memory_scope":"allocated GPU buffer payload only; includes retained global-backward buffers"});
    let report_path = output.join("report.json");
    storage::json_write(&report_path, &report)?;
    let outcome = (|| -> Result<_> {
        let cuda = kernite_cuda::Cuda::new()?;
        let kernels = config::Kernels::new(&cuda)?;
        algebra::threads();
        report["preflight"] = audit::run(
            &root,
            &output.join("preflight.json"),
            &mut ledger,
            &protocol,
            &cuda,
            &kernels,
        )?;
        report["status"] = json!("training");
        storage::json_write(&report_path, &report)?;
        runner::run(&root, output, &mut ledger, &cuda, &kernels)
    })();
    let mut passed = false;
    match outcome {
        Ok(training) => {
            passed = training["valid_completed_cohort"] == true;
            report["training"] = training;
        }
        Err(e) => report["error"] = json!(e.to_string()),
    }
    match ledger.verify() {
        Ok(v) => report["input_verification"] = v,
        Err(e) => {
            passed = false;
            report["input_verification"] = json!({"unchanged":false,"error":e.to_string()});
        }
    }
    report["status"] = json!(if passed {
        "complete"
    } else {
        "incomplete-or-invalid"
    });
    storage::json_write(&report_path, &report)?;
    require(
        passed,
        "pilot incomplete or invalid; all available evidence retained",
    )
}
