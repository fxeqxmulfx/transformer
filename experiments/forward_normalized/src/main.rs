//! Frozen standalone normalized-A training, with unchanged local objectives.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../../residual_dynamics/src/algebra.rs"]
pub mod algebra;
mod audit;
#[path = "../../forward_training/src/config.rs"]
pub mod config;
#[path = "../../forward_alignment/src/coordinates.rs"]
pub mod coordinates;
#[path = "../../single_update/src/inputs.rs"]
pub mod inputs;
#[path = "../../forward_forward/src/kernel.rs"]
pub mod kernel;
#[path = "../../forward_training/src/audit.rs"]
mod legacy_audit;
#[path = "../../forward_alignment/src/pins.rs"]
pub mod legacy_pins;
#[path = "../../forward_training/src/local_norm.rs"]
pub mod local_norm;
#[path = "../../forward_forward/src/loss.rs"]
pub mod loss;
mod normalized;
mod normalized_kernel;
#[path = "../../flat_approx/fixtures/reference.rs"]
pub mod oracle;
#[path = "../../forward_training/src/ownership.rs"]
mod ownership;
#[path = "../../forward_training/src/reference.rs"]
pub mod reference;
mod rule;
mod runner;
mod summary;
mod update_audit;
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
            include_bytes!("normalized.rs"),
            include_bytes!("normalized_kernel.rs"),
            include_bytes!("instrumentation.rs"),
            include_bytes!("rule.rs"),
            include_bytes!("runner.rs"),
            include_bytes!("worker.rs"),
            include_bytes!("summary.rs"),
            include_bytes!("update_audit.rs"),
            include_bytes!("../Cargo.toml"),
            include_bytes!("../Cargo.lock"),
            include_bytes!("../PROTOCOL.md"),
            include_bytes!("../../forward_alignment/src/coordinates.rs"),
            legacy_pins::training_hash().as_bytes(),
        ]
        .concat(),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    require(
        args.len() == 2,
        "usage: rotate90-complex-kernite ORIGINAL_CHECKPOINT_ROOT NEW_OUTPUT_DIRECTORY",
    )?;
    let root = Path::new(&args[0]).canonicalize()?;
    let output = Path::new(&args[1]);
    require(!output.exists(), "refusing to overwrite evidence")?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    require(!parent.starts_with(&root), "output must be outside inputs")?;
    let mut ledger = inputs::Ledger::default();
    let (original, protocol) = inputs::protocol(&mut ledger, &root)?;
    legacy_pins::verify()?;
    fs::create_dir(output)?;
    let mut report = json!({"status":"preflight","protocol":include_str!("../PROTOCOL.md"),
        "source_sha256":source_hash(),"pilot_source_sha256":legacy_pins::PILOT_SOURCE,"original_protocol":original,
        "normalized_ptx_sha256":inputs::hash(normalized_kernel::kernels::PTX.as_bytes()),
        "timing_exclusions":["compilation and JIT warmup","numerical preflight","waiting for other methods","GPU ownership polling","report serialization and checkpoint I/O"],
        "memory_scope":"allocated GPU buffer payload including retained A moments and global-backward buffers"});
    let path = output.join("report.json");
    storage::json_write(&path, &report)?;
    let outcome = (|| -> Result<_> {
        let cuda = kernite_cuda::Cuda::new()?;
        let kernels = config::Kernels::new(&cuda)?;
        let normalized = normalized_kernel::kernels::Kernels::load(&cuda)?;
        algebra::threads();
        report["legacy_preflight"] = legacy_audit::run(
            &root,
            &output.join("legacy-preflight.json"),
            &mut ledger,
            &protocol,
            &cuda,
            &kernels,
        )?;
        report["preflight"] = audit::run(
            &root,
            output,
            &mut ledger,
            &protocol,
            &cuda,
            &kernels,
            &normalized,
        )?;
        report["status"] = json!("training");
        storage::json_write(&path, &report)?;
        runner::run(output, &cuda, &kernels, &normalized)
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
    storage::json_write(&path, &report)?;
    require(
        passed,
        "comparison failed or capped; available evidence retained",
    )
}
