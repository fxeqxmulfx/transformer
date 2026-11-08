//! Fixed hidden-block direction/scale probes; never a continued training run.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../../residual_dynamics/src/algebra.rs"]
pub mod algebra;
#[path = "../../forward_alignment/src/pins.rs"]
pub mod alignment_pins;
mod cohort;
#[path = "../../forward_training/src/config.rs"]
pub mod config;
#[path = "../../forward_alignment/src/coordinates.rs"]
pub mod coordinates;
mod directions;
#[path = "../../forward_alignment/src/finite.rs"]
pub mod finite;
#[path = "../../single_update/src/inputs.rs"]
pub mod inputs;
#[path = "../../forward_forward/src/kernel.rs"]
pub mod kernel;
#[path = "../../forward_training/src/local_norm.rs"]
pub mod local_norm;
#[path = "../../forward_forward/src/loss.rs"]
pub mod loss;
#[path = "../../forward_alignment/src/measure.rs"]
pub mod measure;
#[path = "../../flat_approx/fixtures/reference.rs"]
pub mod oracle;
#[path = "../../forward_training/src/ownership.rs"]
mod ownership;
mod pins;
mod probe;
#[path = "../../forward_training/src/reference.rs"]
pub mod reference;
use domain::Result;
use infrastructure::storage;
use inputs::require;
use serde_json::json;
use std::{fs, path::Path};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    require(
        args.len() == 4,
        "usage: rotate90-complex-kernite ORIGINAL_CHECKPOINT_ROOT PILOT_DIRECTORY ALIGNMENT_DIRECTORY NEW_OUTPUT_DIRECTORY",
    )?;
    let root = Path::new(&args[0]).canonicalize()?;
    let pilot = Path::new(&args[1]).canonicalize()?;
    let alignment = Path::new(&args[2]).canonicalize()?;
    let output = Path::new(&args[3]);
    require(!output.exists(), "refusing to overwrite evidence")?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    require(
        [&root, &pilot, &alignment]
            .iter()
            .all(|p| !parent.starts_with(p)),
        "output must be outside input directories",
    )?;
    let mut ledger = inputs::Ledger::default();
    let (original, protocol) = inputs::protocol(&mut ledger, &root)?;
    pins::verify()?;
    fs::create_dir(output)?;
    let mut report = json!({"status":"running","source_sha256":pins::source_hash(),"protocol":include_str!("../PROTOCOL.md"),
        "original_protocol":original,"pilot_source_sha256":pins::PILOT_SOURCE,"alignment_source_sha256":pins::ALIGNMENT_SOURCE,
        "speed_claim":false});
    let report_path = output.join("report.json");
    storage::json_write(&report_path, &report)?;
    let outcome = (|| -> Result<_> {
        let cuda = kernite_cuda::Cuda::new()?;
        let kernels = config::Kernels::new(&cuda)?;
        algebra::threads();
        report["gpu_before"] = ownership::snapshot(output, "before");
        let value = cohort::run(&mut cohort::Context {
            root: &root,
            pilot: &pilot,
            alignment: &alignment,
            output,
            ledger: &mut ledger,
            protocol: &protocol,
            cuda: &cuda,
            kernels: &kernels,
        });
        report["gpu_after"] = ownership::snapshot(output, "after");
        value
    })();
    let mut passed = false;
    match outcome {
        Ok(value) => {
            passed = value["passed"] == true;
            report["cohort"] = value;
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
    report["status"] = json!(if passed { "pass" } else { "fail" });
    storage::json_write(&report_path, &report)?;
    require(passed, "diagnostic failed; all available evidence retained")
}
