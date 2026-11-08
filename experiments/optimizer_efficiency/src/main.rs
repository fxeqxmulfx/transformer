//! Full-backpropagation optimizer efficiency; no local-learning variants.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../../residual_dynamics/src/algebra.rs"]
pub mod algebra;
#[path = "../../single_update/src/inputs.rs"]
pub mod inputs;
#[path = "../../forward_training/src/ownership.rs"]
mod ownership;
mod profile;
mod state;
mod timer;
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
            include_bytes!("timer.rs"),
            include_bytes!("state.rs"),
            include_bytes!("instrumentation.rs"),
            include_bytes!("../Cargo.toml"),
            include_bytes!("../Cargo.lock"),
            include_bytes!("../PROFILE_PROTOCOL.md"),
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
    require(
        !parent.starts_with(&root),
        "output must be outside checkpoint inputs",
    )?;
    let mut ledger = inputs::Ledger::default();
    let (original, protocol) = inputs::protocol(&mut ledger, &root)?;
    fs::create_dir(output)?;
    let mut report = json!({"status":"running","protocol":include_str!("../PROFILE_PROTOCOL.md"),"source_sha256":source_hash(),"original_protocol":original});
    let path = output.join("report.json");
    storage::json_write(&path, &report)?;
    let outcome = (|| -> Result<_> {
        let cuda = kernite_cuda::Cuda::new()?;
        profile::run(&root, output, &mut ledger, &protocol, &cuda)
    })();
    let mut passed = false;
    match outcome {
        Ok(v) => {
            passed = v["passed"] == true;
            report["profile"] = v;
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
