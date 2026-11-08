//! Pin the reused local implementation as well as its generated kernels.
use crate::{
    domain::Result,
    inputs::{self, require},
};
pub const PILOT_SOURCE: &str = "0f63edd42b7aa9968345bdd7b3815c9d3c9b098d378944449a18f174c2f37aa8";
pub fn training_hash() -> String {
    inputs::hash(
        &[
            include_bytes!("../../forward_training/src/main.rs").as_slice(),
            include_bytes!("../../forward_training/src/audit.rs"),
            include_bytes!("../../forward_training/src/config.rs"),
            include_bytes!("../../forward_training/src/instrumentation.rs"),
            include_bytes!("../../forward_training/src/local_norm.rs"),
            include_bytes!("../../forward_training/src/reference.rs"),
            include_bytes!("../../forward_training/src/runner.rs"),
            include_bytes!("../../forward_training/src/worker.rs"),
            include_bytes!("../../forward_training/src/summary.rs"),
            include_bytes!("../../forward_training/src/ownership.rs"),
            include_bytes!("../../forward_forward/src/loss.rs"),
            include_bytes!("../../forward_forward/src/kernel.rs"),
            include_bytes!("../../forward_training/PROTOCOL.md"),
            include_bytes!("../../forward_training/Cargo.toml"),
            include_bytes!("../../forward_training/Cargo.lock"),
            inputs::source_hash().as_bytes(),
        ]
        .concat(),
    )
}
pub fn verify() -> Result<()> {
    require(
        training_hash() == PILOT_SOURCE,
        "reused pilot source changed",
    )?;
    require(
        inputs::hash(crate::kernel::kernels::PTX.as_bytes())
            == "47d42ba5da6bdaa57adaae76cce739d382d75134102a8af294018b92f8556ab3",
        "pair PTX changed",
    )?;
    require(
        inputs::hash(crate::local_norm::kernels::PTX.as_bytes())
            == "947434a396f64bca795a46dd39042fbacb90252f245a791e3585752f002a85c2",
        "auxiliary RMS PTX changed",
    )
}
pub fn source_hash() -> String {
    inputs::hash(
        &[
            include_bytes!("main.rs").as_slice(),
            include_bytes!("pins.rs"),
            include_bytes!("cohort.rs"),
            include_bytes!("probe.rs"),
            include_bytes!("coordinates.rs"),
            include_bytes!("finite.rs"),
            include_bytes!("measure.rs"),
            include_bytes!("signals.rs"),
            include_bytes!("instrumentation.rs"),
            include_bytes!("../Cargo.toml"),
            include_bytes!("../Cargo.lock"),
            include_bytes!("../PROTOCOL.md"),
            training_hash().as_bytes(),
        ]
        .concat(),
    )
}
