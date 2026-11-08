//! Reuse source and kernels that produced both earlier diagnostics.
use crate::{
    alignment_pins,
    domain::Result,
    inputs::{self, require},
};
pub use alignment_pins::PILOT_SOURCE;
pub const ALIGNMENT_SOURCE: &str =
    "b2d670389e3dbfb36f1b95b42edfbe9fc90d3a94dfa8a95d06f148f8c034d3ee";
pub fn verify() -> Result<()> {
    alignment_pins::verify()?;
    require(
        alignment_pins::source_hash() == ALIGNMENT_SOURCE,
        "reused alignment source changed",
    )
}
pub fn source_hash() -> String {
    inputs::hash(
        &[
            include_bytes!("main.rs").as_slice(),
            include_bytes!("pins.rs"),
            include_bytes!("cohort.rs"),
            include_bytes!("probe.rs"),
            include_bytes!("directions.rs"),
            include_bytes!("instrumentation.rs"),
            include_bytes!("../Cargo.toml"),
            include_bytes!("../Cargo.lock"),
            include_bytes!("../PROTOCOL.md"),
            alignment_pins::source_hash().as_bytes(),
        ]
        .concat(),
    )
}
