//! The frozen QG comparison keeps construction separate from future evidence.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../../../residual_dynamics/src/algebra.rs"]
pub mod algebra;
#[path = "../../audit/src/batch.rs"]
pub mod batch;
#[path = "../../reproduce/src/checkpoints.rs"]
pub mod checkpoints;
#[path = "../../../ce_metric/classes.rs"]
pub mod classes;
mod construct;
mod endpoint;
#[path = "../../audit/src/evidence.rs"]
pub mod evidence;
#[path = "../../audit/src/gpu.rs"]
mod gpu;
#[cfg(test)]
mod harness_tests;
pub mod inputs {
    pub use crate::evidence::require;
}
#[path = "../../../ce_metric/analysis.rs"]
pub mod metric_analysis;
#[path = "../../../flat_approx/src/model.rs"]
pub mod model;
mod observe;
#[path = "../../../class_geometry_wsd/src/optimizer_kernels.rs"]
pub mod optimizer_kernels;
#[path = "../../../flat_approx/fixtures/reference.rs"]
pub mod oracle;
mod prediction;
mod provenance;
#[path = "../../../ce_lyapunov/analysis.rs"]
pub mod reference;
mod sources;
#[path = "../../../ce_metric/state.rs"]
pub mod state;
mod verify;
pub use domain::Result;
pub const CAP: f64 = 7200.;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    probability_gram::single_thread();
    match args.as_slice() {
        [cmd, out, original, refined] if cmd == "construct" => {
            construct::run(std::path::Path::new(out), [original, refined])
        }
        [cmd, out] if cmd == "verify" => verify::run(std::path::Path::new(out)),
        _ => Err(
            "usage: rotate90-complex-kernite construct OUTPUT ORIGINAL REFINED | verify OUTPUT"
                .into(),
        ),
    }
}
