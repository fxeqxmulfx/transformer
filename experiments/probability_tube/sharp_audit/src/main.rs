//! Same frozen comparison with the independent PA1-PA2 encoder refinement.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../../../residual_dynamics/src/algebra.rs"]
pub mod algebra;
mod batch;
#[path = "../../reproduce/src/checkpoints.rs"]
pub mod checkpoints;
#[path = "../../../ce_metric/classes.rs"]
pub mod classes;
mod compare;
#[path = "../../audit/src/construct.rs"]
mod construct;
#[path = "../../audit/src/evidence.rs"]
mod evidence;
#[path = "../../audit/src/gpu.rs"]
mod gpu;
#[path = "../../audit/src/guarded.rs"]
mod guarded;
pub mod inputs {
    pub use crate::evidence::require;
}
#[path = "../../../ce_metric/analysis.rs"]
pub mod metric_analysis;
#[path = "../../../flat_approx/src/model.rs"]
pub mod model;
#[path = "../../../class_geometry_wsd/src/optimizer_kernels.rs"]
pub mod optimizer_kernels;
#[path = "../../../flat_approx/fixtures/reference.rs"]
pub mod oracle;
mod prediction;
mod provenance;
#[path = "../../../ce_lyapunov/analysis.rs"]
pub mod reference;
#[path = "../../../ce_metric/state.rs"]
pub mod state;
#[path = "../../audit/src/verify.rs"]
mod verify;
pub use domain::Result;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd,out,a,b,c] if cmd=="construct"=>construct::run(std::path::Path::new(out),[a,b,c]),
        [cmd,out] if cmd=="verify"=>verify::run(std::path::Path::new(out)),
        [cmd,original,refined] if cmd=="compare"=>compare::run(std::path::Path::new(original),std::path::Path::new(refined)),
        _=>Err("usage: rotate90-complex-kernite construct OUTPUT LR001 LR01 SETTLE | verify OUTPUT | compare ORIGINAL REFINED".into()),
    }
}
