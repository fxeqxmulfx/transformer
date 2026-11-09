//! Frozen task #189 inference comparison; original source models and updates.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../../residual_dynamics/src/algebra.rs"]
pub mod algebra;
mod benchmark;
mod bounds;
mod cache_cpu;
mod device;
#[path = "../../single_update/src/forecast_jet.rs"]
pub mod forecast_jet;
mod gather_gpu;
#[path = "../../residual_dynamics/src/geometry.rs"]
pub mod geometry;
#[path = "../../residual_dynamics/src/gradient.rs"]
pub mod gradient;
#[path = "../../single_update/src/inputs.rs"]
pub mod inputs;
mod io;
#[path = "../../forecast_tightening/src/matrix.rs"]
pub mod matrix;
#[path = "../../single_update/src/measure.rs"]
pub mod measure;
#[path = "../../flat_approx/fixtures/reference.rs"]
pub mod oracle;
#[path = "../../forward_training/src/ownership.rs"]
pub mod ownership;
mod quality;
mod run;
mod serving;
#[path = "../../residual_dynamics/src/svd.rs"]
pub mod svd;
#[path = "../../optimizer_efficiency/src/timer.rs"]
pub mod timer;
fn main() -> domain::Result<()> {
    Err("Inference comparison is a draft; its cohort runner is not wired up.".into())
}
