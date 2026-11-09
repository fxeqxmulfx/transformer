//! Reproduce the original Rotate90 testbed with the selected main-model optimizer.
include!(concat!(env!("OUT_DIR"), "/modules.rs"));
#[path = "../../../residual_dynamics/src/algebra.rs"]
pub mod algebra;
#[path = "../../../signed_transport/src/svd.rs"]
pub mod basis_svd;
mod checkpoints;
#[path = "../../../class_geometry/src/classes.rs"]
pub mod classes;
#[path = "../../../class_geometry_wsd/src/dynamics.rs"]
pub mod dynamics;
#[path = "../../../residual_dynamics/src/geometry.rs"]
pub mod geometry;
#[path = "../../../residual_dynamics/src/gradient.rs"]
pub mod gradient;
#[path = "../../../class_geometry/src/groups.rs"]
pub mod groups;
#[path = "../../../single_update/src/inputs.rs"]
pub mod inputs;
#[path = "../../../single_update/src/measure.rs"]
pub mod measure;
#[path = "../../../flat_approx/src/model.rs"]
pub mod model;
#[path = "../../../class_geometry_wsd/src/optimizer.rs"]
pub mod optimizer;
#[path = "../../../class_geometry_wsd/src/optimizer_kernels.rs"]
pub mod optimizer_kernels;
#[path = "../../../flat_approx/fixtures/reference.rs"]
pub mod oracle;
#[path = "../../../class_geometry/src/probe.rs"]
pub mod probe;
mod provenance;
#[path = "../../../class_geometry/src/reconstruction.rs"]
pub mod reconstruction;
mod runner;
#[path = "../../../class_geometry_wsd/src/schedule.rs"]
pub mod schedule;
#[path = "../../../residual_dynamics/src/svd.rs"]
pub mod svd;
mod training;
pub use domain::Result;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [command] if command == "preflight" => {
            println!(
                "{}",
                serde_json::json!({"provenance":provenance::metadata()?,"optimizer":optimizer::preflight()?,"source":training::source_preflight()?})
            );
            Ok(())
        }
        [command, root] if command == "run" => runner::run(std::path::Path::new(root)),
        _ => Err("usage: rotate90-complex-kernite preflight | run COHORT_OUTPUT_ROOT".into()),
    }
}
