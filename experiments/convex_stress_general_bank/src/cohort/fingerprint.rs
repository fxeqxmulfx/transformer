//! Bind production code, every new dependency and the existing frozen data stack.
use super::*;
pub fn fingerprint() -> String {
    let shared = convex_stress_relabel::cohort::fingerprint();
    let mut bytes = vec![];
    for source in [
        include_bytes!("../lib.rs").as_slice(),
        include_bytes!("../main.rs"),
        include_bytes!("../search.rs"),
        include_bytes!("mod.rs"),
        include_bytes!("fingerprint.rs"),
        include_bytes!("input.rs"),
        include_bytes!("run.rs"),
        include_bytes!("replay.rs"),
        include_bytes!("audit.rs"),
        include_bytes!("summary.rs"),
        include_bytes!("../../Cargo.toml"),
        include_bytes!("../../Cargo.lock"),
        include_bytes!("../../run_cohort.sh"),
        include_bytes!("../../limited.sh"),
        include_bytes!("../../../convex_stress_general/src/lib.rs"),
        include_bytes!("../../../convex_stress_general/src/model.rs"),
        include_bytes!("../../../convex_stress_general/src/solver.rs"),
        include_bytes!("../../../convex_stress_general/src/fft.rs"),
        include_bytes!("../../../convex_stress_general/src/projection.rs"),
        include_bytes!("../../../convex_stress_general/src/endpoint.rs"),
        include_bytes!("../../../convex_stress_general/Cargo.toml"),
        include_bytes!("../../../convex_stress_propagation/src/lib.rs"),
        include_bytes!("../../../convex_stress_propagation/src/types.rs"),
        include_bytes!("../../../convex_stress_propagation/src/preflight.rs"),
        include_bytes!("../../../convex_stress_propagation/src/algorithm.rs"),
        include_bytes!("../../../convex_stress_propagation/src/check.rs"),
        include_bytes!("../../../convex_stress_propagation/Cargo.toml"),
        SUITE.as_bytes(),
        PROTOCOL.as_bytes(),
        shared.as_bytes(),
    ] {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    hash(&bytes)
}
