//! Bind every production dependency, the frozen data stack and the launch recipe.
use super::*;
pub fn fingerprint() -> String {
    let parent = convex_stress_decode::cohort::fingerprint();
    let mut bytes = vec![];
    for source in [
        include_bytes!("../main.rs").as_slice(),
        include_bytes!("mod.rs"),
        include_bytes!("fingerprint.rs"),
        include_bytes!("run.rs"),
        include_bytes!("summary.rs"),
        include_bytes!("../../Cargo.toml"),
        include_bytes!("../../Cargo.lock"),
        include_bytes!("../../run_cohort.sh"),
        include_bytes!("../../../Cargo.toml"),
        include_bytes!("../../../Cargo.lock"),
        include_bytes!("../../../src/lib.rs"),
        include_bytes!("../../../src/main.rs"),
        include_bytes!("../../../src/affine.rs"),
        include_bytes!("../../../src/artifact.rs"),
        include_bytes!("../../../src/audit.rs"),
        include_bytes!("../../../src/comparison.rs"),
        include_bytes!("../../../src/interval.rs"),
        include_bytes!("../../../src/interval_audit.rs"),
        include_bytes!("../../../src/interval_search.rs"),
        include_bytes!("../../../src/matrix.rs"),
        include_bytes!("../../../src/search.rs"),
        include_bytes!("../../../src/solver.rs"),
        include_bytes!("../../../src/assignment_certificate.rs"),
        include_bytes!("../../../src/refinement.rs"),
        include_bytes!("../../../src/refinement_audit.rs"),
        include_bytes!("../../../src/refinement_search.rs"),
        SUITE.as_bytes(),
        PROTOCOL.as_bytes(),
        parent.as_bytes(),
    ] {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    hash(&bytes)
}
