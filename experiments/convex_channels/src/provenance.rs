//! Hash the actual training sources, including the reused transformer backend.

use sha2::{Digest, Sha256};
use std::{fs, path::Path};

pub fn source_hash() -> String {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut hash = Sha256::new();
    for name in [
        "Cargo.toml",
        "Cargo.lock",
        "src/main.rs",
        "src/lib.rs",
        "src/channel.rs",
        "src/task.rs",
        "src/provenance.rs",
        "../convex_recall_compare/Cargo.toml",
        "../convex_recall_compare/src/lib.rs",
        "../convex_recall_compare/src/convex.rs",
        "../convex_recall_compare/src/data.rs",
        "../convex_recall_compare/src/engine.rs",
        "../convex_recall_compare/src/transformer_model.rs",
        "../grok_division/src/adam.rs",
        "../grok_division/src/layout.rs",
        "../grok_division/src/gemm.rs",
        "../grok_division/src/mt.rs",
    ] {
        hash.update(name.as_bytes());
        hash.update(fs::read(base.join(name)).expect("source fingerprint input missing"));
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
