//! Inventory native sources; lower shared code contributes its own fingerprints.
use std::{
    env, fs,
    path::{Path, PathBuf},
};
fn collect(path: &Path, files: &mut Vec<PathBuf>) {
    println!("cargo:rerun-if-changed={}", path.display());
    for entry in fs::read_dir(path).expect("source directory") {
        let path = entry.expect("source entry").path();
        if path.is_dir() {
            collect(&path, files);
        } else if path.extension().is_some_and(|s| s == "rs") {
            files.push(path);
        }
    }
}
fn main() {
    let here = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let experiments = here.parent().unwrap().parent().unwrap();
    let mut files = vec![];
    for name in [
        "convex_stress_quotient",
        "convex_stress_consequences",
        "convex_stress_modular",
        "convex_stress_fibers",
        "convex_stress_tagged",
        "convex_stress_density",
    ] {
        let root = experiments.join(name);
        collect(&root.join("src"), &mut files);
        for file in ["Cargo.toml", "Cargo.lock"] {
            files.push(root.join(file));
        }
    }
    collect(&here.join("src"), &mut files);
    for file in [
        "build.rs",
        "Cargo.toml",
        "Cargo.lock",
        "protocol.json",
        "run_catalogue.sh",
        "original_protocol.json",
        "run_original.sh",
        "freeze_original.sh",
    ] {
        files.push(here.join(file));
    }
    files.push(experiments.join("convex_stress/suite.json"));
    files.sort();
    let mut output = String::from("pub const SOURCES: &[(&str, &[u8])] = &[\n");
    for file in files {
        println!("cargo:rerun-if-changed={}", file.display());
        output.push_str(&format!(
            "({:?}, include_bytes!({:?})),\n",
            file.strip_prefix(experiments).unwrap().to_str().unwrap(),
            file.to_str().unwrap()
        ));
    }
    output.push_str("];\n");
    fs::write(
        PathBuf::from(env::var("OUT_DIR").unwrap()).join("sources.rs"),
        output,
    )
    .unwrap();
}
