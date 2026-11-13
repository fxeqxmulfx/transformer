//! Compile-time source inventory, including the native generator itself.
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
    let here = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let experiments = here.parent().unwrap().parent().unwrap();
    let mut files = vec![];
    for name in [
        "convex_stress_consequences",
        "convex_stress_modular",
        "convex_stress_fibers",
        "convex_stress_tagged",
    ] {
        let root = experiments.join(name);
        collect(&root.join("src"), &mut files);
        for f in ["Cargo.toml", "Cargo.lock"] {
            files.push(root.join(f));
        }
    }
    collect(&here.join("src"), &mut files);
    for f in [
        "build.rs",
        "Cargo.toml",
        "Cargo.lock",
        "protocol.json",
        "run_cohort.sh",
    ] {
        files.push(here.join(f));
    }
    files.push(experiments.join("convex_stress/suite.json"));
    files.sort();
    let mut output = String::from("pub const SOURCES: &[(&str, &[u8])] = &[\n");
    for file in files {
        println!("cargo:rerun-if-changed={}", file.display());
        let label = file.strip_prefix(experiments).unwrap().to_str().unwrap();
        output.push_str(&format!(
            "({label:?}, include_bytes!({:?})),\n",
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
