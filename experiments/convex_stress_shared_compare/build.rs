//! Pin the new orchestration and both reviewed structural components. The
//! imported cohort/selector fingerprints bind data and all earlier sources.
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
    let experiments = here.parent().unwrap();
    let mut files = vec![];
    for package in [
        here.clone(),
        experiments.join("convex_stress_shared_roles/extension"),
        experiments.join("convex_stress_coverage_extension"),
        experiments.join("convex_stress_fallback"),
    ] {
        collect(&package.join("src"), &mut files);
        for file in ["Cargo.toml", "Cargo.lock", "protocol.json"] {
            files.push(package.join(file));
        }
    }
    for file in [
        "build.rs",
        "freeze_source.sh",
        "freeze_original.sh",
        "run_original.sh",
    ] {
        files.push(here.join(file));
    }
    files.sort();
    let mut output = String::from("pub const SOURCES: &[(&str,&[u8])] = &[\n");
    for file in files {
        println!("cargo:rerun-if-changed={}", file.display());
        output.push_str(&format!(
            "({:?},include_bytes!({:?})),\n",
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
