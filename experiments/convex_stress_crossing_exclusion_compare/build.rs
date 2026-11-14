//! Bind this adapter and the reviewed hull-crossing component. The parent
//! fingerprint binds the unchanged generator, metrics and earlier dependencies.
use std::{
    env, fs,
    path::{Path, PathBuf},
};
fn collect(path: &Path, files: &mut Vec<PathBuf>) {
    println!("cargo:rerun-if-changed={}", path.display());
    for entry in fs::read_dir(path).unwrap() {
        let path = entry.unwrap().path();
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
        experiments.join("convex_stress_crossing_exclusion"),
    ] {
        collect(&package.join("src"), &mut files);
        for file in ["Cargo.toml", "Cargo.lock", "protocol.json"] {
            files.push(package.join(file));
        }
    }
    for file in [
        "build.rs",
        "parent.jq",
        "reference.jq",
        "run_original.sh",
        "report.jq",
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
