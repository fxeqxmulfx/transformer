//! Fingerprint the complete comparison source and its original task inventory.
use std::{env, fs, path::PathBuf};
fn main() {
    let here = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut files: Vec<_> = fs::read_dir(here.join("src"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|s| s == "rs"))
        .collect();
    for name in [
        "build.rs",
        "Cargo.toml",
        "Cargo.lock",
        "protocol.json",
        "run.sh",
        "freeze_request.sh",
    ] {
        files.push(here.join(name));
    }
    files.push(here.join("../convex_stress/suite.json"));
    files.sort();
    let mut code = String::from("pub const SOURCES: &[(&str, &[u8])] = &[\n");
    for file in files {
        println!("cargo:rerun-if-changed={}", file.display());
        let name = file.strip_prefix(here.parent().unwrap()).unwrap();
        code.push_str(&format!(
            "({:?}, include_bytes!({:?})),\n",
            name.to_str().unwrap(),
            file.to_str().unwrap()
        ));
    }
    code.push_str("];\n");
    fs::write(
        PathBuf::from(env::var("OUT_DIR").unwrap()).join("sources.rs"),
        code,
    )
    .unwrap();
}
