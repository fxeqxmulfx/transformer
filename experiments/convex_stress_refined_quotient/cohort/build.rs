//! Pin this adapter and its numerical component; the parent pins its dependencies.
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
    let component = here.parent().unwrap();
    let experiments = component.parent().unwrap();
    let mut files = vec![];
    for directory in [&here, component] {
        collect(&directory.join("src"), &mut files);
        for file in ["Cargo.toml", "Cargo.lock"] {
            files.push(directory.join(file));
        }
    }
    for file in [
        "build.rs",
        "protocol.json",
        "run_training.sh",
        "freeze_training.sh",
    ] {
        files.push(here.join(file));
    }
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
