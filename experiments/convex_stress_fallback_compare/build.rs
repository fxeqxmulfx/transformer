use std::{
    env, fs,
    path::{Path, PathBuf},
};
fn collect(dir: &Path, files: &mut Vec<PathBuf>) {
    println!("cargo:rerun-if-changed={}", dir.display());
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect(&path, files);
        } else if path.extension().is_some_and(|s| s == "rs") {
            files.push(path);
        }
    }
}
fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut files = vec![];
    collect(&root.join("src"), &mut files);
    for f in [
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "protocol.json",
        "run_original.sh",
        "freeze_original.sh",
        "freeze_source.sh",
    ] {
        files.push(root.join(f));
    }
    let core = root.join("../convex_stress_fallback");
    collect(&core.join("src"), &mut files);
    for f in [
        "Cargo.toml",
        "Cargo.lock",
        "protocol.json",
        "freeze_source.sh",
    ] {
        files.push(core.join(f));
    }
    files.sort();
    let mut text = String::from("pub const SOURCES: &[(&str,&[u8])] = &[\n");
    for file in files {
        println!("cargo:rerun-if-changed={}", file.display());
        text.push_str(&format!(
            "({:?},include_bytes!({:?})),\n",
            file.strip_prefix(&root).unwrap().to_str().unwrap(),
            file.to_str().unwrap()
        ));
    }
    text.push_str("];\n");
    fs::write(
        PathBuf::from(env::var("OUT_DIR").unwrap()).join("sources.rs"),
        text,
    )
    .unwrap();
}
