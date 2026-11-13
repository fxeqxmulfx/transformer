//! Embed local component/adapter inputs; the quotient dependency binds all
//! inherited numerical sources, original generators and unchanged metrics.
use std::{
    env, fs,
    path::{Path, PathBuf},
};
fn collect(path: &Path, files: &mut Vec<PathBuf>) {
    println!("cargo:rerun-if-changed={}", path.display());
    for entry in fs::read_dir(path).expect("source directory") {
        let p = entry.expect("source entry").path();
        if p.is_dir() {
            collect(&p, files);
        } else if p.extension().is_some_and(|e| e == "rs") {
            files.push(p);
        }
    }
}
fn main() {
    let here = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let component = here.parent().unwrap();
    let experiments = component.parent().unwrap();
    let mut files = vec![];
    for root in [component, here.as_path()] {
        collect(&root.join("src"), &mut files);
        for file in ["Cargo.toml", "Cargo.lock"] {
            files.push(root.join(file));
        }
    }
    for file in [
        "build.rs",
        "protocol.json",
        "run_original.sh",
        "freeze_original.sh",
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
