//! Pin the original trainer; generate an instrumented copy only in Cargo OUT_DIR.
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf, process::Command};

const SOURCE: &str = "14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da";
const FILES: &[&str] = &[
    "src/main.rs",
    "src/application.rs",
    "src/interfaces.rs",
    "src/domain/mod.rs",
    "src/domain/data.rs",
    "src/domain/model.rs",
    "src/infrastructure/mod.rs",
    "src/infrastructure/engine.rs",
    "src/infrastructure/gemm.rs",
    "src/infrastructure/ops.rs",
    "src/infrastructure/product.rs",
    "src/infrastructure/rotate.rs",
    "src/infrastructure/storage.rs",
    "Cargo.toml",
    "Cargo.lock",
];

fn main() {
    // Current selected GPT main model, 6bb9d03a; the kernel is copied unchanged.
    println!("cargo:rerun-if-changed=src/optimizer_kernels.rs");
    let kernel = fs::read("src/optimizer_kernels.rs").unwrap();
    let digest: String = Sha256::digest(&kernel)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        digest,
        "a5acfb922036ca76a555793d83a225a045ff9911e61ee927b96a19e6e2143258"
    );
    println!("cargo:rerun-if-env-changed=ROTATE90_SOURCE");
    let root = PathBuf::from(env::var("ROTATE90_SOURCE").unwrap_or_else(|_| {
        "/home/misha/rust_projects/grokking/experiments/rotate90_complex".into()
    }));
    let dest = PathBuf::from(env::var("OUT_DIR").unwrap()).join("original");
    let mut hasher = Sha256::new();
    for file in FILES {
        println!("cargo:rerun-if-changed={}", root.join(file).display());
        let bytes = fs::read(root.join(file)).expect("pinned original source missing");
        hasher.update(&bytes);
        let to = dest.join(file);
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::write(to, bytes).unwrap();
    }
    let digest: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(digest, SOURCE, "trainer source changed");
    let kernite = "/home/misha/rust_projects/kgen";
    let revision = Command::new("git")
        .args(["-C", kernite, "rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert!(revision.status.success());
    assert_eq!(
        String::from_utf8(revision.stdout).unwrap().trim(),
        "a1b6bc2c956e1a5f35960e26b209e1b6c213096c",
        "Kernite revision changed"
    );
    assert!(
        Command::new("git")
            .args(["-C", kernite, "diff", "--quiet", "HEAD", "--", "crates"])
            .status()
            .unwrap()
            .success(),
        "Kernite sources modified"
    );
    let extension = "src/instrumentation.rs";
    println!("cargo:rerun-if-changed={extension}");
    let engine = dest.join("src/infrastructure/engine.rs");
    let mut bytes = fs::read(&engine).unwrap();
    bytes.push(b'\n');
    bytes.extend(fs::read(extension).unwrap());
    fs::write(engine, bytes).unwrap();
    let oracle = "../flat_approx/fixtures/reference.rs";
    println!("cargo:rerun-if-changed={oracle}");
    fs::copy(oracle, dest.join("src/infrastructure/reference.rs")).unwrap();
    let modules: String = ["application", "domain", "infrastructure"]
        .iter()
        .map(|name| {
            let path = dest.join(if *name == "application" {
                "src/application.rs".into()
            } else {
                format!("src/{name}/mod.rs")
            });
            let compatibility = if *name == "domain" {
                "#[expect(clippy::should_implement_trait, reason=\"Pinned SplitMix64 next returns u64; preserve the source API\")]\n"
            } else { "" };
            format!("{compatibility}#[path={:?}] pub mod {name};\n", path.to_str().unwrap())
        })
        .collect();
    fs::write(dest.parent().unwrap().join("modules.rs"), modules).unwrap();
    println!("cargo:rustc-env=TRAINER_SOURCE_SHA256={SOURCE}");
}
