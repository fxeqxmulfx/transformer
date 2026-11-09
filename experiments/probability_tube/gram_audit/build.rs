//! Source-checked import of b70811e. Only the output copy gets read-only probes.
//! Port of sharp_audit/build.rs at d3893996 with the independent QG library ledger.
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf, process::Command};

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.join("../../..").canonicalize().unwrap();
    let trainer = root.join(".lake/reproduction/grokking/experiments/rotate90_complex");
    let kernite = root.join(".lake/reproduction/kgen");
    let gpt = root.join(".lake/reproduction/diff-gpt/research/main_model");
    let mut inputs = vec![];
    let dest = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("original");
    for file in [
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
    ] {
        let path = trainer.join(file);
        println!("cargo:rerun-if-changed={}", path.display());
        let bytes = fs::read(path).expect("pinned Rotate90 source unavailable");
        inputs.extend_from_slice(&bytes);
        let to = dest.join(file);
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::write(to, bytes).unwrap();
    }
    let source = hash(&inputs);
    assert_eq!(
        source,
        "14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da"
    );
    let rev = Command::new("git")
        .arg("-C")
        .arg(&kernite)
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert!(rev.status.success());
    assert_eq!(
        String::from_utf8(rev.stdout).unwrap().trim(),
        "a1b6bc2c956e1a5f35960e26b209e1b6c213096c"
    );
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(&kernite)
            .args([
                "diff",
                "--quiet",
                "HEAD",
                "--",
                "crates",
                "Cargo.toml",
                "Cargo.lock"
            ])
            .status()
            .unwrap()
            .success(),
        "modified Kernite sources"
    );
    for (file, expected) in [
        (
            "model.json",
            "0af933c3b6ec3e322714cd9654a6ab70b0d6bca193454a8b8dd165635d605e08",
        ),
        (
            "phase_score_native/src/optimizer.rs",
            "ceb430d1d5eff338e8826266f20c01870678e7655f0a99f7c62fbeb84b74974b",
        ),
        (
            "phase_score_native/src/optimizer_kernels.rs",
            "a5acfb922036ca76a555793d83a225a045ff9911e61ee927b96a19e6e2143258",
        ),
        (
            "wsd_suite_native/src/schedule.rs",
            "5853b9f7255428ef8fa7d11b7e98c329dc0129e0990c47430a77eeaef5c22bd4",
        ),
    ] {
        let path = gpt.join(file);
        println!("cargo:rerun-if-changed={}", path.display());
        assert_eq!(
            hash(&fs::read(path).unwrap()),
            expected,
            "changed GPT source: {file}"
        );
    }
    let kernel = root.join("experiments/class_geometry_wsd/src/optimizer_kernels.rs");
    println!("cargo:rerun-if-changed={}", kernel.display());
    assert_eq!(
        hash(&fs::read(kernel).unwrap()),
        "a5acfb922036ca76a555793d83a225a045ff9911e61ee927b96a19e6e2143258"
    );
    let instrument = root.join("experiments/single_update/src/instrumentation.rs");
    println!("cargo:rerun-if-changed={}", instrument.display());
    let engine = dest.join("src/infrastructure/engine.rs");
    let mut code = fs::read_to_string(&engine).unwrap();
    code.push_str(&format!("\ninclude!({instrument:?});\n"));
    code.push_str("impl Engine { pub fn wsd_moments(&self) -> (kernite_cuda::Ptr<f32>, kernite_cuda::Ptr<f32>) { (self.first.ptr(), self.second.ptr()) } }\n");
    fs::write(engine, code).unwrap();
    let oracle = root.join("experiments/flat_approx/fixtures/reference.rs");
    println!("cargo:rerun-if-changed={}", oracle.display());
    fs::copy(oracle, dest.join("src/infrastructure/reference.rs")).unwrap();
    let modules: String = ["application","domain","infrastructure"].iter().map(|name| {
        let path = dest.join(if *name == "application" {"src/application.rs".into()} else {format!("src/{name}/mod.rs")});
        let compat = if *name == "domain" {
            "#[expect(clippy::should_implement_trait, reason=\"Preserve pinned SplitMix64 source API\")]\n"
        } else {""};
        format!("{compat}#[path={path:?}] pub mod {name};\n")
    }).collect();
    fs::write(dest.parent().unwrap().join("modules.rs"), modules).unwrap();
    println!("cargo:rustc-env=TRAINER_SOURCE_SHA256={source}");
    let mut sources: Vec<_> = fs::read_dir(manifest.join("src"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "rs"))
        .collect();
    for file in [
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "PROTOCOL.md",
        "summary.jq",
        "../PROTOCOL.md",
        "../bounds.rs",
        "../adam.rs",
        "../encoder.rs",
        "../Cargo.toml",
        "../Cargo.lock",
        "../gram/lib.rs",
        "../gram/covariance.rs",
        "../gram/tests.rs",
        "../gram/Cargo.toml",
        "../gram/Cargo.lock",
    ] {
        sources.push(manifest.join(file));
    }
    for line in fs::read_to_string(manifest.join("src/main.rs"))
        .unwrap()
        .lines()
    {
        if let Some(path) = line
            .trim()
            .strip_prefix("#[path = \"")
            .and_then(|s| s.strip_suffix("\"]"))
        {
            sources.push(manifest.join("src").join(path));
        }
    }
    sources.push(instrument);
    let mut ledger = String::new();
    sources.sort();
    sources.dedup();
    for path in sources {
        let path = path.canonicalize().unwrap();
        println!("cargo:rerun-if-changed={}", path.display());
        ledger.push_str(&format!(
            "{}\t{}\n",
            path.strip_prefix(&root).unwrap().display(),
            hash(&fs::read(&path).unwrap())
        ));
    }
    fs::write(dest.parent().unwrap().join("sources.tsv"), &ledger).unwrap();
    println!(
        "cargo:rustc-env=REPRO_SOURCE_SHA256={}",
        hash(ledger.as_bytes())
    );
    let rustc = Command::new(env::var_os("RUSTC").unwrap())
        .arg("--version")
        .output()
        .unwrap();
    assert!(rustc.status.success());
    println!(
        "cargo:rustc-env=REPRO_RUSTC={}",
        String::from_utf8(rustc.stdout).unwrap().trim()
    );
}
