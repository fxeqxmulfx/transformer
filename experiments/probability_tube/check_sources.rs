//! Read-only recovery preflight for the exact inputs pinned by
//! experiments/class_geometry_wsd/build.rs and src/main.rs at d1066cfd.
//! Passing this does not recreate a run or validate a training trajectory.
use std::{
    error::Error,
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const TRAINER: &str = "14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da";
const KERNITE: &str = "a1b6bc2c956e1a5f35960e26b209e1b6c213096c";
const FILES: [&str; 14] = [
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
];
fn digest(bytes: &[u8]) -> Result<String> {
    let mut child = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or("sha256sum stdin unavailable")?
        .write_all(bytes)?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err("sha256sum failed".into());
    }
    let value = String::from_utf8(output.stdout)?;
    let value = value.split_whitespace().next().ok_or("missing digest")?;
    if value.len() != 64 || !value.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid digest".into());
    }
    Ok(value.to_owned())
}
fn equal(label: &str, got: &str, expected: &str) -> Result<()> {
    if got != expected {
        return Err(format!("{label}: expected {expected}, got {got}").into());
    }
    println!("{label}: verified {got}");
    Ok(())
}
fn trainer(root: &Path) -> Result<()> {
    let mut bytes = Vec::new();
    // Cargo.lock is last, matching the original build script's exact ordering.
    for file in FILES.into_iter().chain(["Cargo.lock"]) {
        let path = root.join(file);
        bytes.extend(fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?);
    }
    equal("Rotate90 trainer", &digest(&bytes)?, TRAINER)
}
fn kernite(root: &Path) -> Result<()> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "HEAD"])
        .output()?;
    if !output.status.success() {
        return Err("Kernite revision lookup failed".into());
    }
    equal("Kernite", String::from_utf8(output.stdout)?.trim(), KERNITE)?;
    let status = Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "Cargo.toml",
            "Cargo.lock",
            "crates",
        ])
        .status()?;
    if !status.success() {
        return Err("Kernite workspace or crate sources are modified".into());
    }
    Ok(())
}
fn gpt(root: &Path) -> Result<()> {
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
        equal(file, &digest(&fs::read(root.join(file))?)?, expected)?;
    }
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 3 {
        return Err(
            "usage: check-rotate90-sources TRAINER_ROOT KERNITE_ROOT GPT_MAIN_MODEL_ROOT".into(),
        );
    }
    let mut failed = false;
    for result in [
        trainer(Path::new(&args[0])),
        kernite(Path::new(&args[1])),
        gpt(Path::new(&args[2])),
    ] {
        if let Err(e) = result {
            eprintln!("unavailable or mismatched source: {e}");
            failed = true;
        }
    }
    if failed {
        return Err("recreation prerequisite failed; no training was run".into());
    }
    println!(
        "All pinned source inputs verified; training and checkpoint evidence are still required."
    );
    Ok(())
}
