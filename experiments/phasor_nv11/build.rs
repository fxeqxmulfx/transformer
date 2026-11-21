//! Compile the same CPU forward and weight layout as the selected DiffGPT model.
use std::{env, fs, path::PathBuf, process::Command};

const REVISION: &str = "5a37c515639f23037d191b0659a5a0853984328d";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PHASOR_GPT_REPO");
    let repo = env::var_os("PHASOR_GPT_REPO")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../../.lake/reproduction/gpt"));
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut modules = String::new();
    for (name, source) in [
        (
            "oracle",
            "research/main_model/phase_score_native/src/oracle.rs",
        ),
        (
            "layout",
            "research/main_model/phase_score_native/src/layout.rs",
        ),
        (
            "random",
            "research/complex_geometry_suite/native/src/random.rs",
        ),
    ] {
        let file = Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(["show", &format!("{REVISION}:{source}")])
            .output()
            .expect("read the pinned DiffGPT sources with git");
        assert!(
            file.status.success(),
            "Set PHASOR_GPT_REPO to a checkout containing {REVISION}. {}",
            String::from_utf8_lossy(&file.stderr)
        );
        let path = out.join(format!("{name}.rs"));
        fs::write(&path, file.stdout).unwrap();
        modules.push_str(&format!(
            "#[allow(dead_code, clippy::all)]\n#[path = {path:?}]\nmod {name};\n"
        ));
    }
    fs::write(out.join("upstream.rs"), modules).unwrap();
}
