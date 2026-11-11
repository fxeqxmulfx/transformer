//! Hash source bytes embedded at compilation, rather than mutable runtime files.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, process::Command};

macro_rules! source {
    ($name:literal) => {
        (
            $name,
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/", $name)).as_slice(),
        )
    };
}

pub fn source_hash() -> String {
    let mut hash = Sha256::new();
    for (name, bytes) in [
        source!("Cargo.toml"),
        source!("Cargo.lock"),
        source!("src/lib.rs"),
        source!("src/main.rs"),
        source!("src/model.rs"),
        source!("src/tests.rs"),
        source!("src/bench.rs"),
        source!("src/checks.rs"),
        source!("src/provenance.rs"),
        source!("../convex_channels/Cargo.toml"),
        source!("../convex_channels/src/lib.rs"),
        source!("../convex_channels/src/channel.rs"),
        source!("../convex_channels/src/task.rs"),
        source!("../convex_recall_compare/Cargo.toml"),
        source!("../convex_recall_compare/src/lib.rs"),
        source!("../convex_recall_compare/src/data.rs"),
        source!("../convex_recall_compare/src/convex.rs"),
        source!("../grok_division/src/mt.rs"),
    ] {
        hash.update(name.as_bytes());
        hash.update(bytes);
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn output(program: &str, args: &[&str]) -> String {
    let result = Command::new(program).args(args).output().unwrap();
    assert!(result.status.success());
    String::from_utf8(result.stdout).unwrap().trim().to_owned()
}

pub fn environment() -> Value {
    let cpu = fs::read_to_string("/proc/cpuinfo").unwrap();
    let model = cpu
        .lines()
        .find(|line| line.starts_with("model name"))
        .unwrap();
    let status = fs::read_to_string("/proc/self/status").unwrap();
    let affinity = status
        .lines()
        .find(|line| line.starts_with("Cpus_allowed_list:"))
        .unwrap();
    let cpu_id = affinity.split_once(':').unwrap().1.trim();
    assert!(
        cpu_id.parse::<usize>().is_ok(),
        "pin the benchmark to one CPU with taskset"
    );
    json!({"cpu":model,"affinity":cpu_id,"rustc":output("rustc",&["--version"]),
        "commit":output("git",&["rev-parse","HEAD"]),
        "git_status":output("git",&["status","--short"]),
        "load_average":fs::read_to_string("/proc/loadavg").unwrap().trim(),
        "unix_seconds":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()})
}
