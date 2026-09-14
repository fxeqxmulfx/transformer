//! Run the real model against the real reference traces.
//!
//! The artefacts these need are produced by the original Python and are not in
//! this repository: `model.bin` comes from `python -m transformer_vm.build
//! --save-weights=model.bin` and the programs from `compile_wasm.ensure_data()`.
//! When they are absent the test says so and passes — there is no point in
//! failing a checkout for want of a 1.2 MB artefact — but when they are present
//! it is the only test here that checks the whole stack at once.

use std::path::PathBuf;
use std::process::Command;

fn vendored() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../transformer-vm")
}

/// The programs cheap enough to run on every `cargo test`.  `collatz` is
/// 44 589 tokens and `sudoku` is millions; those are for the driver, not here.
const PROGRAMS: [&str; 2] = ["hello", "addition"];

fn run(extra: &[&str]) -> Option<String> {
    let root = vendored();
    let model = root.join("model.bin");
    let data = root.join("transformer_vm/data");
    if !model.exists() {
        eprintln!("skipped: no {} — build it with the original Python", model.display());
        return None;
    }

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_alm-vm"));
    cmd.arg(&model).args(extra);
    for p in PROGRAMS {
        let prog = data.join(format!("{p}.txt"));
        if !prog.exists() {
            eprintln!("skipped: no {}", prog.display());
            return None;
        }
        cmd.arg(prog);
    }

    let out = cmd.output().expect("the driver runs");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(out.status.success(), "the driver reported a failure:\n{text}");
    Some(text)
}

#[test]
fn the_hull_cache_reproduces_the_reference_traces() {
    let Some(text) = run(&[]) else { return };
    assert!(text.contains("2 passed, 0 failed"), "{text}");
    assert!(text.contains("Hello World!"), "{text}");
    assert!(text.contains("19134"), "the addition program's output\n{text}");
}

#[test]
fn the_brute_cache_reproduces_them_too() {
    let Some(text) = run(&["--brute"]) else { return };
    assert!(text.contains("2 passed, 0 failed"), "{text}");
}

#[test]
fn the_two_caches_agree_token_for_token() {
    let (Some(hull), Some(brute)) = (run(&[]), run(&["--brute"])) else { return };
    // Both reproduce the same reference, so the token counts must match; the
    // timings will not, which is why the comparison is on the counts.
    let counts = |s: &str| -> Vec<String> {
        s.lines()
            .filter_map(|l| l.split_once("PASS  ").map(|(_, r)| r.split(" in ").next().unwrap_or("").to_string()))
            .collect()
    };
    assert_eq!(counts(&hull), counts(&brute));
    assert!(!counts(&hull).is_empty());
}
