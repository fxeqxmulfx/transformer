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

#[test]
fn unit_scale_queries_reproduce_them_as_well() {
    // `todo3.md` section 0 asks for the hard-attention query scale to go.  It
    // is an argmax, so dividing the query by `|qy|` cannot change the answer —
    // and on the released weights it does not, to the token.  What it does not
    // do is remove a single off-the-grid query: those come from the 32-bit
    // values the heads are keyed on (section 4b), which no rescaling touches.
    let (Some(shipped), Some(grid)) = (run(&[]), run(&["--grid"])) else { return };
    assert!(grid.contains("2 passed, 0 failed"), "{grid}");
    assert!(grid.contains("Hello World!") && grid.contains("19134"), "{grid}");

    let crossings = |s: &str| -> Vec<String> {
        s.lines().filter(|l| l.contains("OFF THE GRID")).map(|l| l.trim().to_string()).collect()
    };
    assert_eq!(crossings(&shipped), crossings(&grid), "the scale is not what puts them off the grid");
    assert!(!crossings(&grid).is_empty(), "and they are off it");
}

#[test]
fn a_quarter_of_the_real_queries_are_not_integers() {
    // `todo3.md` section 8, measured rather than derived.  The hypothesis
    // `q : Z` of `ALM.FloatGrid.fp_exact_of_grid` is about the query, and the
    // machine forms queries by multiplying an averaged sum back by its count;
    // section 8 predicts that round trip fails 25.8 % of the time from the
    // arithmetic alone.  On the released weights it is 24-26 %.
    let Some(text) = run(&[]) else { return };
    let lines: Vec<&str> = text.lines().filter(|l| l.trim_start().starts_with("queries:")).collect();
    assert_eq!(lines.len(), 2, "one per program\n{text}");

    for line in lines {
        let counts = line.split_whitespace().nth(1).expect("off/total");
        let (off, total) = counts.split_once('/').expect("off/total");
        let (off, total): (f64, f64) = (off.parse().unwrap(), total.parse().unwrap());
        let share = off / total;
        assert!((0.20..0.30).contains(&share), "{share} of {total} in {line}");

        // And what saves it: the offset is one ulp of a query of order 1e10,
        // which is six orders of magnitude inside the half-unit that
        // `ALM.FloatHull.cmp_of_sep` asks for.
        let worst: f64 = line.split("worst offset ").nth(1).unwrap().split(' ').next().unwrap().parse().unwrap();
        assert!(worst < 1e-5, "worst query offset {worst} in {line}");
        assert!(line.contains("(1.0 ulp,"), "and it is exactly one ulp: {line}");
    }
}
