//! Run the real model against the real reference traces.
//!
//! Everything the driver is run on is built here, out of this repository: the
//! weights from the compiled-in `plan.yaml`, the programs from the C sources
//! in `programs/`.  Each artefact is checked against its SHA-256 in
//! `reference/sha256sums` before the driver sees it, so this is a run on the
//! released `model.bin` and the released traces under another name — and the
//! released files themselves, ten megabytes of them, stay out of the tree.
//!
//! What can be missing is a clang that targets wasm32.  Then the test says so
//! and passes; there is no point in failing a checkout for want of a compiler.

use alm_compile::release::{released_digest, sha256};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// The programs cheap enough to run on every `cargo test`.  `collatz` is
/// 44 589 tokens and `sudoku` is millions; those are for the driver, not here.
const PROGRAMS: [&str; 2] = ["hello", "addition"];

/// Refuse to hand the driver anything that is not the released artefact.
fn released(name: &str, bytes: &[u8]) {
    let want = released_digest(name).unwrap_or_else(|| panic!("{name} is not in the manifest"));
    assert_eq!(sha256(bytes), want, "{name} differs from the released one");
}

/// Build `model.bin` and the two traces.  The one thing that can go wrong
/// outside our control is the C compiler, so that is the only `Err`;
/// everything else is a failure of the port and asserts.
fn build_fixture() -> Result<PathBuf, String> {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("reference");
    std::fs::create_dir_all(&dir).expect("a directory under target/");

    let mg = alm_compile::interpreter::build();
    let plan = alm_compile::plan::Plan::load(alm_compile::release::PLAN, &mg.graph)
        .expect("the released plan resolves");
    let model = alm_compile::weights::build(&mg, &plan, Default::default()).to_bytes();
    released("model.bin", &model);
    std::fs::write(dir.join("model.bin"), &model).expect("the model writes");

    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../programs");
    let manifest = std::fs::read_to_string(src.join("manifest.yaml")).expect("the manifest is ours");
    let args = alm_compile::emit::load_manifest(&manifest).expect("the manifest reads");
    for name in PROGRAMS {
        let (_, input) = args.iter().find(|(n, _)| n == name).expect("the manifest has it");
        // clang removes the `.wasm` beside its input, so compile a copy.
        let copied = dir.join(format!("{name}.c"));
        std::fs::copy(src.join(format!("{name}.c")), &copied).expect("the source copies");
        let wasm = alm_compile::emit::compile_c_to_wasm(&copied, &src.join("runtime.h"))?;
        let wasm = std::fs::read(wasm).expect("clang wrote the module");
        let (txt, _, _) =
            alm_compile::emit::compile_program(&wasm, input).expect("the module compiles");
        let (trace, _) =
            alm_compile::reference::generate_ref(&txt, 100_000_000).expect("the program runs");
        for (file, text) in [(format!("{name}.txt"), txt), (format!("{name}_ref.txt"), trace)] {
            released(&format!("data/{file}"), text.as_bytes());
            std::fs::write(dir.join(&file), text).expect("the trace writes");
        }
    }
    Ok(dir)
}

/// The fixture, built once for the whole test binary.
fn fixture() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    match DIR.get_or_init(build_fixture) {
        Ok(dir) => Some(dir),
        Err(e) => {
            eprintln!("skipped: {e}");
            None
        }
    }
}

fn run(extra: &[&str]) -> Option<String> {
    let dir = fixture()?;
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_alm-vm"));
    cmd.arg(dir.join("model.bin")).args(extra);
    for p in PROGRAMS {
        cmd.arg(dir.join(format!("{p}.txt")));
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
fn the_integer_cache_reproduces_them_and_clears_the_grid() {
    // `todo3.md` section 4.  The integer head compares the keys rather than
    // the ordinates they were rounded into, which moves the wall from `2^26.5`
    // to `2^52`; the two things to check on the released weights are that it
    // changes no answer, and that it actually reaches the heads section 4b is
    // about -- the ones keyed on 32-bit values, which are past the old wall
    // from their first token and are also the heads that clear.
    let (Some(shipped), Some(lift)) = (run(&["--grid"]), run(&["--lift"])) else { return };
    assert!(lift.contains("2 passed, 0 failed"), "{lift}");
    assert!(lift.contains("Hello World!") && lift.contains("19134"), "{lift}");

    let lines = |s: &str, mark: &str| -> Vec<String> {
        s.lines().filter(|l| l.contains(mark)).map(|l| l.trim().to_string()).collect()
    };
    // And the crossings, which are the point of the exercise.  Both paths
    // weigh every answer they take from a stored ordinate, so the two counts
    // are comparable: what the integer path removes is the queries it answers
    // in `i128` instead, and what it leaves is the queries it cannot -- the
    // ones off the unit grid, which fall back to the stored points and so to
    // the old wall.  `hello` loses all of its crossings and `addition` keeps
    // the 8 %% that the query scale left one ulp off an integer.
    let crossings = |s: &str| -> Vec<usize> {
        lines(s, "OFF THE GRID")
            .iter()
            .map(|l| l.split(": ").nth(1).and_then(|r| r.split(' ').next()).expect("a count").parse().expect("a number"))
            .collect()
    };
    assert_eq!(crossings(&shipped), vec![26, 788], "the hull path answers these blind\n{shipped}");
    assert_eq!(crossings(&lift), vec![64], "one program left, and only its off-grid queries\n{lift}");

    // Both heads of section 4b are reached, on both programs, and neither
    // retires: the clear markers they are full of are held beside the
    // container rather than in it (`ALM.ClearKey.marked_sup'_eq_live`).
    let past = lines(&lift, "past the old wall");
    assert_eq!(past.len(), 2, "one line per program\n{lift}");
    for l in &past {
        assert!(l.contains("2 kept the integer path and 0 retired"), "{l}");
    }
    for l in lines(&lift, "settled by the marker") {
        let after = l.split(", ").nth(1).expect("the second clause");
        let settled: usize = after.split(' ').next().expect("its count").parse().expect("a number");
        assert!(settled > 0, "{l}");
    }
}

#[test]
fn the_query_scale_inflates_the_non_integer_share() {
    // `todo3.md` section 8a.  The hypothesis `q : Z` of
    // `ALM.FloatGrid.fp_exact_of_grid` is about the query, and the machine
    // forms queries by multiplying an averaged sum back by its count, which
    // does not round-trip.  Counting the failures on the *released* weights
    // overstates them by a factor of five: there `|qy|` is the query scale
    // `s = sqrt(2) * 1e10`, the weight rows were rounded after being
    // multiplied by `s`, and reconstructing the number divides by `s` again.
    // What this test pins is that inflated figure, because `model.bin` is the
    // only artefact it has; the honest 2.8 % to 7.1 % needs weights rebuilt
    // with `patches/on-the-grid.patch` and is recorded in section 8a.
    let Some(text) = run(&[]) else { return };
    let lines: Vec<&str> = text.lines().filter(|l| l.trim_start().starts_with("queries:")).collect();
    assert_eq!(lines.len(), 2, "one per program\n{text}");

    for line in lines {
        let counts = line.split_whitespace().nth(1).expect("off/total");
        let (off, total) = counts.split_once('/').expect("off/total");
        let (off, total): (f64, f64) = (off.parse().unwrap(), total.parse().unwrap());
        let share = off / total;
        assert!((0.20..0.35).contains(&share), "{share} of {total} in {line}");

        // And the offset that goes with it: one ulp of a query of order 1e10,
        // which is `1.9e-6` — an absolute margin six orders inside the
        // half-unit `ALM.FloatHull.cmp_of_sep` asks for, and an artefact of
        // the same scale.  Unscaled, the queries are of order 1e3 and the
        // worst offset is two ulp.
        let worst: f64 = line.split("worst offset ").nth(1).unwrap().split(' ').next().unwrap().parse().unwrap();
        assert!(worst < 1e-5, "worst query offset {worst} in {line}");
        assert!(line.contains("(1.0 ulp,"), "at the shipped scale it is one ulp: {line}");
    }
}

#[test]
fn nothing_on_the_released_weights_is_decided_by_rounding() {
    // `todo3.md` section 2a.  The compiler adds `LATEST_ALPHA * inv_log_pos(p)`
    // to every key so that two writes to one logical key are separated before
    // the comparison; the cache's `last_seq` is a fallback for the case where
    // they arrive equal anyway.  If that separation ever fell to the size of
    // the key path's own rounding, the answer would be whichever way the
    // matvec rounded.  On these programs it does not come close: the closest
    // runner-up is `6.0e-6` key steps in `hello` and `9.8e-7` in `addition`,
    // eight orders above the `4.3e-15` that rebuilding without the
    // perturbation produces.  The two figures are also the trend section 2a
    // leaves open: the separation is `0.3 / (p log^2 p)` and it shrinks with
    // the trace while the rounding does not.
    let Some(text) = run(&["--brute"]) else { return };
    let lines: Vec<&str> = text.lines().filter(|l| l.trim_start().starts_with("gaps:")).collect();
    assert_eq!(lines.len(), 2, "one per program\n{text}");

    for line in lines {
        let noise: usize = line.split(", ").nth(1).unwrap().split(' ').next().unwrap().parse().unwrap();
        assert_eq!(noise, 0, "a gap decided by rounding: {line}");

        let worst: f64 =
            line.split("runner-up ").nth(1).unwrap().split(' ').next().unwrap().parse().unwrap();
        assert!(worst > 1e-7, "the perturbation is still visible: {line}");
        assert!(worst < 0.5, "and inside the unit gap between distinct keys: {line}");
    }
}
