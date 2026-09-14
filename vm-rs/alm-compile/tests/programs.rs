//! The token prefix, against the released one.
//!
//! The six manifest programs are compiled here from their C sources in
//! `programs/`, run, and the result compared with the released
//! `transformer_vm/data/*.txt`, `*_spec.txt` and `*_ref.txt`.  That is the
//! whole of the program path — decode, lower, flatten, format, execute —
//! checked at once against the only artefacts that can settle it.
//!
//! The sources are ours; the released outputs are ten megabytes of build
//! product and are not in this repository, so the comparison is against their
//! SHA-256 in `reference/sha256sums`.  What can still be missing is a clang
//! that targets wasm32, and then the test says so and passes.

use alm_compile::release::{released_digest, sha256};
use std::path::{Path, PathBuf};

fn programs() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../programs")
}

/// Compile a C source into a directory of our own, so that removing the
/// intermediate `.wasm` — which is what the Python does — cannot disturb the
/// sources.
fn build(name: &str, scratch: &Path) -> Option<Vec<u8>> {
    let root = programs();
    let copied = scratch.join(format!("{name}.c"));
    std::fs::copy(root.join(format!("{name}.c")), &copied).expect("the source copies");
    match alm_compile::emit::compile_c_to_wasm(&copied, &root.join("runtime.h")) {
        Ok(wasm) => Some(std::fs::read(wasm).expect("clang wrote the module")),
        Err(e) => {
            eprintln!("skipped: {e}");
            None
        }
    }
}

#[test]
fn the_six_manifest_programs_compile_and_run_to_the_released_files() {
    let manifest =
        std::fs::read_to_string(programs().join("manifest.yaml")).expect("the manifest is ours");
    let programs = alm_compile::emit::load_manifest(&manifest).expect("the manifest reads");
    assert_eq!(programs.len(), 6, "the release ships six programs");

    let scratch = std::env::temp_dir().join(format!("alm-cc-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).expect("a scratch directory");

    let mut checked = 0;
    for (name, args) in &programs {
        let Some(wasm) = build(name, &scratch) else { continue };
        let (txt, spec, input_base) =
            alm_compile::emit::compile_program(&wasm, args).expect("the module compiles");
        assert_ne!(input_base, 0, "{name} takes its input from memory");

        // The trace is the program's own reference run: the tokens the model
        // is expected to produce, down to the carry marks.
        let (reference, run) =
            alm_compile::reference::generate_ref(&txt, 100_000_000).expect("the program runs");
        assert!(run.token_count > 0, "{name} produced no tokens");

        for (suffix, ours) in [(".txt", &txt), ("_spec.txt", &spec), ("_ref.txt", &reference)] {
            let file = format!("data/{name}{suffix}");
            let want = released_digest(&file).expect("the manifest covers the six programs");
            assert_eq!(sha256(ours.as_bytes()), want, "{file} differs from the released one");
            checked += 1;
        }
    }
    let _ = std::fs::remove_dir_all(&scratch);

    if checked == 0 {
        eprintln!("skipped: no clang that targets wasm32");
    } else {
        assert_eq!(checked, 18, "six programs, three files each");
        eprintln!("{checked} released files reproduced byte for byte");
    }
}
