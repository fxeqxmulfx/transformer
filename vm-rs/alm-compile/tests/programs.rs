//! The token prefix, against the released one.
//!
//! The six manifest programs are compiled here from their C sources and the
//! result compared with `transformer_vm/data/*.txt` byte for byte.  That is
//! the whole of the program path — decode, lower, flatten, format — checked
//! at once against the only artefact that can settle it.
//!
//! Both halves are build products of the original Python and are not in this
//! repository, so the test says what is missing and passes when the vendored
//! checkout, or a clang that targets wasm32, is not there.

use std::path::{Path, PathBuf};

fn vendored() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../transformer-vm/transformer_vm")
}

/// Compile a C source into a directory of our own, so that removing the
/// intermediate `.wasm` — which is what the Python does — cannot disturb the
/// vendored tree.
fn build(name: &str, scratch: &Path) -> Option<Vec<u8>> {
    let root = vendored();
    let c_path = root.join(format!("examples/{name}.c"));
    if !c_path.exists() {
        eprintln!("skipped: no {}", c_path.display());
        return None;
    }
    let copied = scratch.join(format!("{name}.c"));
    std::fs::copy(&c_path, &copied).expect("the source copies");
    match alm_compile::emit::compile_c_to_wasm(&copied, &root.join("compilation/runtime.h")) {
        Ok(wasm) => Some(std::fs::read(wasm).expect("clang wrote the module")),
        Err(e) => {
            eprintln!("skipped: {e}");
            None
        }
    }
}

#[test]
fn the_six_manifest_programs_compile_to_the_released_token_prefix() {
    let root = vendored();
    let Ok(manifest) = std::fs::read_to_string(root.join("examples/manifest.yaml")) else {
        eprintln!("skipped: no vendored examples");
        return;
    };
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

        for (suffix, ours) in [(".txt", &txt), ("_spec.txt", &spec)] {
            let path = root.join(format!("data/{name}{suffix}"));
            let Ok(theirs) = std::fs::read_to_string(&path) else {
                eprintln!("skipped: no {}", path.display());
                continue;
            };
            assert_eq!(*ours, theirs, "{name}{suffix} differs from the released one");
            checked += 1;
        }
    }
    let _ = std::fs::remove_dir_all(&scratch);

    if checked == 0 {
        eprintln!("skipped: nothing to compare against");
    } else {
        eprintln!("{checked} released files reproduced byte for byte");
    }
}
