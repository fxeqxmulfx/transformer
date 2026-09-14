//! Compile a C or WebAssembly program to the machine's token prefix.
//!
//! The counterpart of `python -m transformer_vm.compilation.compile_wasm`:
//! writes `<name>.txt` — the dispatch table between braces, then the input
//! tokens — and `<name>_spec.txt` for the specialized model.
//!
//! ```text
//! alm-cc examples/hello.c --args World -o data/hello
//! alm-cc --all --examples examples --out data
//! ```

use std::path::{Path, PathBuf};

fn usage() -> ! {
    eprintln!(
        "usage: alm-cc <program.c|program.wasm> [--args STR] [-o BASE]\n\
         \x20      alm-cc --all [--examples DIR] [--out DIR]"
    );
    std::process::exit(2)
}

/// Compile one file and write the two outputs, returning the instruction count.
fn one(input: &Path, args_str: &str, out_base: &Path, runtime_h: &Path) -> Result<usize, String> {
    let (wasm_path, temporary) = if input.extension().is_some_and(|e| e == "c") {
        (alm_compile::emit::compile_c_to_wasm(input, runtime_h)?, true)
    } else {
        (input.to_path_buf(), false)
    };

    let bytes = std::fs::read(&wasm_path).map_err(|e| format!("{}: {e}", wasm_path.display()))?;
    let (txt, spec, input_base) = alm_compile::emit::compile_program(&bytes, args_str)?;

    if let Some(dir) = out_base.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
    }
    let txt_path = out_base.with_extension("txt");
    let spec_path = PathBuf::from(format!("{}_spec.txt", out_base.display()));
    std::fs::write(&txt_path, &txt).map_err(|e| format!("{}: {e}", txt_path.display()))?;
    std::fs::write(&spec_path, &spec).map_err(|e| format!("{}: {e}", spec_path.display()))?;

    // The `.wasm` is an intermediate when the input was a `.c`, as in the
    // Python; leaving it behind would make the next run read a stale one.
    if temporary {
        let _ = std::fs::remove_file(&wasm_path);
    }

    // The instruction count is the dispatch table's, so it stops at the brace
    // that closes it; the input tokens that follow are not instructions.
    let _ = input_base;
    Ok(txt.lines().take_while(|l| *l != "}").filter(|l| *l != "{").count())
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut input: Option<String> = None;
    let mut args_str = String::new();
    let mut out_base: Option<String> = None;
    let mut all = false;
    let mut examples = "transformer-vm/transformer_vm/examples".to_string();
    let mut out_dir = "transformer-vm/transformer_vm/data".to_string();
    let mut runtime_h: Option<String> = None;

    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        let value = |i: &mut usize| -> String {
            *i += 1;
            argv.get(*i).cloned().unwrap_or_else(|| usage())
        };
        match a.as_str() {
            "--all" => all = true,
            "--args" => args_str = value(&mut i),
            "-o" | "--output" => out_base = Some(value(&mut i)),
            "--examples" => examples = value(&mut i),
            "--out" => out_dir = value(&mut i),
            "--runtime" => runtime_h = Some(value(&mut i)),
            "-h" | "--help" => usage(),
            other if other.starts_with('-') => usage(),
            other => input = Some(other.to_string()),
        }
        i += 1;
    }

    let examples = PathBuf::from(&examples);
    let runtime = runtime_h.map(PathBuf::from).unwrap_or_else(|| {
        examples.parent().unwrap_or(Path::new(".")).join("compilation/runtime.h")
    });

    let result = if all {
        let manifest_path = examples.join("manifest.yaml");
        (|| -> Result<(), String> {
            let text = std::fs::read_to_string(&manifest_path)
                .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
            for (name, args) in alm_compile::emit::load_manifest(&text)? {
                let c_path = examples.join(format!("{name}.c"));
                if !c_path.exists() {
                    eprintln!("skipping {name}: no {}", c_path.display());
                    continue;
                }
                let base = PathBuf::from(&out_dir).join(&name);
                let n = one(&c_path, &args, &base, &runtime)?;
                println!("{}.txt: {n} instructions", base.display());
            }
            Ok(())
        })()
    } else {
        let Some(input) = input else { usage() };
        let input = PathBuf::from(input);
        let base = out_base.map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from(&out_dir).join(input.file_stem().unwrap_or_default())
        });
        one(&input, &args_str, &base, &runtime).map(|n| {
            println!("{}.txt: {n} instructions", base.display());
        })
    };

    if let Err(e) = result {
        eprintln!("alm-cc: {e}");
        std::process::exit(1);
    }
}
