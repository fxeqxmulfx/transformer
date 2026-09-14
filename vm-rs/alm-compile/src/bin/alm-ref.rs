//! Generate a program's reference token trace.
//!
//! The counterpart of `python -m transformer_vm.wasm.reference`: runs the
//! dispatch table in `<name>.txt` and writes the tokens the model is expected
//! to produce to `<name>_ref.txt`.
//!
//! ```text
//! alm-ref data/hello.txt            # writes data/hello_ref.txt
//! alm-ref --all --data data         # every program that has none yet
//! alm-ref --all --regen --data data
//! ```

use std::path::{Path, PathBuf};

fn usage() -> ! {
    eprintln!("usage: alm-ref <program.txt>...\n       alm-ref --all [--regen] [--data DIR]");
    std::process::exit(2)
}

fn ref_path(prog: &Path) -> PathBuf {
    let stem = prog.file_stem().unwrap_or_default().to_string_lossy().into_owned();
    prog.with_file_name(format!("{stem}_ref.txt"))
}

fn one(prog: &Path, max_tokens: u64) -> Result<(), String> {
    let text = std::fs::read_to_string(prog).map_err(|e| format!("{}: {e}", prog.display()))?;
    let (formatted, run) = alm_compile::reference::generate_ref(&text, max_tokens)?;
    let out = ref_path(prog);
    std::fs::write(&out, &formatted).map_err(|e| format!("{}: {e}", out.display()))?;
    println!("{}: {} tokens, output={:?}", out.display(), run.token_count, run.output);
    Ok(())
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut files: Vec<String> = Vec::new();
    let mut all = false;
    let mut regen = false;
    let mut data = "transformer-vm/transformer_vm/data".to_string();
    let mut max_tokens: u64 = 100_000_000;

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--all" => all = true,
            "--regen" => regen = true,
            "--data" => {
                i += 1;
                data = argv.get(i).cloned().unwrap_or_else(|| usage());
            }
            "--max" => {
                i += 1;
                max_tokens = argv
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            "-h" | "--help" => usage(),
            a if a.starts_with('-') => usage(),
            a => files.push(a.to_string()),
        }
        i += 1;
    }

    let result = (|| -> Result<(), String> {
        if all {
            let dir = PathBuf::from(&data);
            let mut progs: Vec<PathBuf> = std::fs::read_dir(&dir)
                .map_err(|e| format!("{}: {e}", dir.display()))?
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    let name = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
                    name.ends_with(".txt") && !name.ends_with("_ref.txt") && !name.ends_with("_spec.txt")
                })
                .collect();
            progs.sort();
            for prog in progs {
                if !regen && ref_path(&prog).exists() {
                    println!("{}: already there", ref_path(&prog).display());
                    continue;
                }
                one(&prog, max_tokens)?;
            }
            return Ok(());
        }
        if files.is_empty() {
            usage();
        }
        for f in &files {
            one(Path::new(f), max_tokens)?;
        }
        Ok(())
    })();

    if let Err(e) = result {
        eprintln!("alm-ref: {e}");
        std::process::exit(1);
    }
}
