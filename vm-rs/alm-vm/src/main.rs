//! `alm-vm` — load a model, run WASM programs through it, check the output.
//!
//! The command line follows `model/transformer.cpp` so the two drivers can be
//! pointed at the same files:
//!
//! ```text
//! alm-vm model.bin [--brute] [--trace[=N]] [--args=STR] [--max=N] prog.txt ...
//! ```

mod program;
mod run;

use alm_hull::TieBreak;
use alm_model::{Alm, Backend, CacheKind, KvCache, RawModel};
use program::Program;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// What the C++ driver generates when there is no reference to bound it.
const MAX_GEN: usize = 6_000_000;

struct Options {
    model: PathBuf,
    programs: Vec<PathBuf>,
    kind: CacheKind,
    trace_every: usize,
    args: Option<String>,
    max_new: Option<usize>,
}

fn parse(argv: &[String]) -> Result<Options, String> {
    let mut model = None;
    let mut programs = Vec::new();
    let mut kind = CacheKind::Hull;
    let mut trace_every = 0;
    let mut args = None;
    let mut max_new = None;

    let mut it = argv.iter();
    while let Some(a) = it.next() {
        if a == "--brute" || a == "--nohull" {
            kind = CacheKind::Brute;
        } else if a == "--trace" {
            trace_every = 1;
        } else if let Some(n) = a.strip_prefix("--trace=") {
            trace_every = n.parse().map_err(|_| format!("--trace={n} is not a count"))?;
        } else if let Some(s) = a.strip_prefix("--args=") {
            args = Some(s.to_string());
        } else if a == "--args" {
            args = it.next().cloned();
        } else if let Some(n) = a.strip_prefix("--max=") {
            max_new = Some(n.parse().map_err(|_| format!("--max={n} is not a count"))?);
        } else if a.starts_with("--") {
            return Err(format!("unknown option {a}"));
        } else if model.is_none() {
            model = Some(PathBuf::from(a));
        } else if !a.contains("_ref") {
            // The C++ driver skips reference files silently, so that a glob
            // over a directory does the right thing.
            programs.push(PathBuf::from(a));
        }
    }

    let model = model.ok_or("usage: alm-vm model.bin [options] prog.txt ...")?;
    if programs.is_empty() {
        return Err("no programs given".into());
    }
    Ok(Options { model, programs, kind, trace_every, args, max_new })
}

/// A fresh cache per program, with the per-head tie-break the file records.
fn cache_for(raw: &RawModel, kind: CacheKind) -> KvCache {
    let mut cache = KvCache::new(raw.shapes.n_layers, raw.shapes.n_heads, kind);
    if let Some(latest) = &raw.latest_heads {
        for (l, row) in latest.iter().enumerate() {
            for (h, &is_latest) in row.iter().enumerate() {
                if is_latest {
                    cache.set_tie_break(l, h, TieBreak::Latest);
                }
            }
        }
    }
    cache
}

/// One program: run it, compare against its reference, report as the original
/// reports.  Returns `None` when there was no reference to compare against.
fn check(model: &Alm<Backend>, raw: &RawModel, opts: &Options, path: &Path) -> std::io::Result<Option<bool>> {
    let program = Program::load(raw, path, opts.args.as_deref())?;
    let max_new = opts.max_new.unwrap_or_else(|| match &program.reference {
        Some(r) => (r.len() + 100).saturating_sub(program.ids.len()).max(100),
        None => MAX_GEN,
    });

    print!("{}: ", program.name);
    use std::io::Write;
    std::io::stdout().flush()?;

    let mut cache = cache_for(raw, opts.kind);
    let result = run::generate(model, &mut cache, &program.ids, max_new, opts.trace_every);
    let (n, ops, dt) = (result.ids.len(), result.ops, result.seconds);
    let rate = if dt > 0.0 { n as f64 / dt } else { 0.0 };

    let verdict = match &program.reference {
        None => {
            println!("RAN   {n} tok, {ops} ops in {dt:.2}s ({rate:.0} tok/s)");
            None
        }
        Some(reference) => {
            match result.ids.iter().zip(reference).position(|(a, b)| a != b) {
                Some(i) => {
                    println!(
                        "  MISMATCH at {i}: predicted={}, expected={}",
                        model.tokens[result.ids[i]], model.tokens[reference[i]]
                    );
                    println!("FAIL ({dt:.2}s)");
                    Some(false)
                }
                None if n < reference.len() => {
                    let total = reference.len();
                    println!("PASS  {n}/{total} tok, {ops} ops in {dt:.2}s ({rate:.0} tok/s) [truncated]");
                    Some(true)
                }
                None if n > reference.len() => {
                    let total = reference.len();
                    println!("PASS  {n} tok (ref {total}), {ops} ops in {dt:.2}s ({rate:.0} tok/s) [ref truncated]");
                    Some(true)
                }
                None => {
                    println!("PASS  {n} tok, {ops} ops in {dt:.2}s ({rate:.0} tok/s)");
                    Some(true)
                }
            }
        }
    };

    if !result.stopped && opts.max_new.is_none() && program.reference.is_none() {
        println!("  (budget exhausted: the run did not reach the stop token)");
    }
    let bytes = run::output_bytes(model, &result.ids);
    if !bytes.is_empty() {
        let shown: String = bytes
            .iter()
            .map(|&c| if (0x20..0x7f).contains(&c) || c == b'\n' || c == b'\t' { c as char } else { '.' })
            .collect();
        println!("  output: {shown}");
    }
    Ok(verdict)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let opts = match parse(&argv) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let raw = match RawModel::load(&opts.model) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot load {}: {e}", opts.model.display());
            return ExitCode::FAILURE;
        }
    };
    let s = &raw.shapes;
    println!(
        "Loaded: vocab={} D={} layers={} heads={} d_ffn={}",
        s.vocab, s.d_model, s.n_layers, s.n_heads, s.d_ffn
    );
    if opts.kind == CacheKind::Brute {
        println!("Using the brute-force O(n) KV cache");
    }

    let device = Default::default();
    let model: Alm<Backend> = Alm::from_raw(&raw, &device);

    let (mut passed, mut failed, mut skipped) = (0, 0, 0);
    for path in &opts.programs {
        match check(&model, &raw, &opts, path) {
            Ok(Some(true)) => passed += 1,
            Ok(Some(false)) => failed += 1,
            Ok(None) => skipped += 1,
            Err(e) => {
                println!("ERROR {}: {e}", path.display());
                failed += 1;
            }
        }
    }
    println!("\n{passed} passed, {failed} failed, {skipped} no-ref");
    if failed > 0 { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}
