//! `alm-vm` — load a model, run WASM programs through it, check the output.
//!
//! The command line follows `model/transformer.cpp` so the two drivers can be
//! pointed at the same files:
//!
//! ```text
//! alm-vm model.bin [--brute|--lift] [--grid] [--trace[=N]] [--args=STR] [--max=N] prog.txt ...
//! ```

mod program;
mod run;

use alm_hull::TieBreak;
use alm_model::{Alm, CacheKind, KvCache, RawModel, Timings};
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
    grid: bool,
}

fn parse(argv: &[String]) -> Result<Options, String> {
    let mut model = None;
    let mut programs = Vec::new();
    let mut kind = CacheKind::Hull;
    let mut trace_every = 0;
    let mut args = None;
    let mut max_new = None;
    let mut grid = false;

    let mut it = argv.iter();
    while let Some(a) = it.next() {
        if a == "--brute" || a == "--nohull" {
            kind = CacheKind::Brute;
        } else if a == "--lift" {
            // The integer head compares keys rather than the ordinates they
            // were rounded into, and it can only do that on a query whose
            // `qy` is a unit -- so this option carries `--grid` with it
            // rather than quietly answering everything the old way.
            kind = CacheKind::Lift;
            grid = true;
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
        } else if a == "--grid" {
            grid = true;
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
    Ok(Options { model, programs, kind, trace_every, args, max_new, grid })
}

/// The benchmark line the C++ driver prints after every program has run.
#[derive(Default)]
struct Totals {
    tokens: usize,
    ops: usize,
    seconds: f64,
    timings: Timings,
}

impl Totals {
    fn add(&mut self, tokens: usize, ops: usize, seconds: f64, t: &Timings) {
        self.tokens += tokens;
        self.ops += ops;
        self.seconds += seconds;
        self.timings.proj += t.proj;
        self.timings.hull += t.hull;
        self.timings.head += t.head;
    }

    fn report(&self) {
        if self.seconds <= 0.0 {
            return;
        }
        let (tok, ops, dt) = (self.tokens as f64, self.ops as f64, self.seconds);
        println!("\nBenchmark: {} tok, {} ops, {dt:.2}s", self.tokens, self.ops);
        println!("  {:.0} tok/s, {:.0} wasm-ops/s", tok / dt, ops / dt);
        let t = &self.timings;
        let misc = dt - t.proj - t.hull - t.head;
        println!("\nTime breakdown:");
        for (name, v) in [("proj", t.proj), ("hull", t.hull), ("head", t.head), ("misc", misc)] {
            println!("  {name}:  {v:.3}s ({:4.1}%)", 100.0 * v / dt);
        }
    }
}

/// A fresh cache per program, with the per-head tie-break the file records.
fn cache_for(raw: &RawModel, kind: CacheKind, grid: bool) -> KvCache {
    let mut cache = KvCache::new(raw.shapes.n_layers, raw.shapes.n_heads, kind);
    cache.set_grid(grid);
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
fn check(
    model: &Alm,
    raw: &RawModel,
    opts: &Options,
    path: &Path,
    total: &mut Totals,
) -> std::io::Result<Option<bool>> {
    let program = Program::load(raw, path, opts.args.as_deref())?;
    let max_new = opts.max_new.unwrap_or_else(|| match &program.reference {
        Some(r) => (r.len() + 100).saturating_sub(program.ids.len()).max(100),
        None => MAX_GEN,
    });

    print!("{}: ", program.name);
    use std::io::Write;
    std::io::stdout().flush()?;

    let mut cache = cache_for(raw, opts.kind, opts.grid);
    let result = run::generate(model, &mut cache, &program.ids, max_new, opts.trace_every);
    let (n, ops, dt) = (result.ids.len(), result.ops, result.seconds);
    total.add(n, ops, dt, &result.timings);
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
    let q = cache.query_witness();
    if q.off > 0 {
        let (at, ulps) = q.worst_at.unwrap_or((0.0, 0.0));
        println!(
            "  queries: {}/{} not integers ({:.1}%), worst offset {:.3e} ({ulps:.1} ulp, at {at})",
            q.off,
            q.total,
            100.0 * q.off as f64 / q.total as f64,
            q.worst
        );
    }
    let gaps = cache.gap_witness();
    if gaps.total > 0 {
        println!(
            "  gaps: closest runner-up {:.3e} key steps, {} of {} under {:.0e} (todo3.md section 2a)",
            gaps.worst,
            gaps.noise,
            gaps.total,
            alm_hull::NOISE
        );
        if let Some(q) = gaps.worst_at {
            println!("    at: query {q:?}");
        }
        println!(
            "    against that gap: worst 2*u*(2+u)*terms/(best-second) = {:.3e}, {} unresolved of {} (ALM.DotError.cmp_of_dot_guard)",
            gaps.worst_guard, gaps.unresolved, gaps.total
        );
        if gaps.unresolved > 0 {
            println!(
                "    of those {} the exact dot product overturns {} (alm_hull::exact::dot_cmp)",
                gaps.unresolved, gaps.misranked
            );
        }
        if let Some(q) = gaps.worst_guard_at {
            println!("    at: query {q:?}");
        }
    }
    let lifts = cache.lift_witness();
    if lifts.total > 0 {
        println!(
            "  keys: {} lifted, {} marked, {} cleared, {} flat, {} off (of {})",
            lifts.lifted, lifts.marked, lifts.cleared, lifts.flat, lifts.off, lifts.total
        );
        println!(
            "    worst live offset {:.3e} of {:.3e}; {} past the wall, {} non-integer",
            lifts.worst,
            alm_hull::MARK_SPREAD,
            lifts.past_wall,
            lifts.noninteger
        );
        let hs = cache.lift_heads();
        let regime = |f: alm_hull::Family| hs.iter().filter(|h| h.regime() == Some(f)).count();
        let ever = |f: fn(&alm_hull::LiftWitness) -> usize| hs.iter().filter(|h| f(h) > 0).count();
        println!(
            "    heads: {} live, {} flat, {} mixed (of {}); {} ever clear, {} past the wall",
            regime(alm_hull::Family::Lifted),
            regime(alm_hull::Family::Flat),
            hs.iter().filter(|h| h.regime().is_none()).count(),
            hs.len(),
            ever(|h| h.cleared),
            ever(|h| h.past_wall)
        );
    }
    let seps = cache.sep_witness();
    if seps.distinct > 1 {
        let ss = cache.sep_heads();
        println!(
            "  spread: closest live keys {:.3e} apart of {} distinct ({} rewrites); floor {:.3e}",
            seps.worst, seps.distinct, seps.repeats, alm_hull::SEP_FLOOR
        );
        if let Some(p) = seps.worst_at {
            println!("    at: keys {p:?}");
        }
        println!(
            "    of the {} pair(s) under the floor, {} are one key rounded twice (ALM.HullTwin)",
            seps.under_floor, seps.twins
        );
        match seps.window() {
            Some(w) => println!(
                "    and the rest are {:.3e} apart, a window of {w:.3e} per key (ALM.HullSep.sq_dist_gap_of_sep)",
                seps.worst_apart
            ),
            None => println!(
                "    COLLISION: {} pair(s) leave no window at all (ALM.HullSep.nearest_fails_of_close) at {:?}",
                seps.collisions(),
                seps.worst_apart_at
            ),
        }
        println!(
            "    heads: {} of {} clear the floor, {} hold no collision",
            ss.iter().filter(|h| h.distinct > 0 && h.clears_the_floor()).count(),
            ss.iter().filter(|h| h.distinct > 0).count(),
            ss.iter().filter(|h| h.distinct > 0 && h.collisions() == 0).count()
        );
    }
    let lift = cache.lift_census();
    if lift.keys > 0 || lift.retired > 0 {
        println!(
            "  lift: {} quer(ies) on the integers, {} on the stored points, {} along an axis, {} on the hull",
            lift.integer, lift.stored, lift.axis, lift.hull
        );
        if lift.near > 0 {
            println!(
                "    {} of those integer(s) carried a residual the normalisation left (todo3.md section 4a)",
                lift.near
            );
        }
        if lift.cleared > 0 {
            println!(
                "    {} cleared key(s) held out of the container, {} quer(ies) settled by the marker and {} scored beside it",
                lift.cleared, lift.dominated, lift.mixed
            );
        }
        let cs = cache.lift_censuses();
        println!(
            "    {} of {} head(s) kept the integer path, holding {} key(s)",
            cs.iter().filter(|c| c.retired == 0 && c.keys > 0).count(),
            cs.iter().filter(|c| c.keys > 0 || c.retired > 0).count(),
            lift.keys
        );
        if let Some(k) = lift.retired_at {
            println!("    first key outside the live family: {k:?}");
        }
        // The heads `todo3.md` section 4b is about: the ones keyed on 32-bit
        // WASM values, whose `k^2` is past `2^53`.  Whether the integer path
        // reached them is the whole question, so it is printed rather than
        // left to be inferred from two other lines.
        let past = |keep: bool| {
            cs.iter()
                .zip(cache.lift_heads())
                .filter(|(c, w)| w.past_wall > 0 && (c.retired == 0) == keep)
                .count()
        };
        println!(
            "    of the head(s) past the old wall, {} kept the integer path and {} retired",
            past(true),
            past(false)
        );
    }
    let grid = cache.grid_witness();
    println!("  grid: worst ulp(score)/margin = {:.3}", grid.worst);
    if let Some(w) = grid.worst_at {
        println!("    at: query {:?}, key {:?}", w.query, w.key);
    }
    // Under the integer cache the ratio above is measured over part of the
    // run, and the part matters: the integer path compares in `i128` and has
    // no rounding to weigh (`ALM.LiftCompare.upper_lt_iff`), so a worst ratio
    // of zero says nothing until the size of the weighed population is beside
    // it.  Everything answered from a stored ordinate is weighed.
    if lift.keys > 0 || lift.retired > 0 {
        let weighed = lift.stored + lift.axis + lift.hull;
        println!("    over the {} quer(ies) answered from a stored ordinate, of {}", weighed, weighed + lift.integer);
    }
    if let Some(first) = grid.first {
        println!(
            "  OFF THE GRID: {} quer(ies) won with no margin to spare (todo3.md section 4)",
            grid.count
        );
        println!("    first: score {} at query {:?}, key {:?}", first.score, first.query, first.key);
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
    if opts.kind == CacheKind::Lift {
        println!("Using the integer KV cache: the wall moves to 2^52 (todo3.md section 4)");
    }
    if opts.grid {
        println!("Querying at unit scale (todo3.md section 0)");
    }

    let model = Alm::from_raw(&raw);
    let (nz, total) = model.head_density();
    println!("Head sparsity: {nz}/{total} nonzero ({:.0}% sparse)", 100.0 * (1.0 - nz as f64 / total as f64));

    let (mut passed, mut failed, mut skipped) = (0, 0, 0);
    let mut totals = Totals::default();
    for path in &opts.programs {
        match check(&model, &raw, &opts, path, &mut totals) {
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
    totals.report();
    if failed > 0 { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}
