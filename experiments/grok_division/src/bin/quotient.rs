//! Measures, at checkpoints of a run, how far the answer's logits are from depending on an
//! equation through its quotient alone.
//!
//!     quotient RUN_DIR [STEP...]
//!
//! Scaling the numerator by `s` and the denominator by `t`, both nonzero, sends the equation
//! `a / b = c` to `(s a) / (t b) = (s / t) c`. The logits are *quotient-equivariant* if those of
//! the second equation are those of the first with every residue `r` relabelled `(s / t) r` and
//! the other 142 tokens kept. The equations with a nonzero numerator are one orbit of the
//! scalings, and each of them comes from `1 / 1 = 1` by one scaling. Those with numerator zero
//! are another orbit, and there `s` is free: equivariant logits are equal on the 96 nonzero
//! residues.
//!
//! For each checkpoint (all of the run if no step is given) the logits at `=` of the 9312
//! equations are computed by the reference model in doubles. A row of logits is relabelled so
//! that its equation becomes `1 / 1 = 1`, or `0 / 1 = 0` in each of the 96 ways. For an anchor,
//! a training equation, the *structural error* is the least `ε` such that every relabelled row
//! of the orbit is within `ε` of the anchor's in every token after one constant is added to the
//! row. Every equation of the orbit then has a margin of at least the anchor's less `2 ε`
//! (`certified_margin`), so a positive value certifies every answer of the orbit from that one
//! training answer and the structure of the logits. The anchor printed has the largest such
//! value among the training equations of the orbit.
//!
//! One line of JSON for each checkpoint: the wrong answers among the training and the held-out
//! equations, how many held-out predictions are those the run recorded at that step, and for
//! each orbit (`nonzero`, `zero`) its equations, its training equations, the least margin of
//! all and of the training ones, the largest of the training ones, the anchor with its margin
//! and structural error, the median over the relabelled rows of their own error
//! (`error_median`), how many of the rows are within half the anchor's margin of the anchor
//! (`rows_certified` of `rows`), and `raw_spread`, the largest difference between two relabelled
//! rows in one token with no constant added.
//!
//! The same is printed for the rows of the 97 residues alone (`residues_nonzero`,
//! `residues_zero`), where a margin is over the residues, beside `other_gap`: the least over the
//! equations of the largest logit of a residue less the largest of another token. Where that is
//! positive the decision is a residue, and the margins over the residues decide it.
//!
//! It shares the data, the layout and the reference model in doubles with the trainer and
//! nothing of the trainer's fast path.

// The shared modules have items that only the trainer or its tests use.
#![allow(dead_code)]

#[path = "../checkpoint.rs"]
mod checkpoint;
#[path = "../data.rs"]
mod data;
#[path = "../layout.rs"]
mod layout;
#[path = "../mt.rs"]
mod mt;
#[path = "../reference.rs"]
mod reference;

use std::path::Path;
use std::process::ExitCode;

use data::{ANSWER_POS, Equation, NUM0, P};
use layout::Dims;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("quotient: {message}");
            ExitCode::FAILURE
        }
    }
}

/// The logits of the answer of every equation, by the reference model.
fn answer_logits(dims: Dims, p: &[f64], equations: &[Equation], threads: usize) -> Vec<Vec<f64>> {
    let mut rows = vec![Vec::new(); equations.len()];
    let chunk = equations.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for (part, out) in equations.chunks(chunk).zip(rows.chunks_mut(chunk)) {
            scope.spawn(move || {
                for (equation, row) in part.iter().zip(out) {
                    let all = reference::logits(dims, p, &equation.tokens()[..dims.ctx]);
                    *row = all[ANSWER_POS * dims.vocab..(ANSWER_POS + 1) * dims.vocab].to_vec();
                }
            });
        }
    });
    rows
}

/// The row with the residues relabelled: the entry of the residue `r` is the logit of the
/// residue `scale * r`, and the other tokens keep their entries. The residue 0 is at `first`.
fn relabel(row: &[f64], scale: usize, first: usize) -> Vec<f64> {
    let mut out = row.to_vec();
    for r in 0..P {
        out[first + r] = row[first + scale * r % P];
    }
    out
}

/// The logit of `answer` less the largest of the others.
fn margin(row: &[f64], answer: usize) -> f64 {
    let others = row.iter().enumerate().filter(|(token, _)| *token != answer).map(|(_, v)| *v);
    row[answer] - others.fold(f64::NEG_INFINITY, f64::max)
}

/// The first of the largest logits.
fn decision(row: &[f64]) -> usize {
    let top = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    row.iter().position(|&v| v == top).expect("a row has a largest entry")
}

/// An orbit of the scalings.
struct Orbit {
    /// The token of the answer in a relabelled row.
    answer: usize,
    /// The relabelled rows, one after another.
    rows: Vec<f64>,
    /// For each equation of the orbit: its margin and whether it is a training equation.
    equations: Vec<(Equation, f64, bool)>,
    /// The anchors: the index of a training equation's row, and of the equation.
    anchors: Vec<(usize, usize)>,
}

/// The orbit of the nonzero numerators, then that of the numerator zero, from rows of logits
/// that have the residue 0 at `first`: rows of all tokens, or of the residues alone.
fn orbits(first: usize, equations: &[Equation], logits: &[Vec<f64>], training: &[bool]) -> [Orbit; 2] {
    let empty = |answer| Orbit { answer, rows: Vec::new(), equations: Vec::new(), anchors: Vec::new() };
    let (mut nonzero, mut zero) = (empty(first + 1), empty(first));
    for ((equation, row), &train) in equations.iter().zip(logits).zip(training) {
        let quo = equation.quo as usize;
        // The scalings that bring the equation to the first of its orbit: one, or all 96.
        let (orbit, scales) = if quo != 0 { (&mut nonzero, quo..quo + 1) } else { (&mut zero, 1..P) };
        if train {
            orbit.anchors.push((orbit.rows.len() / row.len(), orbit.equations.len()));
        }
        for scale in scales {
            orbit.rows.extend(relabel(row, scale, first));
        }
        orbit.equations.push((*equation, margin(row, first + quo), train));
    }
    [nonzero, zero]
}

/// For each row of `rows`, the least `ε` such that the row is within `ε` of `anchor` in every
/// token after one constant is added to the row: half the range of their differences.
fn errors<'a>(anchor: &'a [f64], rows: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    rows.chunks_exact(anchor.len()).map(move |row| {
        let (mut least, mut largest) = (f64::INFINITY, f64::NEG_INFINITY);
        for (a, b) in row.iter().zip(anchor) {
            least = least.min(a - b);
            largest = largest.max(a - b);
        }
        (largest - least) / 2.0
    })
}

/// The structural error of an anchor: the largest of the errors of the rows.
fn structural_error(anchor: &[f64], rows: &[f64]) -> f64 {
    errors(anchor, rows).fold(0.0, f64::max)
}

/// The anchor with the largest margin less twice its structural error: its row, the equation,
/// its margin and its error.
fn best_anchor(orbit: &Orbit, vocab: usize, threads: usize) -> Option<(usize, Equation, f64, f64)> {
    let chunk = orbit.anchors.len().div_ceil(threads).max(1);
    let mut found: Vec<(usize, Equation, f64, f64)> = Vec::new();
    std::thread::scope(|scope| {
        let workers: Vec<_> = orbit
            .anchors
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    let measured = part.iter().map(|&(row, equation)| {
                        let error = structural_error(&orbit.rows[row * vocab..(row + 1) * vocab], &orbit.rows);
                        (row, orbit.equations[equation].0, orbit.equations[equation].1, error)
                    });
                    measured.collect::<Vec<_>>()
                })
            })
            .collect();
        for worker in workers {
            found.extend(worker.join().expect("a worker of the anchors"));
        }
    });
    let certified = |a: &(usize, Equation, f64, f64)| a.2 - 2.0 * a.3;
    found.into_iter().reduce(|best, a| if certified(&a) > certified(&best) { a } else { best })
}

fn num(value: f64) -> String {
    if value.is_finite() { format!("{value:.6}") } else { "null".to_string() }
}

/// The fields of an orbit, as the head of the file lists them.
fn orbit_fields(orbit: &Orbit, vocab: usize, threads: usize) -> String {
    let least = |margins: &mut dyn Iterator<Item = f64>| margins.fold(f64::INFINITY, f64::min);
    let training = || orbit.equations.iter().filter(|e| e.2).map(|e| e.1);
    let spread = (0..vocab)
        .map(|token| {
            let column = || orbit.rows.iter().skip(token).step_by(vocab).copied();
            column().fold(f64::NEG_INFINITY, f64::max) - column().fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max);
    let anchor = match best_anchor(orbit, vocab, threads) {
        Some((row, equation, margin, error)) => {
            let mut all: Vec<f64> = errors(&orbit.rows[row * vocab..(row + 1) * vocab], &orbit.rows).collect();
            all.sort_by(f64::total_cmp);
            format!(
                "\"anchor\":[{},{}],\"anchor_margin\":{},\"structural_error\":{},\"certified_margin\":{},\
                 \"error_median\":{},\"rows\":{},\"rows_certified\":{}",
                equation.num,
                equation.den,
                num(margin),
                num(error),
                num(margin - 2.0 * error),
                num(all[all.len() / 2]),
                all.len(),
                all.iter().filter(|e| margin - 2.0 * **e > 0.0).count()
            )
        }
        None => "\"anchor\":null".into(),
    };
    format!(
        "{{\"equations\":{},\"training\":{},\"margin_min\":{},\"margin_min_training\":{},\
         \"margin_max_training\":{},{anchor},\"raw_spread\":{}}}",
        orbit.equations.len(),
        training().count(),
        num(least(&mut orbit.equations.iter().map(|e| e.1))),
        num(least(&mut training())),
        num(training().fold(f64::NEG_INFINITY, f64::max)),
        num(spread)
    )
}

/// The held-out equations of a run, in the order of its records.
fn heldout(run: &Path) -> Result<Vec<Equation>, String> {
    let text = std::fs::read_to_string(run.join("heldout.txt")).map_err(|e| format!("heldout.txt: {e}"))?;
    text.lines()
        .map(|line| {
            let fields: Vec<usize> = line.split_whitespace().filter_map(|f| f.parse().ok()).collect();
            match fields[..] {
                [num, den, quo] if num < P && (1..P).contains(&den) && quo < P && quo * den % P == num => {
                    Ok(Equation { num: num as u8, den: den as u8, quo: quo as u8 })
                }
                _ => Err(format!("heldout.txt: not an equation: {line}")),
            }
        })
        .collect()
}

/// The predictions that the run recorded for its held-out equations at `step`, if it did.
fn recorded(run: &Path, step: u64, heldout: usize) -> Option<Vec<u8>> {
    let history = std::fs::read_to_string(run.join("history.jsonl")).ok()?;
    let at = history.lines().position(|line| line.starts_with(&format!("{{\"step\":{step},")))?;
    let bytes = std::fs::read(run.join("heldout_predictions.bin")).ok()?;
    bytes.get(at * heldout..(at + 1) * heldout).map(<[u8]>::to_vec)
}

/// The steps of the checkpoints of a run, in order.
fn checkpoints(run: &Path) -> Result<Vec<u64>, String> {
    let entries = std::fs::read_dir(run).map_err(|e| format!("{}: {e}", run.display()))?;
    let names = entries.filter_map(|entry| entry.ok()?.file_name().into_string().ok());
    let mut steps: Vec<u64> =
        names.filter_map(|name| name.strip_prefix("checkpoint-")?.strip_suffix(".bin")?.parse().ok()).collect();
    steps.sort_unstable();
    Ok(steps)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(dir) = args.first().map(Path::new) else {
        return Err("usage: quotient RUN_DIR [STEP...]".into());
    };
    let steps: Vec<u64> = match &args[1..] {
        [] => checkpoints(dir)?,
        given => given.iter().map(|s| s.parse().map_err(|_| format!("{s} is not a step"))).collect::<Result<_, _>>()?,
    };
    let (dims, equations, heldout) = (Dims::PAPER, data::equations(), heldout(dir)?);
    let index = |e: &Equation| e.quo as usize * (P - 1) + e.den as usize - 1;
    let mut training = vec![true; equations.len()];
    for equation in &heldout {
        training[index(equation)] = false;
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16);
    for step in steps {
        let p = checkpoint::parameters(&dir.join(format!("checkpoint-{step:07}.bin")), dims)?;
        let logits = answer_logits(dims, &p, &equations, threads);
        let wrong = |train: bool| {
            let rows = equations.iter().zip(&logits).zip(&training).filter(|(_, t)| **t == train);
            rows.filter(|((equation, row), _)| decision(row) != NUM0 + equation.quo as usize).count()
        };
        let agree = recorded(dir, step, heldout.len()).map(|bytes| {
            let same = |(equation, byte): &(&Equation, &u8)| decision(&logits[index(equation)]) == **byte as usize;
            heldout.iter().zip(&bytes).filter(same).count()
        });
        // The largest logit of a residue less the largest of another token, at its least.
        let largest = |row: &[f64]| row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let gap = |row: &Vec<f64>| largest(&row[NUM0..NUM0 + P]) - largest(&row[..NUM0]).max(largest(&row[NUM0 + P..]));
        let other_gap = logits.iter().map(gap).fold(f64::INFINITY, f64::min);
        let residues: Vec<Vec<f64>> = logits.iter().map(|row| row[NUM0..NUM0 + P].to_vec()).collect();
        let [nonzero, zero] = orbits(NUM0, &equations, &logits, &training);
        let [residues_nonzero, residues_zero] = orbits(0, &equations, &residues, &training);
        println!(
            "{{\"run\":\"{}\",\"step\":{step},\"errors_training\":{},\"errors_heldout\":{},\"recorded_agree\":{},\
             \"nonzero\":{},\"zero\":{},\"other_gap\":{},\"residues_nonzero\":{},\"residues_zero\":{}}}",
            dir.display(),
            wrong(true),
            wrong(false),
            agree.map_or("null".to_string(), |n| n.to_string()),
            orbit_fields(&nonzero, dims.vocab, threads),
            orbit_fields(&zero, dims.vocab, threads),
            num(other_gap),
            orbit_fields(&residues_nonzero, P, threads),
            orbit_fields(&residues_zero, P, threads)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB: usize = 239;

    /// Rows of logits that are a function of the quotient of the class to the answer, with the
    /// answer `bonus` above it, and equal on the nonzero residues where the numerator is zero.
    fn table(shape: impl Fn(usize) -> f64, bonus: f64, bend: impl Fn(&Equation, usize) -> f64) -> Vec<Vec<f64>> {
        let inverse = |c: usize| (1..P).find(|i| i * c % P == 1).expect("a nonzero residue has an inverse");
        let rows = data::equations().into_iter().map(|equation| {
            let quo = equation.quo as usize;
            let mut row = vec![-1.0; VOCAB];
            for class in 0..P {
                let value = if quo != 0 { shape(class * inverse(quo) % P) } else { 0.5 };
                row[NUM0 + class] = value + if class == quo { bonus } else { 0.0 } + bend(&equation, class);
            }
            row
        });
        rows.collect()
    }

    #[test]
    fn relabelling_permutes_the_residues_and_keeps_the_margin() {
        let row: Vec<f64> = (0..VOCAB).map(|token| ((token * 37) % 101) as f64).collect();
        let moved = relabel(&row, 5, NUM0);
        assert_eq!(moved[NUM0 + 3], row[NUM0 + 15]);
        assert_eq!(moved[NUM0 + 96], row[NUM0 + 5 * 96 % 97]);
        assert_eq!(
            (moved[..NUM0].to_vec(), moved[NUM0 + P..].to_vec()),
            (row[..NUM0].to_vec(), row[NUM0 + P..].to_vec())
        );
        // The answer 5 of an equation becomes the class 1, with the margin it had.
        assert_eq!(margin(&moved, NUM0 + 1), margin(&row, NUM0 + 5));
        assert_eq!(decision(&[1.0, 3.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn structural_error_is_half_the_range_of_the_differences() {
        let anchor = [1.0, 4.0, -2.0];
        // The first row is the anchor moved by a constant; the second also has one entry 3 higher.
        let rows = [11.0, 14.0, 8.0, 0.0, 6.0, -3.0];
        assert_eq!(structural_error(&anchor, &rows[..3]), 0.0);
        assert_eq!(structural_error(&anchor, &rows), 1.5);
    }

    /// Logits that are a function of the quotient have no structural error, and every margin is
    /// the anchor's; a bend of one training row is found, and the bound it gives holds.
    #[test]
    fn equivariant_logits_are_certified_and_a_bend_is_measured() {
        let equations = data::equations();
        let training: Vec<bool> = (0..equations.len()).map(|i| i % 40 == 0).collect();
        let shape = |ratio: usize| ((ratio * ratio) % 7) as f64 / 7.0;
        let exact = table(shape, 2.0, |_, _| 0.0);
        for orbit in orbits(NUM0, &equations, &exact, &training) {
            let (_, _, margin, error) = best_anchor(&orbit, VOCAB, 3).expect("the orbit has training equations");
            assert!(error < 1e-12 && margin > 1.0, "error {error}, margin {margin}");
            assert!(orbit.equations.iter().all(|e| (e.1 - margin).abs() < 1e-12));
        }
        // One held-out equation has a competitor raised by 0.75: every anchor is that far from it.
        let bent = table(shape, 2.0, |e, class| if (e.num, e.den, class) == (4, 1, 9) { 0.75 } else { 0.0 });
        let [nonzero, zero] = orbits(NUM0, &equations, &bent, &training);
        assert_eq!((nonzero.equations.len(), zero.equations.len()), (96 * 96, 96));
        assert_eq!((nonzero.rows.len(), zero.rows.len()), (96 * 96 * VOCAB, 96 * 96 * VOCAB));
        let (_, _, margin, error) = best_anchor(&nonzero, VOCAB, 2).expect("anchors");
        assert!((error - 0.375).abs() < 1e-12, "error {error}");
        let least = nonzero.equations.iter().map(|e| e.1).fold(f64::INFINITY, f64::min);
        assert!(least >= margin - 2.0 * error - 1e-12 && least < margin);
        assert!(best_anchor(&zero, VOCAB, 2).expect("anchors").3 < 1e-12);
    }
}
