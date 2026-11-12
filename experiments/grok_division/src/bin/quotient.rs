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
//! `certified_exact` repeats the test of the anchor without rounding: the scores of the model
//! are doubles, each is the rational that its bits say, and the margin is compared with the
//! range of the differences in exact arithmetic. It is `null` if a number leaves the 127 bits of
//! the exact type.
//!
//! One anchor for a whole orbit asks more than the answers need. For each equation of the orbit
//! that is not a training equation (`heldout` of them), `nearest_certified` counts those with
//! some training anchor whose margin exceeds twice the error of the equation's row against it;
//! on the orbit of numerator zero any one of the 96 relabellings of the equation serves.
//! `nearest_exact` counts those whose best pair passes the test in exact arithmetic, and
//! `nearest_min` is the least over the equations of the best margin less twice the error. If
//! every held-out equation of both orbits is counted, every held-out answer follows from
//! training answers and the measured errors.
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

/// The least `ε` such that `row` is within `ε` of `anchor` in every token after one constant is
/// added to the row: half the range of their differences.
fn error(row: &[f64], anchor: &[f64]) -> f64 {
    let (mut least, mut largest) = (f64::INFINITY, f64::NEG_INFINITY);
    for (a, b) in row.iter().zip(anchor) {
        least = least.min(a - b);
        largest = largest.max(a - b);
    }
    (largest - least) / 2.0
}

/// The errors of the rows of `rows` against `anchor`.
fn errors<'a>(anchor: &'a [f64], rows: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    rows.chunks_exact(anchor.len()).map(move |row| error(row, anchor))
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

/// A binary score as the rational `mantissa * 2^exponent`, with the mantissa odd or zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Dyadic {
    mantissa: i128,
    exponent: i32,
}

impl Dyadic {
    fn new(mantissa: i128, exponent: i32) -> Dyadic {
        if mantissa == 0 {
            return Dyadic { mantissa: 0, exponent: 0 };
        }
        let twos = mantissa.trailing_zeros();
        Dyadic { mantissa: mantissa >> twos, exponent: exponent + twos as i32 }
    }

    /// The value of a finite double, read from its bits.
    fn of(value: f64) -> Option<Dyadic> {
        if !value.is_finite() {
            return None;
        }
        let bits = value.to_bits();
        let (biased, fraction) = (((bits >> 52) & 0x7ff) as i32, (bits & ((1 << 52) - 1)) as i128);
        let magnitude =
            if biased == 0 { Dyadic::new(fraction, -1074) } else { Dyadic::new(fraction | 1 << 52, biased - 1075) };
        Some(if bits >> 63 == 0 { magnitude } else { Dyadic { mantissa: -magnitude.mantissa, ..magnitude } })
    }

    /// The two mantissas over a common exponent, and that exponent: the smaller of the two, or
    /// that of the one that is not zero. Nothing if a mantissa leaves 127 bits.
    fn aligned(self, other: Dyadic) -> Option<(i128, i128, i32)> {
        let exponent = match (self.mantissa, other.mantissa) {
            (0, _) => other.exponent,
            (_, 0) => self.exponent,
            _ => self.exponent.min(other.exponent),
        };
        let lift = |d: Dyadic| match (d.mantissa, d.exponent - exponent) {
            (0, _) => Some(0),
            (_, shift) if shift >= 126 => None,
            (mantissa, shift) => mantissa.checked_mul(1i128 << shift),
        };
        Some((lift(self)?, lift(other)?, exponent))
    }

    fn sub(self, other: Dyadic) -> Option<Dyadic> {
        let (a, b, exponent) = self.aligned(other)?;
        Some(Dyadic::new(a.checked_sub(b)?, exponent))
    }

    fn cmp(self, other: Dyadic) -> Option<std::cmp::Ordering> {
        let (a, b, _) = self.aligned(other)?;
        Some(a.cmp(&b))
    }

    fn max(self, other: Dyadic) -> Option<Dyadic> {
        Some(if self.cmp(other)?.is_lt() { other } else { self })
    }

    fn min(self, other: Dyadic) -> Option<Dyadic> {
        Some(if self.cmp(other)?.is_gt() { other } else { self })
    }
}

/// Whether the margin of `answer` in `anchor` exceeds twice the structural error of `anchor`
/// over `rows`, with the scores taken as the rationals they are: for every row, the range of
/// its differences from the anchor is less than the margin. Nothing is rounded; `None` if a
/// score is not finite or a number leaves the 127 bits of the exact type.
fn certified_exactly(anchor: &[f64], rows: &[f64], answer: usize) -> Option<bool> {
    let exact: Vec<Dyadic> = anchor.iter().map(|v| Dyadic::of(*v)).collect::<Option<_>>()?;
    let mut others = exact.iter().enumerate().filter(|(token, _)| *token != answer).map(|(_, v)| *v);
    let first = others.next()?;
    let competitor = others.try_fold(first, Dyadic::max)?;
    let margin = exact[answer].sub(competitor)?;
    for row in rows.chunks_exact(anchor.len()) {
        let mut differences = row.iter().zip(&exact).map(|(a, b)| Dyadic::of(*a)?.sub(*b));
        let first = differences.next()??;
        let (mut least, mut largest) = (first, first);
        for difference in differences {
            let difference = difference?;
            (least, largest) = (least.min(difference)?, largest.max(difference)?);
        }
        if !largest.sub(least)?.cmp(margin)?.is_lt() {
            return Some(false);
        }
    }
    Some(true)
}

/// For each equation of the orbit that is not a training equation: the training anchor and the
/// relabelling of the equation with the largest margin of the anchor less twice the error of
/// the equation's row against the anchor's, as (row of the equation, row of the anchor, value).
/// An equation has `per` relabelled rows, of which any one serves.
fn nearest(orbit: &Orbit, vocab: usize, per: usize, threads: usize) -> Vec<(usize, usize, f64)> {
    let heldout: Vec<usize> = (0..orbit.equations.len()).filter(|k| !orbit.equations[*k].2).collect();
    let row = |index: usize| &orbit.rows[index * vocab..(index + 1) * vocab];
    let chunk = heldout.len().div_ceil(threads).max(1);
    let mut found = Vec::new();
    std::thread::scope(|scope| {
        let workers: Vec<_> = heldout
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    let best = |k: &usize| {
                        let pairs =
                            (k * per..(k + 1) * per).flat_map(|own| orbit.anchors.iter().map(move |a| (own, *a)));
                        let valued = pairs.map(|(own, (anchor, equation))| {
                            (own, anchor, orbit.equations[equation].1 - 2.0 * error(row(own), row(anchor)))
                        });
                        valued.reduce(|best, pair| if pair.2 > best.2 { pair } else { best })
                    };
                    part.iter().filter_map(best).collect::<Vec<_>>()
                })
            })
            .collect();
        for worker in workers {
            found.extend(worker.join().expect("a worker of the nearest anchors"));
        }
    });
    found
}

fn num(value: f64) -> String {
    if value.is_finite() { format!("{value:.6}") } else { "null".to_string() }
}

/// The fields of an orbit, as the head of the file lists them. With `per`, the relabelled rows
/// of an equation, also the fields of the nearest anchors.
fn orbit_fields(orbit: &Orbit, vocab: usize, per: Option<usize>, threads: usize) -> String {
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
            let exact = certified_exactly(&orbit.rows[row * vocab..(row + 1) * vocab], &orbit.rows, orbit.answer);
            format!(
                "\"anchor\":[{},{}],\"anchor_margin\":{},\"structural_error\":{},\"certified_margin\":{},\
                 \"certified_exact\":{},\"error_median\":{},\"rows\":{},\"rows_certified\":{}",
                equation.num,
                equation.den,
                num(margin),
                num(error),
                num(margin - 2.0 * error),
                exact.map_or("null".to_string(), |certified| certified.to_string()),
                num(all[all.len() / 2]),
                all.len(),
                all.iter().filter(|e| margin - 2.0 * **e > 0.0).count()
            )
        }
        None => "\"anchor\":null".into(),
    };
    let nearest = per.map_or(String::new(), |per| {
        let pairs = nearest(orbit, vocab, per, threads);
        let row = |index: usize| &orbit.rows[index * vocab..(index + 1) * vocab];
        let exact = |(own, anchor, _): &&(usize, usize, f64)| {
            certified_exactly(row(*anchor), row(*own), orbit.answer) == Some(true)
        };
        format!(
            ",\"heldout\":{},\"nearest_certified\":{},\"nearest_exact\":{},\"nearest_min\":{}",
            orbit.equations.iter().filter(|e| !e.2).count(),
            pairs.iter().filter(|pair| pair.2 > 0.0).count(),
            pairs.iter().filter(exact).count(),
            num(least(&mut pairs.iter().map(|pair| pair.2)))
        )
    });
    format!(
        "{{\"equations\":{},\"training\":{},\"margin_min\":{},\"margin_min_training\":{},\
         \"margin_max_training\":{},{anchor},\"raw_spread\":{}{nearest}}}",
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
            orbit_fields(&nonzero, dims.vocab, Some(1), threads),
            orbit_fields(&zero, dims.vocab, Some(P - 1), threads),
            num(other_gap),
            orbit_fields(&residues_nonzero, P, None, threads),
            orbit_fields(&residues_zero, P, None, threads)
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

    /// A double is the rational that its bits say: `0.1` is `3602879701896397 / 2^55`, and ten
    /// times that exceeds `1` by `2^-54`, which is computed here in integers.
    #[test]
    fn binary_scores_are_read_as_exact_rationals() {
        assert_eq!(Dyadic::of(0.1), Some(Dyadic { mantissa: 3602879701896397, exponent: -55 }));
        assert_eq!(10 * 3602879701896397_i128 - (1_i128 << 55), 2);
        let excess = Dyadic::new(10 * 3602879701896397, -55).sub(Dyadic::of(1.0).unwrap());
        assert_eq!(excess, Some(Dyadic { mantissa: 1, exponent: -54 }));
        assert_eq!(Dyadic::of(-7.25), Some(Dyadic { mantissa: -29, exponent: -2 }));
        assert_eq!(Dyadic::of(-0.0), Some(Dyadic { mantissa: 0, exponent: 0 }));
        assert_eq!(Dyadic::of(f64::from_bits(1)), Some(Dyadic { mantissa: 1, exponent: -1074 }));
        assert_eq!((Dyadic::of(f64::NAN), Dyadic::of(f64::INFINITY)), (None, None));
        // The doubles 0.1 and 0.2 sum to the double 0.3 and 2^-55, in integers over 2^55; the
        // sum in doubles is rounded to the next double, 2^-54 above.
        assert_eq!(Dyadic::of(0.3), Some(Dyadic { mantissa: 5404319552844595, exponent: -54 }));
        assert_eq!(3602879701896397_i128 + 2 * 3602879701896397 - 2 * 5404319552844595, 1);
        let (tenth, fifth, sum) = (Dyadic::of(0.1).unwrap(), Dyadic::of(0.2).unwrap(), Dyadic::of(0.3).unwrap());
        let left = tenth.sub(Dyadic { mantissa: -fifth.mantissa, ..fifth }).unwrap();
        assert_eq!(left.sub(sum), Some(Dyadic { mantissa: 1, exponent: -55 }));
        assert_eq!(left.cmp(sum), Some(std::cmp::Ordering::Greater));
        assert_eq!(Dyadic::of(0.1 + 0.2 - 0.3), Some(Dyadic { mantissa: 1, exponent: -54 }));
        // A zero aligns with anything; two numbers 200 binary orders apart do not fit.
        assert_eq!(Dyadic::of(0.0).unwrap().sub(Dyadic::of(1e-300).unwrap()), Dyadic::of(-1e-300));
        assert_eq!(Dyadic::of(1e30).unwrap().sub(Dyadic::of(1e-30).unwrap()), None);
    }

    /// On scores that are eighths the exact test is a calculation in integers, and the doubles
    /// agree with it. Where a difference of two scores is rounded, the doubles lose the
    /// certificate that the exact test keeps.
    #[test]
    fn exact_certificate_agrees_with_integers_and_not_with_rounded_doubles() {
        let eighths = |values: &[i64]| values.iter().map(|v| *v as f64 / 8.0).collect::<Vec<f64>>();
        let anchor = [3_i64, 40, -16, 7];
        let range = |row: &[i64; 4]| {
            let differences = || row.iter().zip(&anchor).map(|(a, b)| a - b);
            differences().max().unwrap() - differences().min().unwrap()
        };
        // In eighths: the margin of the answer 1 is 40 - 7 = 33.
        let margin_eighths =
            anchor[1] - anchor.iter().enumerate().filter(|(t, _)| *t != 1).map(|(_, v)| *v).max().unwrap();
        assert_eq!(margin_eighths, 33);
        for (last, certified) in [([0_i64, 52, -30, 9], true), ([0, 59, -30, 9], false), ([0, 60, -30, 9], false)] {
            let rows = [[11_i64, 48, -8, 15], last];
            // The first row is the anchor moved by a constant: its range is 0.
            assert_eq!(range(&rows[0]), 0);
            assert_eq!(range(&last) < margin_eighths, certified, "range {}", range(&last));
            let scores: Vec<f64> = rows.iter().flat_map(|row| eighths(row)).collect();
            assert_eq!(certified_exactly(&eighths(&anchor), &scores, 1), Some(certified));
            let doubles = margin(&eighths(&anchor), 1) - 2.0 * structural_error(&eighths(&anchor), &scores);
            assert_eq!(doubles, (margin_eighths - range(&last)) as f64 / 8.0);
        }
        // 3 + 10^16 is not a double: the range 10^16 + 3 is rounded to the margin 10^16 + 4.
        let (anchor, row) = ([1e16 + 4.0, 0.0, -1e16], [1e16 + 4.0, 0.0, 3.0]);
        assert_eq!(10_i128.pow(16) + 3 < 10_i128.pow(16) + 4, true);
        assert_eq!(certified_exactly(&anchor, &row, 0), Some(true));
        assert_eq!(margin(&anchor, 0) - 2.0 * structural_error(&anchor, &row), 0.0);
        assert_eq!(certified_exactly(&[1.0, f64::NAN], &[0.0, 0.0], 0), None);
    }

    /// Every held-out equation of an equivariant table has an anchor at no distance; one whose
    /// row is bent by more than the margin has none, and it alone is not certified.
    #[test]
    fn heldout_equations_find_their_nearest_anchors() {
        let equations = data::equations();
        let training: Vec<bool> = (0..equations.len()).map(|i| i % 40 == 0).collect();
        let shape = |ratio: usize| ((ratio * ratio) % 7) as f64 / 7.0;
        let bent = table(shape, 2.0, |e, class| match (e.num, e.den, class) {
            (4, 1, 9) => 0.75,
            (10, 2, 9) => 9.0,
            (0, 3, 2) => 9.0,
            _ => 0.0,
        });
        let [nonzero, zero] = orbits(NUM0, &equations, &bent, &training);
        for (orbit, per, broken) in [(&nonzero, 1, 1), (&zero, P - 1, 1)] {
            let pairs = nearest(orbit, VOCAB, per, 3);
            let heldout = orbit.equations.iter().filter(|e| !e.2).count();
            assert_eq!(pairs.len(), heldout);
            assert_eq!(pairs.iter().filter(|pair| pair.2 > 0.0).count(), heldout - broken);
            let row = |index: usize| &orbit.rows[index * VOCAB..(index + 1) * VOCAB];
            for (own, anchor, value) in &pairs {
                let exact = certified_exactly(row(*anchor), row(*own), orbit.answer);
                assert_eq!(exact, Some(*value > 0.0), "row {own} against {anchor}: {value}");
            }
        }
        // The equation bent by 0.75 keeps the margin of its anchor less twice 0.375.
        let values: Vec<f64> = nearest(&nonzero, VOCAB, 1, 2).iter().map(|pair| pair.2).collect();
        let largest = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(values.iter().any(|value| (value - (largest - 0.75)).abs() < 1e-12));
    }
}
