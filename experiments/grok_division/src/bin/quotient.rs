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
//! With `--enclose` before the run, the line also has `enclosed`: what holds for the decoder in
//! exact reals of `Protocol/Model.lean` at the checkpoint's parameters, which the model in
//! doubles only approximates. `enclosure.rs` computes, for every logit, two doubles that the
//! real logit lies between: in intervals of doubles, and again in its wider arithmetic for the
//! `refined` rows, those whose intervals are wider than `REFINE`, or than the width given as
//! `--enclose=WIDTH`, or do not decide the answer. The two enclosures of a refined row hold the
//! same logits, so they have a real in common in every token, or the tool stops; with
//! `--enclose=0` every row is computed twice and compared so. `doubles` has the largest width
//! and the number of answers not decided of the intervals of doubles alone, before a row was
//! refined. `width` is the largest distance between the two doubles. `training` and
//! `heldout` count the equations whose answer is the largest logit for certain, those where it
//! is not for certain, and those that the enclosure does not decide; `margin_lower_min` is the
//! least lower bound of a margin. For each orbit, `certified` counts the held-out equations
//! whose nearest anchor passes the test on the ends of the intervals: the anchor's margin is at
//! least the lower end of its answer less the largest upper end of another token, the
//! differences of the two rows lie between the least lower end less upper end and the largest
//! upper end less lower end, and the first exceeds the distance of the last two. `slack_min` is
//! the least excess. `enclosure_gives_certificate` of `Protocol/Enclosure.lean` proves that
//! this test gives the premise of the certificate for the real logits.
//!
//! It shares the data, the layout and the reference model in doubles with the trainer and
//! nothing of the trainer's fast path.

// The shared modules have items that only the trainer or its tests use.
#![allow(dead_code)]

#[path = "../checkpoint.rs"]
mod checkpoint;
// The tests of `enclosure.rs` put a thread into the trainer's flushing arithmetic.
#[cfg(test)]
#[path = "../arith.rs"]
mod arith;
#[path = "../data.rs"]
mod data;
#[path = "../enclosure.rs"]
mod enclosure;
#[path = "../layout.rs"]
mod layout;
#[path = "../mt.rs"]
mod mt;
#[path = "../reference.rs"]
mod reference;

use std::path::Path;
use std::process::ExitCode;

use data::{ANSWER_POS, Equation, NUM0, P};
use enclosure::{Enclosure, Iv, Wide};
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

/// A row whose enclosure in intervals of doubles is wider than this is enclosed again in the
/// wider arithmetic. Both enclose the logits; this only chooses the cheaper one where it is
/// narrow, which it is at the checkpoints trained with weight decay.
const REFINE: f64 = 1e-3;

/// The width from which a row is refined, by the first argument: `REFINE` for `--enclose`, the
/// width of `--enclose=WIDTH`, and none for an argument that is not the flag.
fn refine_width(flag: &str) -> Result<Option<f64>, String> {
    match flag.strip_prefix("--enclose") {
        None => Ok(None),
        Some("") => Ok(Some(REFINE)),
        Some(rest) => match rest.strip_prefix('=').and_then(|width| width.parse::<f64>().ok()) {
            Some(width) if width >= 0.0 => Ok(Some(width)),
            _ => Err(format!("{flag} is not --enclose or --enclose=WIDTH with a width from 0")),
        },
    }
}

/// What the intervals of doubles gave before any row was refined: the largest width of one, the
/// rows whose answer they do not decide, and the rows computed again in the wider arithmetic.
#[derive(Clone, Copy, Default)]
struct Coarse {
    width: f64,
    undecided: usize,
    refined: usize,
}

/// Enclosures of the logits of the answer of every equation, by the pass of `enclosure.rs`, with
/// the rows whose intervals of doubles are wider than `refine` or do not decide the answer
/// computed again in the wider arithmetic.
fn enclosed_logits(
    dims: Dims,
    p: &[f64],
    equations: &[Equation],
    refine: f64,
    threads: usize,
) -> (Vec<Vec<Iv>>, Coarse) {
    let pe = enclosure::positions::<Iv>(dims, ANSWER_POS + 1);
    let wide_pe = enclosure::positions::<Wide>(dims, ANSWER_POS + 1);
    let mut rows = vec![Vec::new(); equations.len()];
    let chunk = equations.len().div_ceil(threads);
    let coarse = std::thread::scope(|scope| {
        let workers: Vec<_> = equations
            .chunks(chunk)
            .zip(rows.chunks_mut(chunk))
            .map(|(part, out)| {
                let (pe, wide_pe) = (&pe, &wide_pe);
                scope.spawn(move || {
                    let mut coarse = Coarse::default();
                    for (equation, row) in part.iter().zip(out) {
                        let tokens = equation.tokens();
                        *row = enclosure::logits_at(dims, p, pe, &tokens[..dims.ctx], ANSWER_POS);
                        let width = row.iter().map(|v| v.hi - v.lo).fold(0.0, f64::max);
                        let open = decided(row, NUM0 + equation.quo as usize).is_none();
                        coarse.width = coarse.width.max(width);
                        coarse.undecided += usize::from(open);
                        if width > refine || open {
                            let wide = enclosure::logits_at(dims, p, wide_pe, &tokens[..dims.ctx], ANSWER_POS);
                            let wide: Vec<Iv> = wide.iter().map(|v| v.iv()).collect();
                            // Both enclose the logits, so each pair has a real in common.
                            let met = wide.iter().zip(row.iter()).all(|(a, b)| a.lo <= b.hi && b.lo <= a.hi);
                            assert!(met, "the two enclosures of the row of {equation:?} have no real in common");
                            *row = wide;
                            coarse.refined += 1;
                        }
                    }
                    coarse
                })
            })
            .collect();
        workers.into_iter().map(|worker| worker.join().expect("a worker of the enclosures")).fold(
            Coarse::default(),
            |all, part| Coarse {
                width: all.width.max(part.width),
                undecided: all.undecided + part.undecided,
                refined: all.refined + part.refined,
            },
        )
    });
    (rows, coarse)
}

/// What an enclosure decides of the answer of a row: `Some(true)` if the lower end of its logit
/// is above the upper end of every other, so that it is the prediction; `Some(false)` if the
/// lower end of another logit is above its upper end, so that it is not; nothing otherwise.
fn decided(row: &[Iv], answer: usize) -> Option<bool> {
    let others = || row.iter().enumerate().filter(|(token, _)| *token != answer).map(|(_, v)| *v);
    if others().all(|v| v.hi < row[answer].lo) {
        Some(true)
    } else if others().any(|v| v.lo > row[answer].hi) {
        Some(false)
    } else {
        None
    }
}

/// A lower bound of the margin of `answer` in a row, from the ends of its enclosure: the lower
/// end of the answer less the largest upper end of another token, rounded down.
fn sure_margin(lower: &[f64], upper: &[f64], answer: usize) -> f64 {
    let rivals = upper.iter().enumerate().filter(|(token, _)| *token != answer).map(|(_, v)| *v);
    (lower[answer] - rivals.fold(f64::NEG_INFINITY, f64::max)).next_down()
}

/// The test of `enclosure_gives_certificate` on the ends of the enclosures of a row and of an
/// anchor's row: a lower bound `m` of the anchor's margin, a number `least` below every
/// difference of the two rows and a number `largest` above every one, and `m` less
/// `largest - least`, every step rounded against the claim. A positive value is the premise of
/// the certificate for the real rows.
fn sure_slack(own: (&[f64], &[f64]), anchor: (&[f64], &[f64]), answer: usize) -> f64 {
    let (mut least, mut largest) = (f64::INFINITY, f64::NEG_INFINITY);
    for token in 0..own.0.len() {
        least = least.min((own.0[token] - anchor.1[token]).next_down());
        largest = largest.max((own.1[token] - anchor.0[token]).next_up());
    }
    (sure_margin(anchor.0, anchor.1, answer) - (largest - least).next_up()).next_down()
}

/// The field `enclosed` of the head of the file.
fn enclosed_fields(
    dims: Dims,
    p: &[f64],
    equations: &[Equation],
    training: &[bool],
    refine: f64,
    threads: usize,
) -> String {
    let (rows, coarse) = enclosed_logits(dims, p, equations, refine, threads);
    let ends = |end: fn(&Iv) -> f64| rows.iter().map(|row| row.iter().map(end).collect()).collect::<Vec<Vec<f64>>>();
    let (lower, upper, middle) = (ends(|v| v.lo), ends(|v| v.hi), ends(|v| 0.5 * (v.lo + v.hi)));
    let answer = |k: usize| NUM0 + equations[k].quo as usize;
    let counts = |train: bool| {
        let of = |verdict: Option<bool>| {
            let same = |k: &usize| training[*k] == train && decided(&rows[*k], answer(*k)) == verdict;
            (0..equations.len()).filter(same).count()
        };
        format!("[{},{},{}]", of(Some(true)), of(Some(false)), of(None))
    };
    let margins = (0..equations.len()).map(|k| sure_margin(&lower[k], &upper[k], answer(k)));
    let width = rows.iter().flatten().map(|v| v.hi - v.lo).fold(0.0, f64::max);
    let [middle_nonzero, middle_zero] = orbits(NUM0, equations, &middle, training);
    let [lower_nonzero, lower_zero] = orbits(NUM0, equations, &lower, training);
    let [upper_nonzero, upper_zero] = orbits(NUM0, equations, &upper, training);
    let orbit = |middle: &Orbit, lower: &Orbit, upper: &Orbit, per: usize| {
        let row = |orbit: &Orbit, index: usize| orbit.rows[index * dims.vocab..(index + 1) * dims.vocab].to_vec();
        let ends = |index: usize| (row(lower, index), row(upper, index));
        let slack = |(own, anchor, _): &(usize, usize, f64)| {
            let (own, anchor) = (ends(*own), ends(*anchor));
            sure_slack((&own.0, &own.1), (&anchor.0, &anchor.1), middle.answer)
        };
        let slacks: Vec<f64> = nearest(middle, dims.vocab, per, threads).iter().map(slack).collect();
        format!(
            "{{\"heldout\":{},\"certified\":{},\"slack_min\":{}}}",
            slacks.len(),
            slacks.iter().filter(|s| **s > 0.0).count(),
            num(slacks.iter().cloned().fold(f64::INFINITY, f64::min))
        )
    };
    format!(
        "{{\"width\":{width:e},\"refined\":{},\"doubles\":{{\"width\":{:e},\"undecided\":{}}},\"training\":{},\
         \"heldout\":{},\"margin_lower_min\":{},\"nonzero\":{},\"zero\":{}}}",
        coarse.refined,
        coarse.width,
        coarse.undecided,
        counts(true),
        counts(false),
        num(margins.fold(f64::INFINITY, f64::min)),
        orbit(&middle_nonzero, &lower_nonzero, &upper_nonzero, 1),
        orbit(&middle_zero, &lower_zero, &upper_zero, P - 1)
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
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let refine = match args.first() {
        Some(flag) => refine_width(flag)?,
        None => None,
    };
    if refine.is_some() {
        args.remove(0);
    }
    let Some(dir) = args.first().map(Path::new) else {
        return Err("usage: quotient [--enclose[=WIDTH]] RUN_DIR [STEP...]".into());
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
        let enclosed = match refine {
            Some(width) => {
                format!(",\"enclosed\":{}", enclosed_fields(dims, &p, &equations, &training, width, threads))
            }
            None => String::new(),
        };
        println!(
            "{{\"run\":\"{}\",\"step\":{step},\"errors_training\":{},\"errors_heldout\":{},\"recorded_agree\":{},\
             \"nonzero\":{},\"zero\":{},\"other_gap\":{},\"residues_nonzero\":{},\"residues_zero\":{}{enclosed}}}",
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

    #[test]
    fn the_first_argument_gives_the_width_to_refine_from() {
        assert_eq!(refine_width("runs/adamw-seed0"), Ok(None));
        assert_eq!(refine_width("--enclose"), Ok(Some(REFINE)));
        assert_eq!(refine_width("--enclose=0"), Ok(Some(0.0)));
        assert_eq!(refine_width("--enclose=2.5e-4"), Ok(Some(2.5e-4)));
        for refused in ["--enclose=-1", "--enclose=NaN", "--enclose=x", "--enclosed", "--enclose="] {
            assert!(refine_width(refused).is_err(), "{refused}");
        }
    }

    /// The ends of enclosures decide an answer that is above the others for certain or below
    /// one for certain and nothing else, bound a margin from below, and pass the test of the
    /// certificate only where the exact comparison of every row inside them does.
    #[test]
    fn ends_of_enclosures_decide_answers_and_bound_the_slack() {
        let iv = |lo: f64, hi: f64| Iv { lo, hi };
        assert_eq!(decided(&[iv(0.0, 1.0), iv(1.5, 2.0), iv(-1.0, 1.25)], 1), Some(true));
        assert_eq!(decided(&[iv(0.0, 1.0), iv(1.5, 2.0), iv(2.5, 3.0)], 1), Some(false));
        assert_eq!(decided(&[iv(0.0, 1.0), iv(1.5, 2.0), iv(-1.0, 1.75)], 1), None);
        // Ends that touch decide nothing: the first of two equal logits is the prediction.
        assert_eq!(decided(&[iv(0.0, 1.5), iv(1.5, 2.0)], 1), None);
        // The margin of the token 1 is at least 1.5 - 1.25, and the bound is rounded down.
        let bound = sure_margin(&[0.0, 1.5, -1.0], &[1.0, 2.0, 1.25], 1);
        assert!(bound < 0.25 && bound > 0.25 - 1e-15);
        // An anchor (0, 2, 0) with the margin 2, and a row within 1/4 of it after the constant
        // 5: the differences are from 4.75 to 5.25, so the margin exceeds their range by 1.5.
        let anchor = [0.0, 2.0, 0.0];
        let slack = sure_slack((&[4.75, 6.75, 4.75], &[5.25, 7.25, 5.25]), (&anchor, &anchor), 1);
        assert!(slack < 1.5 && slack > 1.5 - 1e-14);
        // A row bent by 3, more than the margin, does not pass.
        let bent = [4.0, 9.0, 4.0];
        assert!(sure_slack((&bent, &bent), (&anchor, &anchor), 1) < 0.0);
        // An anchor known within 1/2 has a margin of at least 1 and differences 1 wider.
        let slack = sure_slack((&[4.75, 6.75, 4.75], &[5.25, 7.25, 5.25]), (&[-0.5, 1.5, -0.5], &[0.5, 2.5, 0.5]), 1);
        assert!(slack < -0.5 && slack > -0.5 - 1e-14);
        // On the table of the test above, with enclosures of no width, the test passes exactly
        // where the comparison in exact rationals does.
        let equations = data::equations();
        let training: Vec<bool> = (0..equations.len()).map(|i| i % 40 == 0).collect();
        let shape = |ratio: usize| ((ratio * ratio) % 7) as f64 / 7.0;
        let table = table(shape, 2.0, |e, class| match (e.num, e.den, class) {
            (4, 1, 9) => 0.75,
            (10, 2, 9) => 9.0,
            (0, 3, 2) => 9.0,
            _ => 0.0,
        });
        let [nonzero, zero] = orbits(NUM0, &equations, &table, &training);
        for (orbit, per) in [(&nonzero, 1), (&zero, P - 1)] {
            let row = |index: usize| &orbit.rows[index * VOCAB..(index + 1) * VOCAB];
            let pairs = nearest(orbit, VOCAB, per, 3);
            let passed = |pair: &&(usize, usize, f64)| {
                sure_slack((row(pair.0), row(pair.0)), (row(pair.1), row(pair.1)), orbit.answer) > 0.0
            };
            for pair in &pairs {
                let exact = certified_exactly(row(pair.1), row(pair.0), orbit.answer);
                assert_eq!(exact, Some(passed(&pair)), "row {} against {}", pair.0, pair.1);
            }
            assert_eq!(pairs.iter().filter(passed).count(), pairs.len() - 1);
        }
    }
}
