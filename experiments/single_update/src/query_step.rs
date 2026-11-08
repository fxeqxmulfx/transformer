//! JC1-JC3: signed joint increments, mixed/rounding bounds and actual query CE.
use crate::{algebra::linear, domain::Result, gradient::energy, inputs::require};
use serde_json::{Value, json};

pub struct Problem<'a> {
    pub old_features: &'a [f64],
    pub new_features: &'a [f64],
    pub old_embedding: &'a [f64],
    pub new_embedding: &'a [f64],
    pub increment: &'a [f64],
    pub rounding: &'a [f64],
    pub old_logits: &'a [f64],
    pub new_logits: &'a [f64],
    pub labels: &'a [usize],
    pub width: usize,
    pub classes: usize,
}
#[derive(Clone, Copy)]
pub struct Interval {
    pub margin: [f64; 2],
    pub loss: [f64; 2],
}
pub struct Row {
    pub target: usize,
    pub worst: [usize; 2],
    pub correct: [bool; 2],
    pub margin: [f64; 2],
    pub loss: [f64; 2],
    pub intervals: [Interval; 2],
    pub identity_error: f64,
    pub bound_error: f64,
    pub weighted_error: f64,
    pub mean_abs_terms: [f64; 6],
}
impl Row {
    pub fn json(&self) -> Value {
        json!([
            self.target,
            self.worst[0],
            self.worst[1],
            self.correct[0],
            self.correct[1],
            self.margin[0],
            self.margin[1],
            self.loss[0],
            self.loss[1],
            self.intervals[0].margin[0],
            self.intervals[0].margin[1],
            self.intervals[0].loss[0],
            self.intervals[0].loss[1],
            self.intervals[1].margin[0],
            self.intervals[1].margin[1],
            self.intervals[1].loss[0],
            self.intervals[1].loss[1]
        ])
    }
}
pub const ROW_FIELDS: [&str; 17] = [
    "target",
    "old_worst_competitor",
    "new_worst_competitor",
    "old_correct",
    "new_correct",
    "old_margin",
    "new_margin",
    "old_loss",
    "new_loss",
    "joint_margin_lower",
    "joint_margin_upper",
    "joint_loss_lower",
    "joint_loss_upper",
    "separate_margin_lower",
    "separate_margin_upper",
    "separate_loss_lower",
    "separate_loss_upper",
];

pub fn tolerance(values: &[f64]) -> f64 {
    1e-8 * (1. + values.iter().map(|v| v.abs()).fold(0., f64::max))
}
fn logsumexp(x: &[f64]) -> f64 {
    let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    m + x.iter().map(|v| (v - m).exp()).sum::<f64>().ln()
}
fn contrast_norms(e: &[f64], classes: usize, width: usize) -> Vec<f64> {
    (0..classes)
        .flat_map(|t| {
            (0..classes).map(move |c| {
                (0..width)
                    .map(|k| (e[c * width + k] - e[t * width + k]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
        })
        .collect()
}
fn interval(lo: &[f64], hi: &[f64], target: usize) -> Interval {
    let worst = |x: &[f64]| {
        x.iter()
            .enumerate()
            .filter(|(c, _)| *c != target)
            .map(|(_, v)| *v)
            .fold(f64::NEG_INFINITY, f64::max)
    };
    Interval {
        margin: [-worst(hi), -worst(lo)],
        loss: [logsumexp(lo), logsumexp(hi)],
    }
}

pub fn analyze(p: Problem<'_>) -> Result<Vec<Row>> {
    let (n, w, c) = (p.labels.len(), p.width, p.classes);
    require(
        n > 0 && w > 0 && c > 1 && p.labels.iter().all(|&t| t < c),
        "invalid query dimensions",
    )?;
    for (array, len) in [
        (p.old_features, n * w),
        (p.new_features, n * w),
        (p.old_embedding, c * w),
        (p.new_embedding, c * w),
        (p.increment, c * w),
        (p.rounding, c * w),
        (p.old_logits, n * c),
        (p.new_logits, n * c),
    ] {
        require(
            array.len() == len && array.iter().all(|x| x.is_finite()),
            "invalid query input",
        )?;
    }
    for k in 0..c * w {
        require(
            (p.new_embedding[k] - p.old_embedding[k] - p.increment[k] - p.rounding[k]).abs()
                <= tolerance(&[p.new_embedding[k], p.old_embedding[k]]),
            "increment/rounding does not reconstruct parameters",
        )?;
    }
    let dx: Vec<_> = p
        .new_features
        .iter()
        .zip(p.old_features)
        .map(|(a, b)| a - b)
        .collect();
    let parameter = linear(p.old_features, n, w, p.increment, c);
    let feature = linear(&dx, n, w, p.old_embedding, c);
    let mixed = linear(&dx, n, w, p.increment, c);
    let rounding = linear(p.new_features, n, w, p.rounding, c);
    let old_math = linear(p.old_features, n, w, p.old_embedding, c);
    let new_math = linear(p.new_features, n, w, p.new_embedding, c);
    let wn = contrast_norms(p.old_embedding, c, w);
    let pn = contrast_norms(p.increment, c, w);
    let rn = contrast_norms(p.rounding, c, w);
    let mut rows = Vec::with_capacity(n);
    for (q, &target) in p.labels.iter().enumerate() {
        let at = q * c;
        let gap = |x: &[f64], j: usize| x[at + j] - x[at + target];
        let xnorm = energy(&p.old_features[q * w..(q + 1) * w]).sqrt();
        let newnorm = energy(&p.new_features[q * w..(q + 1) * w]).sqrt();
        let dxnorm = energy(&dx[q * w..(q + 1) * w]).sqrt();
        let old: Vec<_> = (0..c).map(|j| gap(p.old_logits, j)).collect();
        let new: Vec<_> = (0..c).map(|j| gap(p.new_logits, j)).collect();
        let mut lower = vec![0.; c];
        let mut upper = vec![0.; c];
        let mut separate_lo = vec![0.; c];
        let mut separate_hi = vec![0.; c];
        let mut relative_upper = vec![0.; c];
        let (mut identity_error, mut bound_error) = (0f64, 0f64);
        let mut mean_abs_terms = [0.; 6];
        for j in 0..c {
            let a = gap(&parameter, j);
            let b = gap(&feature, j);
            let joint = a + b;
            let mix = gap(&mixed, j);
            let round = gap(&rounding, j);
            let execution = (new[j] - gap(&new_math, j)) - (old[j] - gap(&old_math, j));
            let radius =
                pn[target * c + j] * dxnorm + rn[target * c + j] * newnorm + execution.abs();
            let separated = pn[target * c + j] * xnorm + wn[target * c + j] * dxnorm + radius;
            lower[j] = old[j] + joint - radius;
            upper[j] = old[j] + joint + radius;
            separate_lo[j] = old[j] - separated;
            separate_hi[j] = old[j] + separated;
            relative_upper[j] = joint + radius;
            let scale = 1.
                + [
                    old[j], new[j], joint, mix, round, execution, radius, separated,
                ]
                .iter()
                .map(|x| x.abs())
                .fold(0., f64::max);
            identity_error = identity_error
                .max((new[j] - old[j] - joint - mix - round - execution).abs() / scale);
            bound_error = bound_error.max(
                (lower[j] - new[j])
                    .max(new[j] - upper[j])
                    .max(separate_lo[j] - new[j])
                    .max(new[j] - separate_hi[j])
                    / scale,
            );
            if j != target {
                for (i, v) in [
                    joint.abs(),
                    a.abs() + b.abs(),
                    mix.abs(),
                    radius,
                    round.abs(),
                    execution.abs(),
                ]
                .iter()
                .enumerate()
                {
                    mean_abs_terms[i] += v / (c - 1) as f64;
                }
            }
        }
        let loss = [logsumexp(&old), logsumexp(&new)];
        let joint = interval(&lower, &upper, target);
        let separate = interval(&separate_lo, &separate_hi, target);
        let weighted = (0..c)
            .map(|j| (old[j] - loss[0]).exp() * (new[j] - old[j]).exp())
            .sum::<f64>();
        let weighted_upper = (0..c)
            .map(|j| (old[j] - loss[0]).exp() * relative_upper[j].exp())
            .sum::<f64>();
        let weighted_error = ((loss[1] - loss[0]) - weighted.ln())
            .abs()
            .max(((joint.loss[1] - loss[0]) - weighted_upper.ln()).abs())
            / (1. + loss[0] + loss[1] + joint.loss[1]);
        let worst = |x: &[f64]| {
            (0..c)
                .filter(|&j| j != target)
                .reduce(|a, b| if x[b] > x[a] { b } else { a })
                .unwrap()
        };
        let best = |x: &[f64]| (1..c).fold(0, |a, b| if x[b] > x[a] { b } else { a });
        let competitors = [worst(&old), worst(&new)];
        let margin = [-old[competitors[0]], -new[competitors[1]]];
        let intervals = [joint, separate];
        require(
            loss.iter()
                .chain(&margin)
                .chain(
                    intervals
                        .iter()
                        .flat_map(|b| b.margin.iter().chain(&b.loss)),
                )
                .all(|x| x.is_finite())
                && weighted_error.is_finite(),
            "nonfinite query bounds",
        )?;
        require(
            identity_error <= 1e-8 && bound_error <= 1e-8 && weighted_error <= 1e-8,
            &format!(
                "query {q}: identity={identity_error}, bound={bound_error}, weighted={weighted_error}"
            ),
        )?;
        for b in intervals {
            require(
                b.margin[0] <= margin[1] + tolerance(&b.margin)
                    && margin[1] <= b.margin[1] + tolerance(&b.margin)
                    && b.loss[0] <= loss[1] + tolerance(&b.loss)
                    && loss[1] <= b.loss[1] + tolerance(&b.loss),
                "query margin/loss interval failed",
            )?;
        }
        rows.push(Row {
            target,
            worst: competitors,
            correct: [best(&old) == target, best(&new) == target],
            margin,
            loss,
            intervals,
            identity_error,
            bound_error,
            weighted_error,
            mean_abs_terms,
        });
    }
    Ok(rows)
}

#[cfg(test)]
pub fn fixture(
    e: &[f64],
    p: &[f64],
    rho: &[f64],
    x: &[f64],
    xn: &[f64],
    labels: &[usize],
    width: usize,
) -> Vec<Row> {
    let en: Vec<_> = e
        .iter()
        .zip(p)
        .zip(rho)
        .map(|((e, p), r)| e + p + r)
        .collect();
    // Independent scalar oracle, separate from the BLAS operations under test.
    let score = |x: &[f64], e: &[f64]| {
        x.chunks_exact(width)
            .flat_map(|r| {
                e.chunks_exact(width)
                    .map(move |w| r.iter().zip(w).map(|(a, b)| a * b).sum::<f64>())
            })
            .collect::<Vec<_>>()
    };
    let old = score(x, e);
    let new = score(xn, &en);
    analyze(Problem {
        old_features: x,
        new_features: xn,
        old_embedding: e,
        new_embedding: &en,
        increment: p,
        rounding: rho,
        old_logits: &old,
        new_logits: &new,
        labels,
        width,
        classes: e.len() / width,
    })
    .unwrap()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancelling_first_order_terms_still_have_a_mixed_remainder() {
        let r = fixture(&[0., 2.], &[0., 1.], &[0., 0.], &[2.], &[1.], &[1], 1).remove(0);
        assert_eq!(r.margin, [4., 3.]);
        assert_eq!(r.mean_abs_terms, [0., 4., 1., 1., 0., 0.]);
        assert_eq!(r.intervals[0].margin, [3., 5.]);
        assert_eq!(r.intervals[1].margin, [-1., 9.]);
    }
    #[test]
    fn nonzero_rounding_stays_in_the_remainder() {
        let r = fixture(&[0., 2.], &[0., 1.], &[0., 0.01], &[2.], &[3.], &[1], 1).remove(0);
        assert!((r.margin[1] - 9.03).abs() < 1e-12);
        assert!((r.mean_abs_terms[4] - 0.03).abs() < 1e-12);
        assert!(r.intervals[0].margin[0] > 6.9);
        assert!(r.intervals[0].loss[1] < r.loss[0]);
    }
    #[test]
    fn new_worst_competitor_can_destroy_a_correct_decision() {
        let r = fixture(
            &[0., -1., -2.],
            &[0., -2., 2.5],
            &[0.; 3],
            &[1.],
            &[1.],
            &[0],
            1,
        )
        .remove(0);
        assert_eq!(r.worst, [1, 2]);
        assert_eq!(r.correct, [true, false]);
        assert_eq!(r.intervals[0].margin, [-0.5, -0.5]);
    }
    #[test]
    fn mean_margin_growth_does_not_control_actual_mean_ce() {
        let l8 = 8f64.ln();
        let l256 = 256f64.ln();
        let l2 = 2f64.ln();
        let rows = fixture(
            &[0., 0., l8, l8],
            &[0., 0., l256 - l8, l2 - l8],
            &[0.; 4],
            &[1., 0., 0., 1.],
            &[1., 0., 0., 1.],
            &[1, 1],
            2,
        );
        assert!(rows.iter().all(|r| r.correct == [true, true]));
        assert!(rows.iter().map(|r| r.margin[1] - r.margin[0]).sum::<f64>() > 0.);
        assert!(rows.iter().map(|r| r.loss[1] - r.loss[0]).sum::<f64>() > 0.);
        assert!(rows[1].intervals[0].loss[0] > rows[1].loss[0]);
    }
}
