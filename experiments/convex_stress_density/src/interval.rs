//! CM22 (Lean a4388412): an explicit convex interval domain, distinct from
//! numerical acceptance tolerances. A bounded primal-dual proximal iteration
//! minimizes I_density(X)-<C,X>+I_box(A X). With extrapolated Xbar:
//! y'=v-sigma*clip(v/sigma,lower,upper), v=y+sigma*A Xbar;
//! X'=proj_density(X+tau*(C-A^T y')); Xbar'=2X'-X.
//! tau=sigma=0.9/max(1,||A||_F), so their product times ||A||_op^2 is <1.
//! No convergence theorem or exact/interval-arithmetic certificate is claimed.
//! A single real objective may leave a mixture, especially at a real-score tie.
use crate::{C, Result, matrix, memory_bound};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Options {
    pub model_tolerance: f64,
    pub iterations: usize,
    pub feasibility_tolerance: f64,
    pub gap_tolerance: f64,
    pub step_tolerance: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            model_tolerance: 0.1,
            iterations: 16_384,
            feasibility_tolerance: 1e-8,
            gap_tolerance: 1e-8,
            step_tolerance: 1e-8,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Input {
    pub n: usize,
    pub rank: usize,
    pub features: Vec<C>,
    pub rows: [usize; 2],
    pub options: Options,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub iteration: usize,
    pub objective: f64,
    pub dual_bound: f64,
    pub dual_gap: f64,
    pub interval_violation: f64,
    pub primal_step: f64,
    pub dual_step: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    pub format: u32,
    pub input: Input,
    pub buffer_bound_bytes: u64,
    pub accepted: bool,
    pub reason: String,
    pub matrix: Vec<C>,
    pub multipliers: Vec<f64>,
    pub extrapolated: Vec<f64>,
    pub step_size: f64,
    pub trace: Vec<Step>,
    pub purity_error: f64,
    /// Actual kernel column; a candidate even when mixed, never a certified mode.
    pub kernel_column: Option<Vec<C>>,
}

pub fn validate(input: &Input) -> Result<u64> {
    let bound = memory_bound(input.n, input.rank)?;
    let o = input.options;
    if input.features.len() != input.n * input.rank
        || input.rows[0] >= input.n
        || input.rows[1] >= input.n
        || input.rows[0] == input.rows[1]
        || input
            .features
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite() || z.norm() > 1e4)
        || !o.model_tolerance.is_finite()
        || !(0.0..=1.0).contains(&o.model_tolerance)
        || !(1..=16_384).contains(&o.iterations)
        || [o.feasibility_tolerance, o.gap_tolerance, o.step_tolerance]
            .iter()
            .any(|x| !x.is_finite() || !(1e-12..=1e-3).contains(x))
    {
        return Err("invalid bounded interval input".into());
    }
    Ok(bound)
}

/// Row normalization rescales both interval endpoints and the dual variables.
pub(crate) struct BoxRows {
    pub rows: Vec<Vec<f64>>,
    pub scales: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub step: f64,
}
impl BoxRows {
    pub fn new(input: &Input) -> Self {
        let (n, r) = (input.n, input.rank);
        let mut rows: Vec<_> = (0..n)
            .map(|x| matrix::objective(&input.features, n, r, x, x, false))
            .collect();
        let scales: Vec<_> = rows
            .iter()
            .map(|a| matrix::dot(a, a).sqrt().max(1e-14))
            .collect();
        for (row, scale) in rows.iter_mut().zip(&scales) {
            for x in row {
                *x /= scale;
            }
        }
        let norm = rows.iter().map(|a| matrix::dot(a, a)).sum::<f64>().sqrt();
        Self {
            rows,
            lower: scales
                .iter()
                .map(|s| (1. - input.options.model_tolerance) / s)
                .collect(),
            upper: scales
                .iter()
                .map(|s| (1. + input.options.model_tolerance) / s)
                .collect(),
            scales,
            step: 0.9 / norm.max(1.),
        }
    }
    pub fn transpose(&self, y: &[f64]) -> Vec<f64> {
        let mut out = vec![0.; self.rows[0].len()];
        for (row, y) in self.rows.iter().zip(y) {
            for (out, a) in out.iter_mut().zip(row) {
                *out += a * y;
            }
        }
        out
    }
    pub fn support(&self, y: &[f64]) -> f64 {
        y.iter()
            .enumerate()
            .map(|(i, y)| {
                y * if *y >= 0. {
                    self.upper[i]
                } else {
                    self.lower[i]
                }
            })
            .sum()
    }
    pub fn violation(&self, x: &[f64]) -> f64 {
        self.rows
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let z = matrix::dot(a, x);
                (self.lower[i] - z).max(z - self.upper[i]).max(0.) * self.scales[i]
            })
            .fold(0., f64::max)
    }
}

pub fn solve(input: Input) -> Result<Outcome> {
    let bound = validate(&input)?;
    let (n, r) = (input.n, input.rank);
    let o = input.options;
    let a = BoxRows::new(&input);
    let c = matrix::objective(&input.features, n, r, input.rows[0], input.rows[1], false);
    let mut x = vec![0.; r * r];
    x[..r].fill(1. / r as f64);
    let mut out = Outcome {
        format: 1,
        input,
        buffer_bound_bytes: bound,
        accepted: false,
        reason: "iteration_cap".into(),
        matrix: matrix::unpack(&x, r),
        multipliers: vec![0.; n],
        extrapolated: x.clone(),
        step_size: a.step,
        trace: vec![],
        purity_error: 1. - 1. / r as f64,
        kernel_column: None,
    };
    for iteration in 1..=o.iterations {
        let y: Vec<_> = a
            .rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let v = out.multipliers[i] + a.step * matrix::dot(row, &out.extrapolated);
                v - a.step * (v / a.step).clamp(a.lower[i], a.upper[i])
            })
            .collect();
        let aty = a.transpose(&y);
        let t: Vec<_> = x
            .iter()
            .zip(&c)
            .zip(&aty)
            .map(|((x, c), aty)| x + a.step * (c - aty))
            .collect();
        let density = matrix::density(&matrix::unpack(&t, r), r)?;
        let next = matrix::pack(&density, r);
        let slack: Vec<_> = c.iter().zip(&aty).map(|(c, a)| c - a).collect();
        let upper = matrix::eigen(&matrix::unpack(&slack, r), r)?.values[r - 1] + a.support(&y);
        let objective = matrix::dot(&c, &next);
        let step = Step {
            iteration,
            objective,
            dual_bound: upper,
            dual_gap: upper - objective,
            interval_violation: a.violation(&next),
            primal_step: matrix::distance(&x, &next) / a.step,
            dual_step: matrix::distance(&out.multipliers, &y) / a.step,
        };
        if [
            step.objective,
            step.dual_bound,
            step.dual_gap,
            step.interval_violation,
            step.primal_step,
            step.dual_step,
        ]
        .iter()
        .any(|x| !x.is_finite())
        {
            return Err("nonfinite interval iteration diagnostics".into());
        }
        out.accepted = step.interval_violation <= o.feasibility_tolerance
            && step.dual_gap.abs() <= o.gap_tolerance
            && step.primal_step <= o.step_tolerance
            && step.dual_step <= o.step_tolerance;
        out.extrapolated = next.iter().zip(&x).map(|(n, x)| 2. * n - x).collect();
        x = next;
        out.matrix = density;
        out.multipliers = y;
        out.trace.push(step);
        if out.accepted {
            out.reason = "numerical_interval_optimum".into();
            break;
        }
    }
    out.purity_error = (1. - matrix::eigen(&out.matrix, r)?.values[r - 1]).abs();
    if out.accepted {
        out.kernel_column = Some(
            (0..n)
                .map(|x| {
                    matrix::reading(&out.input.features, n, r, &out.matrix, x, out.input.rows[0])
                })
                .collect(),
        );
    }
    Ok(out)
}

pub fn replay(original: &Input, saved: &Outcome) -> Result<()> {
    if saved.format != 1 || &saved.input != original || solve(original.clone())? != *saved {
        return Err("interval density input or replay mismatch".into());
    }
    Ok(())
}
