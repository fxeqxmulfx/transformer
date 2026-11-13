//! Independent raw kernel/PSD/interval/support-box dual audit for CM22's
//! numerical prototype. Operators use polarization, not the solver's packing.
use crate::{
    C, Result, audit,
    interval::{self, Input, Outcome},
    matrix,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 * (1. + a.abs() + b.abs())
}
pub fn endpoint(original: &Input, saved: &Outcome) -> Result<Value> {
    let bound = interval::validate(original)?;
    let (n, r) = (original.n, original.rank);
    let o = original.options;
    if saved.format != 1
        || &saved.input != original
        || saved.buffer_bound_bytes != bound
        || saved.trace.is_empty()
        || saved.trace.len() > o.iterations
        || saved.multipliers.len() != n
        || saved.extrapolated.len() != r * r
        || saved
            .multipliers
            .iter()
            .chain(&saved.extrapolated)
            .any(|x| !x.is_finite())
    {
        return Err("invalid interval outcome envelope".into());
    }
    let b = &original.features;
    let e = matrix::eigen(&saved.matrix, r)?;
    let trace: C = (0..r).map(|i| saved.matrix[i + r * i]).sum();
    if e.values[0] < -1e-10 || (trace - 1.).norm() > 1e-10 {
        return Err("raw interval density PSD or trace violation".into());
    }
    let purity = (1. - e.values[r - 1]).abs();
    let [x, y] = original.rows;
    let mut slack = audit::operator(b, n, r, x, y, false);
    let mut support = 0.;
    let mut violation = 0f64;
    let mut norm_sq = 0.;
    let mut imaginary = 0f64;
    for (i, &multiplier) in saved.multipliers.iter().enumerate() {
        let op = audit::operator(b, n, r, i, i, false);
        let sq: f64 = op.iter().map(|z| z.norm_sqr()).sum();
        let scale = sq.sqrt().max(1e-14);
        norm_sq += sq / scale.powi(2);
        let reading = matrix::reading(b, n, r, &saved.matrix, i, i);
        imaginary = imaginary.max(reading.im.abs());
        violation = violation.max((reading.re - 1.).abs() - o.model_tolerance);
        support += multiplier
            * (if multiplier >= 0. {
                1. + o.model_tolerance
            } else {
                1. - o.model_tolerance
            })
            / scale;
        for (s, a) in slack.iter_mut().zip(op) {
            *s -= a * (multiplier / scale);
        }
    }
    let upper = matrix::eigen(&slack, r)?.values[r - 1] + support;
    let objective = matrix::reading(b, n, r, &saved.matrix, x, y).re;
    let last = saved.trace.last().unwrap();
    if !close(last.objective, objective)
        || !close(last.dual_bound, upper)
        || !close(last.dual_gap, upper - objective)
        || !close(last.interval_violation, violation)
        || !close(saved.purity_error, purity)
        || !close(saved.step_size, 0.9 / norm_sq.sqrt().max(1.))
        || last.iteration != saved.trace.len()
        || imaginary > 1e-10
    {
        return Err("raw interval readings or support-box bound mismatch".into());
    }
    if saved.accepted {
        if violation > o.feasibility_tolerance * 1.001
            || (upper - objective).abs() > o.gap_tolerance * 1.001
        {
            return Err("accepted interval outcome violates numerical thresholds".into());
        }
        let column = saved
            .kernel_column
            .as_ref()
            .ok_or("missing interval kernel column")?;
        if column.len() != n
            || column
                .iter()
                .any(|z| !z.re.is_finite() || !z.im.is_finite())
            || column
                .iter()
                .enumerate()
                .any(|(i, z)| (*z - matrix::reading(b, n, r, &saved.matrix, i, x)).norm() > 1e-12)
        {
            return Err("interval kernel column mismatch".into());
        }
    } else if saved.kernel_column.is_some() {
        return Err("refused interval outcome has a candidate column".into());
    }
    Ok(
        json!({"raw_entry_audit":true,"dual_operator_method":"polarization",
        "exact_arithmetic_certificate":false,"interval_arithmetic_certificate":false,
        "certified_mode":false,"accepted":saved.accepted,"iterations":saved.trace.len(),
        "objective":objective,"dual_bound":upper,"dual_gap":upper-objective,
        "model_tolerance":o.model_tolerance,"interval_violation":violation,
        "minimum_eigenvalue":e.values[0],"purity_error":purity,"trace_error":(trace-1.).norm()}),
    )
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Artifact {
    pub outcome: Outcome,
    pub raw_audit: Value,
}
pub fn run(input: Input) -> Result<Artifact> {
    let outcome = interval::solve(input)?;
    let raw_audit = endpoint(&outcome.input, &outcome)?;
    Ok(Artifact { outcome, raw_audit })
}
pub fn replay(original: &Input, saved: &Artifact) -> Result<()> {
    interval::replay(original, &saved.outcome)?;
    if endpoint(original, &saved.outcome)? != saved.raw_audit {
        return Err("interval raw audit replay mismatch".into());
    }
    Ok(())
}
