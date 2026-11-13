//! Independent entrywise audit of saved density endpoints and SDP dual bounds.
//! The objective/constraint operators are reconstructed by polarization from
//! raw kernel readings, without using the solver's packed affine rows.
use crate::{
    C, Result, matrix,
    solver::{Extraction, input_check},
};
use serde_json::{Value, json};

pub(crate) fn operator(b: &[C], n: usize, r: usize, x: usize, y: usize, imag: bool) -> Vec<C> {
    let value = |m: &[C]| {
        let z = matrix::reading(b, n, r, m, x, y);
        if imag { z.im } else { z.re }
    };
    let mut out = vec![C::default(); r * r];
    for j in 0..r {
        let mut m = vec![C::default(); r * r];
        m[j + r * j] = C::new(1., 0.);
        out[j + r * j] = C::new(value(&m), 0.);
        for i in 0..j {
            m.fill(C::default());
            m[i + r * j] = C::new(1., 0.);
            m[j + r * i] = C::new(1., 0.);
            let re = value(&m) / 2.;
            m[i + r * j] = C::new(0., 1.);
            m[j + r * i] = C::new(0., -1.);
            out[i + r * j] = C::new(re, value(&m) / 2.);
            out[j + r * i] = out[i + r * j].conj();
        }
    }
    out
}
fn norm(m: &[C]) -> f64 {
    m.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}
fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 * (1. + a.abs() + b.abs())
}

pub fn endpoint(b: &[C], saved: &Extraction) -> Result<Value> {
    let (n, r) = (saved.n, saved.rank);
    let bound = input_check(b, n, r, saved.rows, saved.options)?;
    if saved.buffer_bound_bytes != bound || saved.stages.len() > 2 {
        return Err("invalid extraction envelope".into());
    }
    let (x, y) = (saved.rows[0], saved.rows[1]);
    let mut stages = vec![];
    let mut final_diagonal = 0.;
    let mut final_purity = 0.;
    for (index, s) in saved.stages.iter().enumerate() {
        let e = matrix::eigen(&s.matrix, r)?;
        let trace: C = (0..r).map(|i| s.matrix[i + r * i]).sum();
        if e.values[0] < -1e-10
            || (trace - 1.).norm() > 1e-10
            || s.multipliers.len() != n + index
            || s.trace.len() > saved.options.iterations
        {
            return Err("raw density shape, PSD, trace or budget violation".into());
        }
        let readings: Vec<_> = (0..n)
            .map(|i| matrix::reading(b, n, r, &s.matrix, i, i))
            .collect();
        let diagonal = readings.iter().map(|z| (*z - 1.).norm()).fold(0., f64::max);
        final_diagonal = diagonal;
        final_purity = (1. - e.values[r - 1]).abs();
        if let Some(last) = s.trace.last() {
            let c = operator(b, n, r, x, y, index == 1);
            let mut slack = c.clone();
            let mut constant = 0.;
            let mut residual = 0f64;
            for (a, &multiplier) in s.multipliers.iter().enumerate() {
                let (row, target, reading) = if a < n {
                    (operator(b, n, r, a, a, false), 1., readings[a].re)
                } else {
                    let target = matrix::reading(b, n, r, &saved.stages[0].matrix, x, y).re;
                    (
                        operator(b, n, r, x, y, false),
                        target,
                        matrix::reading(b, n, r, &s.matrix, x, y).re,
                    )
                };
                let scale = norm(&row).max(1e-14);
                residual = residual.max((reading - target).abs() / scale);
                constant += multiplier * target / scale;
                for (z, aij) in slack.iter_mut().zip(row) {
                    *z -= aij * (multiplier / scale);
                }
            }
            let upper = matrix::eigen(&slack, r)?.values[r - 1] + constant;
            let obs = matrix::reading(b, n, r, &s.matrix, x, y);
            let primal = if index == 0 { obs.re } else { obs.im };
            if !close(primal, last.objective)
                || !close(upper, last.dual_bound)
                || !close(upper - primal, last.dual_gap)
                || !close(residual, last.affine_residual)
                || last.iteration != s.trace.len()
            {
                return Err("raw reading/dual bound differs from saved stage diagnostics".into());
            }
            if s.accepted
                && (residual > saved.options.feasibility_tolerance * 1.001
                    || (upper - primal).abs() > saved.options.gap_tolerance * 1.001)
            {
                return Err("accepted stage fails independent numerical tolerances".into());
            }
            stages.push(json!({"stage":index,"objective":primal,"dual_bound":upper,
                "dual_gap":upper-primal,"normalized_affine_residual":residual,
                "raw_diagonal_residual":diagonal,"trace_error":(trace-1.).norm(),
                "minimum_eigenvalue":e.values[0],"maximum_eigenvalue":e.values[r-1],
                "iterations":s.trace.len(),"accepted":s.accepted}));
        } else if s.accepted {
            return Err("accepted empty optimization stage".into());
        }
    }
    if saved.accepted {
        if saved.stages.len() != 2
            || saved.stages.iter().any(|s| !s.accepted)
            || final_diagonal > saved.options.feasibility_tolerance * 1.001
            || final_purity > saved.options.purity_tolerance * 1.001
            || !saved.purity_error.is_some_and(|p| close(p, final_purity))
            || !saved
                .diagonal_residual
                .is_some_and(|d| close(d, final_diagonal))
        {
            return Err("invalid accepted extraction".into());
        }
        let phase = saved
            .phase_column
            .as_ref()
            .ok_or("missing accepted phase column")?;
        if phase.len() != n
            || phase.iter().any(|z| !z.re.is_finite() || !z.im.is_finite())
            || (0..n).any(|i| {
                (phase[i] - matrix::reading(b, n, r, &saved.stages[1].matrix, i, x)).norm() > 1e-12
            })
        {
            return Err("saved phase column differs from raw kernel".into());
        }
    } else if saved.phase_column.is_some() {
        return Err("refused extraction contains a phase column".into());
    }
    Ok(
        json!({"raw_entry_audit":true,"dual_operator_method":"polarization",
        "exact_arithmetic_certificate":false,"interval_certificate":false,
        "accepted":saved.accepted,"stages":stages}),
    )
}
