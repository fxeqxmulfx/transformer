use super::*;

/// Independent full-row i128 evaluation; it never calls producer arithmetic.
/// Acceptance proves the original anchored kernel is zero. A refusal claims
/// no completeness theorem; deterministic replay checks its detailed reason.
pub fn check(input: &Problem, out: &Outcome) -> Result<()> {
    let bound = input.validate()?;
    if &out.input != input || out.buffer_bound_bytes != bound {
        return Err("ring certificate input or storage forecast differs".into());
    }
    if out.operations.len() > input.policy.row_operations
        || out.matrix.len() != input.rows.len()
        || out.matrix.iter().any(|r| r.len() != input.variables)
        || out.pivots.len() > input.variables
    {
        return Err("ring saved certificate exceeds its shape or operation cap".into());
    }
    let mut a: Vec<Vec<i128>> = input
        .rows
        .iter()
        .map(|r| r.iter().map(|&x| x as i128).collect())
        .collect();
    let q = input.modulus as i128;
    let (mut updates, mut copied) = (0_u64, 0_u64);
    for op in &out.operations {
        match *op {
            Op::Swap { i, j } => {
                if i >= a.len() || j >= a.len() || i == j {
                    return Err("invalid ring swap indices".into());
                }
                let old = a[i].clone();
                a[i] = a[j].clone();
                a[j] = old;
                copied += 2 * input.variables as u64;
            }
            Op::Scale { i, factor } => {
                if i >= a.len()
                    || factor == 0
                    || factor >= input.modulus
                    || !(1..q).any(|v| v * factor as i128 % q == 1)
                {
                    return Err("ring scaling does not use a checked unit".into());
                }
                a[i] = a[i]
                    .iter()
                    .map(|&v| (factor as i128 * v).rem_euclid(q))
                    .collect();
                updates += input.variables as u64;
            }
            Op::Add { i, j, factor } => {
                if i >= a.len() || j >= a.len() || i == j || factor == 0 || factor >= input.modulus
                {
                    return Err("invalid actual ring row addition".into());
                }
                let old = a[i].clone();
                a[i] = old
                    .iter()
                    .zip(&a[j])
                    .map(|(&v, &w)| (v + factor as i128 * w).rem_euclid(q))
                    .collect();
                updates += input.variables as u64;
            }
        }
    }
    let actual: Vec<Vec<u16>> = a
        .iter()
        .map(|r| r.iter().map(|&v| v as u16).collect())
        .collect();
    let w = &out.work;
    let p = &input.policy;
    if actual != out.matrix
        || w.row_operations != out.operations.len()
        || w.scalar_updates != updates
        || w.swapped_coefficients != copied
        || w.scalar_updates > p.scalar_updates
        || w.row_operations > p.row_operations
        || w.pivot_tests > p.pivot_tests
        || w.combination_tests > p.combination_tests
        || w.inverse_tests > p.inverse_tests
        || w.ce_gradients != 0
        || out.pivots != (0..out.pivots.len()).collect::<Vec<_>>()
        || out.pivots.len() > input.variables
    {
        return Err("ring full-row state, work or pivot binding differs".into());
    }
    for (col, &row) in out.pivots.iter().enumerate() {
        for (k, &value) in actual[row].iter().enumerate() {
            // Later free columns may survive until the remaining elimination.
            if k < out.pivots.len() && value != u16::from(k == col) {
                return Err("completed ring pivot columns are not coordinate columns".into());
            }
        }
    }
    if out.complete {
        if out.reason != "coordinate_rows" || out.pivots.len() != input.variables {
            return Err("partial ring replay claimed a zero kernel".into());
        }
        for (col, &row) in out.pivots.iter().enumerate() {
            if actual[row]
                .iter()
                .enumerate()
                .any(|(k, &v)| v != u16::from(col == k))
            {
                return Err("ring certificate lacks an actual coordinate row".into());
            }
        }
    } else if out.reason == "coordinate_rows" || out.pivots.len() == input.variables {
        return Err("ring completion status contradicts its saved state".into());
    }
    Ok(())
}
