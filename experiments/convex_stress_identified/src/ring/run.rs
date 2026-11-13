use super::*;

fn gcd(mut a: u16, mut b: u16, work: &mut Work) -> u16 {
    while b != 0 {
        work.gcd_steps += 1;
        (a, b) = (b, a % b);
    }
    a
}
fn charge(counter: &mut u64, cap: u64, reason: &mut String, name: &str) -> bool {
    if *counter == cap {
        *reason = name.into();
        false
    } else {
        *counter += 1;
        true
    }
}
fn apply(out: &mut Outcome, op: Op) -> bool {
    let updates = match op {
        Op::Swap { .. } => 0,
        _ => out.input.variables as u64,
    };
    if out.work.row_operations == out.input.policy.row_operations {
        out.reason = "row_operation_cap".into();
        return false;
    }
    if updates > out.input.policy.scalar_updates - out.work.scalar_updates {
        out.reason = "scalar_update_cap".into();
        return false;
    }
    let q = out.input.modulus as u32;
    match op {
        Op::Swap { i, j } => {
            out.matrix.swap(i, j);
            out.work.swapped_coefficients += 2 * out.input.variables as u64;
        }
        Op::Scale { i, factor } => {
            for x in &mut out.matrix[i] {
                *x = (*x as u32 * factor as u32 % q) as u16;
            }
        }
        Op::Add { i, j, factor } => {
            for k in 0..out.input.variables {
                out.matrix[i][k] = ((out.matrix[i][k] as u32
                    + factor as u32 * out.matrix[j][k] as u32)
                    % q) as u16;
            }
        }
    }
    out.operations.push(op);
    out.work.row_operations += 1;
    out.work.scalar_updates += updates;
    true
}

fn pivot(out: &mut Outcome, col: usize) -> bool {
    let q = out.input.modulus;
    let (mut first, mut unit) = (None, None);
    for row in col..out.matrix.len() {
        if !charge(
            &mut out.work.pivot_tests,
            out.input.policy.pivot_tests,
            &mut out.reason,
            "pivot_test_cap",
        ) {
            return false;
        }
        let x = out.matrix[row][col];
        if x != 0 && first.is_none() {
            first = Some(row);
        }
        if gcd(x, q, &mut out.work) == 1 {
            unit = Some(row);
            break;
        }
    }
    let Some(row) = unit.or(first) else {
        out.reason = "zero_remaining_column".into();
        return false;
    };
    if row != col && !apply(out, Op::Swap { i: col, j: row }) {
        return false;
    }
    if unit.is_some() {
        return true;
    }
    let mut d = gcd(out.matrix[col][col], q, &mut out.work);
    for row in col + 1..out.matrix.len() {
        let wanted = gcd(d, out.matrix[row][col], &mut out.work);
        if wanted == d {
            continue;
        }
        let mut found = false;
        for factor in 1..q {
            if !charge(
                &mut out.work.combination_tests,
                out.input.policy.combination_tests,
                &mut out.reason,
                "combination_test_cap",
            ) {
                return false;
            }
            let x = ((out.matrix[col][col] as u32 + factor as u32 * out.matrix[row][col] as u32)
                % q as u32) as u16;
            if gcd(x, q, &mut out.work) == wanted {
                if !apply(
                    out,
                    Op::Add {
                        i: col,
                        j: row,
                        factor,
                    },
                ) {
                    return false;
                }
                d = wanted;
                found = true;
                break;
            }
        }
        if !found {
            out.reason = "unit_combination_refused".into();
            return false;
        }
        if d == 1 {
            return true;
        }
    }
    out.reason = "nonunit_remaining_column".into();
    false
}

pub fn run(input: Problem) -> Result<Outcome> {
    let bound = input.validate()?;
    let mut out = Outcome {
        matrix: input.rows.clone(),
        input,
        buffer_bound_bytes: bound,
        operations: vec![],
        pivots: vec![],
        complete: false,
        reason: "started".into(),
        work: Work::default(),
    };
    if out.matrix.len() < out.input.variables {
        out.reason = "insufficient_rows".into();
        return Ok(out);
    }
    let q = out.input.modulus;
    for col in 0..out.input.variables {
        if !pivot(&mut out, col) {
            return Ok(out);
        }
        let mut inverse = None;
        for factor in 1..q {
            if !charge(
                &mut out.work.inverse_tests,
                out.input.policy.inverse_tests,
                &mut out.reason,
                "inverse_test_cap",
            ) {
                return Ok(out);
            }
            if factor as u32 * out.matrix[col][col] as u32 % q as u32 == 1 {
                inverse = Some(factor);
                break;
            }
        }
        let factor = inverse.ok_or("unit pivot has no checked inverse")?;
        if factor != 1 && !apply(&mut out, Op::Scale { i: col, factor }) {
            return Ok(out);
        }
        for row in 0..out.matrix.len() {
            let x = out.matrix[row][col];
            if row != col
                && x != 0
                && !apply(
                    &mut out,
                    Op::Add {
                        i: row,
                        j: col,
                        factor: q - x,
                    },
                )
            {
                return Ok(out);
            }
        }
        out.pivots.push(col);
    }
    out.complete = true;
    out.reason = "coordinate_rows".into();
    Ok(out)
}
