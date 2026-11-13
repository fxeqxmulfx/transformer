use super::*;

fn gcd(mut a: u16, mut b: u16, work: &mut Work) -> u16 {
    while b != 0 {
        work.gcd_steps += 1;
        (a, b) = (b, a % b);
    }
    a
}
fn charge(counter: &mut u64, limit: u64, reason: &mut String, name: &str) -> bool {
    if *counter == limit {
        *reason = name.into();
        false
    } else {
        *counter += 1;
        true
    }
}
fn apply(out: &mut Outcome, op: Op) -> bool {
    let updates = if matches!(op, Op::Swap { .. }) {
        0
    } else {
        out.input.variables as u64
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

// Accumulate the actual column ideal. Row addition searches are bounded and
// explicitly retained; they never assert that an unfound combination exists.
fn combine(out: &mut Outcome, row: usize, col: usize) -> Option<u16> {
    let q = out.input.modulus;
    let mut d = gcd(out.matrix[row][col], q, &mut out.work);
    for source in row + 1..out.matrix.len() {
        let wanted = gcd(d, out.matrix[source][col], &mut out.work);
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
                return None;
            }
            let x = ((out.matrix[row][col] as u32 + factor as u32 * out.matrix[source][col] as u32)
                % q as u32) as u16;
            if gcd(x, q, &mut out.work) == wanted {
                if !apply(
                    out,
                    Op::Add {
                        i: row,
                        j: source,
                        factor,
                    },
                ) {
                    return None;
                }
                found = true;
                d = wanted;
                break;
            }
        }
        if !found {
            out.reason = "combination_refused".into();
            return None;
        }
    }
    Some(d)
}

pub fn run(input: Problem) -> Result<Outcome> {
    let bound = input.validate()?;
    let mut out = Outcome {
        matrix: input.rows.clone(),
        input,
        buffer_preflight_bytes: bound,
        operations: vec![],
        pivots: vec![],
        visited_columns: 0,
        reason: "started".into(),
        work: Work::default(),
    };
    let q = out.input.modulus;
    for col in 0..out.input.variables {
        let row = out.pivots.len();
        if row == out.matrix.len() {
            out.reason = "rows_exhausted".into();
            return Ok(out);
        }
        out.visited_columns = col + 1;
        let mut first = None;
        for source in row..out.matrix.len() {
            if !charge(
                &mut out.work.pivot_tests,
                out.input.policy.pivot_tests,
                &mut out.reason,
                "pivot_test_cap",
            ) {
                return Ok(out);
            }
            if out.matrix[source][col] != 0 {
                first = Some(source);
                break;
            }
        }
        let Some(source) = first else { continue };
        if source != row && !apply(&mut out, Op::Swap { i: row, j: source }) {
            return Ok(out);
        }
        let Some(divisor) = combine(&mut out, row, col) else {
            return Ok(out);
        };
        let mut unit = None;
        for factor in 1..q {
            if !charge(
                &mut out.work.normalization_tests,
                out.input.policy.normalization_tests,
                &mut out.reason,
                "normalization_test_cap",
            ) {
                return Ok(out);
            }
            if (factor as u32 * out.matrix[row][col] as u32 % q as u32) as u16 == divisor
                && gcd(factor, q, &mut out.work) == 1
            {
                unit = Some(factor);
                break;
            }
        }
        let Some(factor) = unit else {
            out.reason = "normalization_refused".into();
            return Ok(out);
        };
        if factor != 1 && !apply(&mut out, Op::Scale { i: row, factor }) {
            return Ok(out);
        }
        // A unit pivot can be cleared above as well as below, leaving short
        // query representations. For nonunits, keep the earlier rows intact:
        // arbitrary division there could lose necessary torsion combinations.
        for target in 0..out.matrix.len() {
            if target == row || (target < row && divisor != 1) {
                continue;
            }
            let x = out.matrix[target][col];
            if !x.is_multiple_of(divisor) {
                return Err("column ideal did not divide an actual remaining entry".into());
            }
            if x != 0
                && !apply(
                    &mut out,
                    Op::Add {
                        i: target,
                        j: row,
                        factor: q - x / divisor,
                    },
                )
            {
                return Ok(out);
            }
        }
        out.pivots.push(Pivot {
            row,
            column: col,
            divisor,
        });
    }
    out.reason = "columns_scanned".into();
    Ok(out)
}
