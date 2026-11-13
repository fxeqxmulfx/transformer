use super::*;

fn apply(out: &mut Outcome, op: Operation) -> bool {
    let width = out.problem.system.variables + 1;
    let updates = match op {
        Operation::Swap { .. } => 0,
        Operation::Scale { start, .. } | Operation::Subtract { start, .. } => {
            (width - start) as u64
        }
    };
    if out.work.row_operations == out.problem.policy.row_operations {
        out.reason = "row_operation_cap".into();
        return false;
    }
    if out.work.scalar_updates + updates > out.problem.policy.scalar_updates {
        out.reason = "scalar_update_cap".into();
        return false;
    }
    let p = out.problem.system.prime as u32;
    match op {
        Operation::Swap { a, b } => out.augmented.swap(a, b),
        Operation::Scale { row, start, factor } => {
            for x in &mut out.augmented[row][start..] {
                *x = (*x as u32 * factor as u32 % p) as u16;
            }
        }
        Operation::Subtract {
            target,
            source,
            start,
            factor,
        } => {
            for j in start..width {
                out.augmented[target][j] = ((out.augmented[target][j] as u32 + p
                    - factor as u32 * out.augmented[source][j] as u32 % p)
                    % p) as u16;
            }
        }
    }
    out.work.row_operations += 1;
    out.work.scalar_updates += updates;
    out.operations.push(op);
    true
}
pub fn solve(problem: Problem) -> Result<Outcome> {
    let buffer_bound_bytes = problem.validate()?;
    let s = &problem.system;
    let (m, n, p) = (s.rows.len(), s.variables, s.prime);
    let augmented = s
        .rows
        .iter()
        .zip(&s.right)
        .map(|(a, &b)| {
            let mut row = a.clone();
            row.push(b);
            row
        })
        .collect();
    let mut out = Outcome {
        problem,
        augmented,
        operations: vec![],
        pivots: vec![],
        free: vec![],
        contradiction: None,
        space: None,
        reason: "complete".into(),
        work: Work::default(),
        buffer_bound_bytes,
    };
    let mut active = 0;
    for column in 0..n {
        let mut selected = None;
        for row in active..m {
            out.work.pivot_tests += 1;
            if out.augmented[row][column] != 0 {
                selected = Some(row);
                break;
            }
        }
        let Some(row) = selected else {
            continue;
        };
        if row != active && !apply(&mut out, Operation::Swap { a: row, b: active }) {
            return Ok(out);
        }
        let entry = out.augmented[active][column];
        // Bounded exhaustive inverse search is independently checked; no
        // unchecked primality-dependent exponent shortcut is needed.
        let inverse = (1..p)
            .find(|&z| {
                out.work.inverse_tests += 1;
                z as u32 * entry as u32 % p as u32 == 1
            })
            .ok_or("prime-field inverse absent")?;
        if !apply(
            &mut out,
            Operation::Scale {
                row: active,
                start: column,
                factor: inverse,
            },
        ) {
            return Ok(out);
        }
        for target in 0..m {
            if target == active {
                continue;
            }
            let factor = out.augmented[target][column];
            if factor != 0
                && !apply(
                    &mut out,
                    Operation::Subtract {
                        target,
                        source: active,
                        start: column,
                        factor,
                    },
                )
            {
                return Ok(out);
            }
        }
        out.pivots.push(column);
        active += 1;
        if active == m {
            break;
        }
    }
    out.free = (0..n).filter(|j| !out.pivots.contains(j)).collect();
    out.contradiction = out
        .augmented
        .iter()
        .position(|r| r[..n].iter().all(|&x| x == 0) && r[n] != 0);
    if out.contradiction.is_some() {
        out.reason = "inconsistent".into();
        return Ok(out);
    }
    if out.free.len() > out.problem.policy.nullity {
        out.reason = "nullity_cap".into();
        return Ok(out);
    }
    let mut particular = vec![0; n];
    for (row, &pivot) in out.pivots.iter().enumerate() {
        particular[pivot] = out.augmented[row][n];
    }
    let mut directions = vec![];
    for &free in &out.free {
        let mut direction = vec![0; n];
        direction[free] = 1;
        for (row, &pivot) in out.pivots.iter().enumerate() {
            direction[pivot] = (p - out.augmented[row][free]) % p;
        }
        directions.push(direction);
    }
    out.space = Some(Space {
        particular,
        directions,
    });
    Ok(out)
}
