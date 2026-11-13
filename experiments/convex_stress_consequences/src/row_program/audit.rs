use super::*;

/// Replay every complete row with i128. This checks algebraic work and actual
/// echelon entries. `replay` additionally binds all search/control counters.
pub fn check(input: &Problem, out: &Outcome) -> Result<()> {
    let bound = input.validate()?;
    if out.input != *input
        || out.buffer_preflight_bytes != bound
        || out.operations.len() > input.policy.row_operations
        || out.matrix.len() != input.rows.len()
        || out.matrix.iter().any(|row| row.len() != input.variables)
        || out.pivots.len() > input.variables.min(input.rows.len())
        || out.visited_columns > input.variables
    {
        return Err("row program original input, extent or preflight differs".into());
    }
    let q = input.modulus as i128;
    let mut actual: Vec<Vec<i128>> = input
        .rows
        .iter()
        .map(|row| row.iter().map(|&x| x as i128).collect())
        .collect();
    let (mut updates, mut copies) = (0_u64, 0_u64);
    for op in &out.operations {
        match *op {
            Op::Swap { i, j } => {
                if i >= actual.len() || j >= actual.len() || i == j {
                    return Err("invalid row-program swap".into());
                }
                let old = actual[i].clone();
                actual[i] = actual[j].clone();
                actual[j] = old;
                copies += 2 * input.variables as u64;
            }
            Op::Scale { i, factor } => {
                if i >= actual.len()
                    || factor == 0
                    || factor >= input.modulus
                    || !(1..q).any(|v| v * factor as i128 % q == 1)
                {
                    return Err("row-program normalization is not a checked unit".into());
                }
                for x in &mut actual[i] {
                    *x = (*x * factor as i128).rem_euclid(q);
                }
                updates += input.variables as u64;
            }
            Op::Add { i, j, factor } => {
                if i >= actual.len()
                    || j >= actual.len()
                    || i == j
                    || factor == 0
                    || factor >= input.modulus
                {
                    return Err("invalid row-program addition".into());
                }
                actual[i] = actual[i]
                    .iter()
                    .zip(&actual[j])
                    .map(|(&a, &b)| (a + factor as i128 * b).rem_euclid(q))
                    .collect();
                updates += input.variables as u64;
            }
        }
    }
    let matrix: Vec<Vec<u16>> = actual
        .iter()
        .map(|row| row.iter().map(|&x| x as u16).collect())
        .collect();
    let w = &out.work;
    let p = &input.policy;
    if matrix != out.matrix
        || w.scalar_updates != updates
        || w.swapped_coefficients != copies
        || w.row_operations != out.operations.len()
        || updates > p.scalar_updates
        || w.pivot_tests > p.pivot_tests
        || w.combination_tests > p.combination_tests
        || w.normalization_tests > p.normalization_tests
        || w.ce_gradients != 0
    {
        return Err("actual full-row program state or work differs".into());
    }
    let mut previous_column = None;
    for (index, pivot) in out.pivots.iter().enumerate() {
        let (row, col, d) = (pivot.row, pivot.column, pivot.divisor);
        if row != index
            || col >= out.visited_columns
            || d == 0
            || d >= input.modulus
            || !input.modulus.is_multiple_of(d)
            || previous_column.is_some_and(|old| old >= col)
            || matrix[row][col] != d
            || matrix[row][..col].iter().any(|&x| x != 0)
            || matrix[row + 1..].iter().any(|r| r[col] != 0)
            || (d == 1 && matrix[..row].iter().any(|r| r[col] != 0))
        {
            return Err("claimed divisor pivot differs from actual echelon coefficients".into());
        }
        previous_column = Some(col);
    }
    if out.reason == "columns_scanned" && out.visited_columns != input.variables {
        return Err("row program claims unvisited columns were scanned".into());
    }
    if out.reason == "rows_exhausted" && out.pivots.len() != input.rows.len() {
        return Err("row program claims unconsumed rows were exhausted".into());
    }
    Ok(())
}
