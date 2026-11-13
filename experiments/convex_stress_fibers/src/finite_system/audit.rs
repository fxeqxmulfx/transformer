//! Independent full-row replay and original-equation checks. The producer
//! updates suffixes only; this checker verifies the untouched zero prefixes
//! and applies each invertible operation to every coordinate, including RHS.
use super::*;
use serde_json::{Value, json};

fn satisfies(s: &System, x: &[u16], homogeneous: bool, products: &mut u64) -> bool {
    if x.len() != s.variables || x.iter().any(|&z| z >= s.prime) {
        return false;
    }
    s.rows.iter().zip(&s.right).all(|(a, &b)| {
        let mut value = 0u64;
        for (&a, &x) in a.iter().zip(x) {
            *products += 1;
            value += a as u64 * x as u64;
        }
        value % s.prime as u64 == if homogeneous { 0 } else { b as u64 }
    })
}
pub fn check(problem: &Problem, out: &Outcome) -> Result<Value> {
    let bound = problem.validate()?;
    let s = &problem.system;
    let (n, m, p) = (s.variables, s.rows.len(), s.prime as u32);
    if &out.problem != problem
        || out.buffer_bound_bytes != bound
        || out.operations.len() > problem.policy.row_operations
        || out.work.pivot_tests > n * m
        || out.work.inverse_tests > n * (p as usize - 1)
        || (2..s.prime).any(|d| s.prime.is_multiple_of(d))
    {
        return Err("finite-system audit input, prime or work bounds failed".into());
    }
    let mut rows: Vec<_> = s
        .rows
        .iter()
        .zip(&s.right)
        .map(|(a, &b)| {
            let mut r = a.clone();
            r.push(b);
            r
        })
        .collect();
    let (mut scalar_updates, mut full_updates) = (0u64, 0u64);
    for op in &out.operations {
        match *op {
            Operation::Swap { a, b } => {
                if a >= m || b >= m || a == b {
                    return Err("invalid certified row swap".into());
                }
                rows.swap(a, b);
            }
            Operation::Scale { row, start, factor } => {
                if row >= m
                    || start >= n
                    || factor == 0
                    || factor as u32 >= p
                    || rows[row][..start].iter().any(|&z| z != 0)
                    || rows[row][start] as u32 * factor as u32 % p != 1
                {
                    return Err("invalid invertible pivot normalization".into());
                }
                for x in &mut rows[row] {
                    *x = (*x as u32 * factor as u32 % p) as u16;
                }
                scalar_updates += (n + 1 - start) as u64;
                full_updates += (n + 1) as u64;
            }
            Operation::Subtract {
                target,
                source,
                start,
                factor,
            } => {
                if target >= m
                    || source >= m
                    || target == source
                    || start >= n
                    || factor == 0
                    || factor as u32 >= p
                    || rows[source][start] != 1
                    || rows[target][start] != factor
                    || rows[source][..start].iter().any(|&z| z != 0)
                {
                    return Err("invalid elimination row operation".into());
                }
                let source_row = rows[source].clone();
                for (x, y) in rows[target].iter_mut().zip(source_row) {
                    *x = ((*x as u32 + p - factor as u32 * y as u32 % p) % p) as u16;
                }
                scalar_updates += (n + 1 - start) as u64;
                full_updates += (n + 1) as u64;
            }
        }
    }
    if rows != out.augmented
        || scalar_updates != out.work.scalar_updates
        || out.operations.len() != out.work.row_operations
        || scalar_updates > problem.policy.scalar_updates
    {
        return Err("full-row state or charged scalar work differs".into());
    }
    let mut products = 0;
    let capped = matches!(
        out.reason.as_str(),
        "row_operation_cap" | "scalar_update_cap"
    );
    if capped {
        if out.space.is_some() || out.contradiction.is_some() || !out.free.is_empty() {
            return Err("partial elimination claimed a solution certificate".into());
        }
        if (out.reason == "row_operation_cap"
            && out.work.row_operations != problem.policy.row_operations)
            || (out.reason == "scalar_update_cap"
                && problem.policy.scalar_updates - scalar_updates >= (n + 1) as u64)
        {
            return Err("declared elimination cap was not reached".into());
        }
    } else {
        let rank = out.pivots.len();
        if rank > m
            || rank > n
            || out.pivots.iter().any(|&j| j >= n)
            || out.pivots.windows(2).any(|w| w[0] >= w[1])
        {
            return Err("invalid reduced-row pivot list".into());
        }
        for (i, row) in rows.iter().enumerate() {
            if i < rank {
                let pivot = out.pivots[i];
                if row[pivot] != 1
                    || row[..pivot].iter().any(|&a| a != 0)
                    || rows
                        .iter()
                        .enumerate()
                        .any(|(j, r)| j != i && r[pivot] != 0)
                {
                    return Err("matrix is not reduced at its claimed pivots".into());
                }
            } else if row[..n].iter().any(|&a| a != 0) {
                return Err("an unreported coefficient pivot remains".into());
            }
        }
        let free: Vec<_> = (0..n).filter(|j| !out.pivots.contains(j)).collect();
        let contradiction = rows
            .iter()
            .position(|r| r[..n].iter().all(|&a| a == 0) && r[n] != 0);
        if free != out.free || contradiction != out.contradiction {
            return Err("free columns or contradiction differ".into());
        }
        if contradiction.is_some() {
            if out.reason != "inconsistent" || out.space.is_some() {
                return Err("wrong inconsistency outcome".into());
            }
        } else if free.len() > problem.policy.nullity {
            if out.reason != "nullity_cap" || out.space.is_some() {
                return Err("unrecorded nullity cap".into());
            }
        } else {
            let space = out
                .space
                .as_ref()
                .ok_or("missing complete affine solution space")?;
            if out.reason != "complete"
                || space.directions.len() != free.len()
                || !satisfies(s, &space.particular, false, &mut products)
                || free.iter().any(|&j| space.particular[j] != 0)
            {
                return Err(
                    "particular solution does not satisfy the original affine system".into(),
                );
            }
            for (i, direction) in space.directions.iter().enumerate() {
                if !satisfies(s, direction, true, &mut products)
                    || free
                        .iter()
                        .enumerate()
                        .any(|(j, &k)| direction[k] != u16::from(i == j))
                {
                    return Err(
                        "nullspace direction or independent free-coordinate basis failed".into(),
                    );
                }
            }
        }
    }
    Ok(
        json!({"row_operations_checked":out.operations.len(),"full_row_entry_updates":full_updates,
        "charged_suffix_updates":scalar_updates,"original_equation_products":products,
        "equivalent_system_checked":true,"affine_space_checked":out.reason=="complete",
        "infeasibility_checked":out.reason=="inconsistent","capped":capped||out.reason=="nullity_cap"}),
    )
}
