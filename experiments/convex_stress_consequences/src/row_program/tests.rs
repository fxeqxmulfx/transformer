use super::*;

fn problem(q: u16, rows: Vec<Vec<u16>>) -> Problem {
    Problem {
        modulus: q,
        variables: rows[0].len(),
        rows,
        policy: Policy::default(),
    }
}
fn checked(r: &Problem) -> Outcome {
    let out = run(r.clone()).unwrap();
    check(r, &out).unwrap();
    replay(r, &out).unwrap();
    out
}
fn span(rows: &[Vec<u16>], q: u16) -> std::collections::BTreeSet<Vec<u16>> {
    (0..(q as usize).pow(rows.len() as u32))
        .map(|mut code| {
            let mut value = vec![0_u32; rows[0].len()];
            for row in rows {
                let factor = (code % q as usize) as u32;
                code /= q as usize;
                for (sum, &x) in value.iter_mut().zip(row) {
                    *sum += factor * x as u32;
                }
            }
            value.iter().map(|x| (x % q as u32) as u16).collect()
        })
        .collect()
}
#[test]
fn all2274_small_composite_matrices_preserve_actual_row_spans() {
    let mut count = 0;
    for q in 2_u16..=6 {
        for code in 0..(q as usize).pow(4) {
            let mut x = code;
            let entries: Vec<_> = (0..4)
                .map(|_| {
                    let z = (x % q as usize) as u16;
                    x /= q as usize;
                    z
                })
                .collect();
            let r = problem(q, vec![entries[..2].to_vec(), entries[2..].to_vec()]);
            let out = checked(&r);
            assert_eq!(span(&r.rows, q), span(&out.matrix, q));
            assert!(matches!(
                out.reason.as_str(),
                "columns_scanned" | "rows_exhausted"
            ));
            count += 1;
        }
    }
    assert_eq!(count, 2274);
}
#[test]
fn divisor_pivots_free_columns_and_nonunit_combinations_are_actual_rows() {
    let r = problem(12, vec![vec![0, 8, 1], vec![0, 4, 3], vec![0, 0, 6]]);
    let out = checked(&r);
    assert_eq!(
        out.pivots[0],
        Pivot {
            row: 0,
            column: 1,
            divisor: 4
        }
    );
    assert!(
        out.operations
            .iter()
            .any(|op| matches!(op, Op::Scale { .. }))
    );
    assert_eq!(span(&r.rows, 12), span(&out.matrix, 12));
    let out = checked(&problem(6, vec![vec![2, 1], vec![3, 1]]));
    assert_eq!(out.pivots[0].divisor, 1);
    assert!(out.work.combination_tests > 0);
    let out = checked(&problem(257, vec![vec![0, 0, 13], vec![0, 0, 0]]));
    assert_eq!(out.pivots[0].column, 2);
    assert_eq!(out.pivots[0].divisor, 1);
}
#[test]
fn each_budget_retains_only_actual_executed_rows() {
    let base = problem(6, vec![vec![2, 1], vec![3, 1]]);
    for (slot, reason) in [
        "scalar_update_cap",
        "row_operation_cap",
        "pivot_test_cap",
        "combination_test_cap",
        "normalization_test_cap",
    ]
    .iter()
    .enumerate()
    {
        let mut r = base.clone();
        match slot {
            0 => r.policy.scalar_updates = 0,
            1 => r.policy.row_operations = 0,
            2 => r.policy.pivot_tests = 0,
            3 => r.policy.combination_tests = 0,
            _ => r.policy.normalization_tests = 0,
        }
        let out = checked(&r);
        assert_eq!(&out.reason, reason);
        assert_eq!(span(&r.rows, 6), span(&out.matrix, 6));
    }
}
#[test]
fn altered_programs_pivots_matrices_and_actual_work_are_rejected() {
    let r = problem(6, vec![vec![2, 1], vec![3, 1]]);
    let out = checked(&r);
    for slot in 0..8 {
        let mut bad = out.clone();
        match slot {
            0 => {
                bad.operations[0] = Op::Add {
                    i: 0,
                    j: 1,
                    factor: 2,
                }
            }
            1 => bad.matrix[0][0] = 0,
            2 => bad.pivots[0].divisor = 2,
            3 => bad.pivots[0].row = usize::MAX,
            4 => bad.work.scalar_updates -= 1,
            5 => bad.work.ce_gradients = 1,
            6 => bad.visited_columns = 0,
            _ => {
                bad.operations[0] = Op::Swap {
                    i: usize::MAX,
                    j: 0,
                }
            }
        }
        assert!(check(&r, &bad).is_err(), "mutation {slot}");
        assert!(replay(&r, &bad).is_err());
    }
    let mut bad = out;
    bad.work.gcd_steps += 1;
    assert!(replay(&r, &bad).is_err());
}
