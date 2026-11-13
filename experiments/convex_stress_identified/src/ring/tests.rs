use super::*;

fn problem(q: u16, rows: Vec<Vec<u16>>) -> Problem {
    Problem {
        modulus: q,
        variables: rows[0].len(),
        rows,
        policy: Policy::default(),
    }
}
fn kernel_count(p: &Problem) -> usize {
    assert_eq!(p.variables, 2);
    let q = p.modulus as u64;
    (0..q)
        .flat_map(|a| (0..q).map(move |b| [a, b]))
        .filter(|x| {
            p.rows
                .iter()
                .all(|r| r.iter().zip(x).map(|(&a, &b)| a as u64 * b).sum::<u64>() % q == 0)
        })
        .count()
}

#[test]
fn exhaustive_2274_composite_and_prime_matrices() {
    let mut matrices = 0;
    for q in 2_u16..=6 {
        for code in 0..(q as usize).pow(4) {
            let mut rest = code;
            let mut rows = vec![vec![0; 2]; 2];
            for row in &mut rows {
                for a in row {
                    *a = (rest % q as usize) as u16;
                    rest /= q as usize;
                }
            }
            let p = problem(q, rows);
            let out = run(p.clone()).unwrap();
            check(&p, &out).unwrap();
            replay(&p, &out).unwrap();
            assert_eq!(out.complete, kernel_count(&p) == 1, "q={q}, code={code}");
            matrices += 1;
        }
    }
    assert_eq!(matrices, 2274);
}

#[test]
fn actual_nonunit_pivots_match_the_lean_six_program() {
    let p = problem(6, vec![vec![2, 1], vec![3, 1]]);
    let out = run(p.clone()).unwrap();
    assert!(out.complete);
    assert_eq!(
        out.operations,
        vec![
            Op::Add {
                i: 0,
                j: 1,
                factor: 1
            },
            Op::Scale { i: 0, factor: 5 },
            Op::Add {
                i: 1,
                j: 0,
                factor: 3
            },
            Op::Add {
                i: 0,
                j: 1,
                factor: 2
            }
        ]
    );
    assert_eq!(out.matrix, vec![vec![1, 0], vec![0, 1]]);
    check(&p, &out).unwrap();
    let json = serde_json::to_vec(&out).unwrap();
    let restored: Outcome = serde_json::from_slice(&json).unwrap();
    assert_eq!(out, restored);
    replay(&p, &restored).unwrap();
}

#[test]
fn every_work_cap_and_refusal_preserves_replayable_state() {
    let base = problem(6, vec![vec![2, 1], vec![3, 1]]);
    for (slot, reason) in [
        "scalar_update_cap",
        "row_operation_cap",
        "pivot_test_cap",
        "combination_test_cap",
        "inverse_test_cap",
    ]
    .iter()
    .enumerate()
    {
        let mut p = base.clone();
        match slot {
            0 => p.policy.scalar_updates = 0,
            1 => p.policy.row_operations = 0,
            2 => p.policy.pivot_tests = 0,
            3 => p.policy.combination_tests = 0,
            _ => p.policy.inverse_tests = 0,
        }
        let out = run(p.clone()).unwrap();
        assert!(!out.complete);
        assert_eq!(out.reason, *reason);
        check(&p, &out).unwrap();
        replay(&p, &out).unwrap();
    }
    for p in [
        problem(6, vec![vec![2, 0], vec![0, 3]]),
        problem(4, vec![vec![1, 0], vec![0, 0]]),
        problem(6, vec![vec![1, 0]]),
    ] {
        let out = run(p.clone()).unwrap();
        assert!(!out.complete);
        check(&p, &out).unwrap();
        replay(&p, &out).unwrap();
    }
    // A cap after a completed first pivot keeps both the partial state and prefix.
    let mut p = base;
    p.policy.row_operations = 3;
    let out = run(p.clone()).unwrap();
    assert_eq!(out.pivots, vec![0]);
    assert!(!out.complete);
    check(&p, &out).unwrap();
    replay(&p, &out).unwrap();
}

#[test]
fn independent_rows_and_full_control_replay_reject_tampering() {
    let p = problem(6, vec![vec![2, 1], vec![3, 1]]);
    let original = run(p.clone()).unwrap();
    for slot in 0..11 {
        let mut out = original.clone();
        match slot {
            0 => out.input.rows[0][0] = 1,
            1 => out.buffer_bound_bytes += 1,
            2 => {
                out.operations[0] = Op::Add {
                    i: 0,
                    j: 1,
                    factor: 2,
                }
            }
            3 => out.operations[1] = Op::Scale { i: 0, factor: 2 },
            4 => out.operations[0] = Op::Swap { i: 0, j: 99 },
            5 => out.matrix[0][1] = 1,
            6 => out.work.scalar_updates += 1,
            7 => out.work.row_operations += 1,
            8 => out.pivots.reverse(),
            9 => out.reason = "invented_complete".into(),
            _ => out.work.ce_gradients = 1,
        }
        assert!(check(&p, &out).is_err(), "independent mutation {slot}");
        assert!(replay(&p, &out).is_err());
    }
    for slot in 0..4 {
        let mut out = original.clone();
        match slot {
            0 => out.work.pivot_tests += 1,
            1 => out.work.combination_tests += 1,
            2 => out.work.inverse_tests += 1,
            _ => out.work.gcd_steps += 1,
        }
        assert!(replay(&p, &out).is_err(), "control mutation {slot}");
    }
    let p = problem(6, vec![vec![2, 0], vec![0, 3]]);
    let mut out = run(p.clone()).unwrap();
    out.complete = true;
    out.reason = "coordinate_rows".into();
    assert!(check(&p, &out).is_err());
}

#[test]
fn larger_rectangular_domains_and_invalid_inputs() {
    for q in [8, 12, 32, 96, 127, 128, 192, 257] {
        let n = 24;
        let mut rows = vec![vec![0; n]; n + 3];
        for (i, row) in rows.iter_mut().enumerate().take(n) {
            for (j, a) in row.iter_mut().enumerate().skip(i) {
                *a = if i == j {
                    1
                } else {
                    ((i + 2 * j) % q as usize) as u16
                };
            }
        }
        rows.swap(0, n - 1);
        let p = problem(q, rows);
        let out = run(p.clone()).unwrap();
        assert!(out.complete);
        check(&p, &out).unwrap();
        replay(&p, &out).unwrap();
    }
    let original = problem(6, vec![vec![2, 1], vec![3, 1]]);
    for slot in 0..6 {
        let mut p = original.clone();
        match slot {
            0 => p.modulus = 1,
            1 => p.rows[0][0] = 6,
            2 => p.variables = 0,
            3 => p.rows.clear(),
            4 => p.rows[0].pop().map(|_| ()).unwrap(),
            _ => p.policy.row_operations = 2_000_001,
        }
        assert!(run(p).is_err());
    }
}
