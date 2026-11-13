use super::*;
use std::collections::BTreeSet;

fn vectors(p: u16, n: usize) -> Vec<Vec<u16>> {
    (0..(p as usize).pow(n as u32))
        .map(|mut k| {
            (0..n)
                .map(|_| {
                    let z = (k % p as usize) as u16;
                    k /= p as usize;
                    z
                })
                .collect()
        })
        .collect()
}
fn actual_solutions(s: &System) -> BTreeSet<Vec<u16>> {
    vectors(s.prime, s.variables)
        .into_iter()
        .filter(|x| {
            s.rows.iter().zip(&s.right).all(|(a, &b)| {
                a.iter()
                    .zip(x)
                    .map(|(&a, &x)| a as u64 * x as u64)
                    .sum::<u64>()
                    % s.prime as u64
                    == b as u64
            })
        })
        .collect()
}
fn represented(out: &Outcome) -> BTreeSet<Vec<u16>> {
    let Some(space) = &out.space else {
        return BTreeSet::new();
    };
    vectors(out.problem.system.prime, space.directions.len())
        .into_iter()
        .map(|weights| {
            (0..space.particular.len())
                .map(|j| {
                    (space.particular[j] as u64
                        + weights
                            .iter()
                            .zip(&space.directions)
                            .map(|(&a, d)| a as u64 * d[j] as u64)
                            .sum::<u64>())
                    .rem_euclid(out.problem.system.prime as u64) as u16
                })
                .collect()
        })
        .collect()
}
fn fixture() -> Problem {
    Problem {
        system: System {
            prime: 5,
            variables: 3,
            rows: vec![vec![0, 2, 1], vec![1, 1, 1], vec![2, 0, 1]],
            right: vec![3, 4, 0],
        },
        policy: Policy::default(),
    }
}

#[test]
fn every_small_binary_and_ternary_affine_system_matches_exhaustive_solutions() {
    // 793 different matrices/right sides. The oracle enumerates every possible
    // vector and directly tests original equations; it does no elimination.
    for p in [2u16, 3] {
        for entries in vectors(p, 6) {
            let problem = Problem {
                system: System {
                    prime: p,
                    variables: 2,
                    rows: vec![entries[..2].to_vec(), entries[2..4].to_vec()],
                    right: entries[4..].to_vec(),
                },
                policy: Policy::default(),
            };
            let out = solve(problem.clone()).unwrap();
            check(&problem, &out).unwrap();
            replay(&problem, &out).unwrap();
            assert_eq!(represented(&out), actual_solutions(&problem.system));
            assert_eq!(
                out.reason == "inconsistent",
                actual_solutions(&problem.system).is_empty()
            );
        }
    }
}

#[test]
fn all_operations_and_affine_nullspace_are_checked_against_original_rows() {
    let problem = fixture();
    let out = solve(problem.clone()).unwrap();
    let audit = check(&problem, &out).unwrap();
    assert_eq!(audit["affine_space_checked"], true);
    assert!(
        out.operations
            .iter()
            .any(|op| matches!(op, Operation::Swap { .. }))
    );
    assert!(
        out.operations
            .iter()
            .any(|op| matches!(op, Operation::Scale { .. }))
    );
    assert!(
        out.operations
            .iter()
            .any(|op| matches!(op, Operation::Subtract { .. }))
    );
    assert_eq!(represented(&out), actual_solutions(&problem.system));
    let raw = serde_json::to_vec(&out).unwrap();
    assert_eq!(serde_json::from_slice::<Outcome>(&raw).unwrap(), out);
    let under = Problem {
        system: System {
            prime: 5,
            variables: 3,
            rows: vec![vec![1, 2, 3]],
            right: vec![4],
        },
        policy: Policy::default(),
    };
    let out = solve(under.clone()).unwrap();
    assert_eq!(out.free.len(), 2);
    check(&under, &out).unwrap();
    assert_eq!(represented(&out), actual_solutions(&under.system));
}

#[test]
fn caps_remain_partial_and_dimension_and_prime_bounds_are_enforced() {
    for scalar in [true, false] {
        let mut problem = fixture();
        if scalar {
            problem.policy.scalar_updates = 0;
        } else {
            problem.policy.row_operations = 0;
        }
        let out = solve(problem.clone()).unwrap();
        assert_eq!(
            out.reason,
            if scalar {
                "scalar_update_cap"
            } else {
                "row_operation_cap"
            }
        );
        assert!(out.space.is_none());
        assert_eq!(check(&problem, &out).unwrap()["capped"], true);
        replay(&problem, &out).unwrap();
    }
    let mut problem = Problem {
        system: System {
            prime: 5,
            variables: 2,
            rows: vec![vec![0, 0]],
            right: vec![0],
        },
        policy: Policy {
            nullity: 0,
            ..Policy::default()
        },
    };
    let out = solve(problem.clone()).unwrap();
    assert_eq!(out.reason, "nullity_cap");
    assert_eq!(out.free, vec![0, 1]);
    check(&problem, &out).unwrap();
    replay(&problem, &out).unwrap();
    for p in [0, 1, 4, 9, 255, 258] {
        problem.system.prime = p;
        assert!(solve(problem.clone()).is_err());
    }
    problem.system.prime = 257;
    assert!(problem.validate().unwrap() < 1024 * 1024 * 1024);
    problem.system.variables = 1025;
    assert!(solve(problem.clone()).is_err());
    problem = fixture();
    problem.system.rows[0][0] = problem.system.prime;
    assert!(solve(problem).is_err());
}

#[test]
fn altered_operations_matrix_solution_input_and_work_cannot_replay() {
    let problem = fixture();
    let out = solve(problem.clone()).unwrap();
    let mut wrong = out.clone();
    let index = wrong
        .operations
        .iter()
        .position(|op| matches!(op, Operation::Scale { .. }))
        .unwrap();
    if let Operation::Scale { factor, .. } = &mut wrong.operations[index] {
        *factor = 0;
    }
    assert!(check(&problem, &wrong).is_err());
    wrong = out.clone();
    wrong.augmented[0][0] = (wrong.augmented[0][0] + 1) % 5;
    assert!(check(&problem, &wrong).is_err());
    wrong = out.clone();
    wrong.space.as_mut().unwrap().particular[0] =
        (wrong.space.as_ref().unwrap().particular[0] + 1) % 5;
    assert!(check(&problem, &wrong).is_err());
    wrong = out.clone();
    wrong.work.scalar_updates += 1;
    assert!(check(&problem, &wrong).is_err());
    wrong = out.clone();
    wrong.work.inverse_tests += 1;
    assert!(replay(&problem, &wrong).is_err());
    let mut changed = problem.clone();
    changed.system.right[0] = (changed.system.right[0] + 1) % 5;
    assert!(check(&changed, &out).is_err());
    assert!(replay(&changed, &out).is_err());
}
