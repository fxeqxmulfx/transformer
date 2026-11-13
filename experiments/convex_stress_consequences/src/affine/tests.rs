use super::*;
use std::collections::BTreeSet;
fn input(prime: u16, rows: Vec<Vec<u16>>, right: Vec<u16>) -> Request {
    Request {
        problem: finite_system::Problem {
            system: finite_system::System {
                prime,
                variables: rows[0].len(),
                rows,
                right,
            },
            policy: finite_system::Policy {
                nullity: 8,
                ..finite_system::Policy::default()
            },
        },
        policy: Policy::default(),
    }
}
fn checked(r: &Request) -> Report {
    let out = run(r.clone()).unwrap();
    check(r, &out).unwrap();
    replay(r, &out).unwrap();
    assert_eq!(
        serde_json::from_slice::<Report>(&serde_json::to_vec(&out).unwrap()).unwrap(),
        out
    );
    out
}

#[test]
fn all793_two_variable_affine_systems_match_exhaustive_solutions() {
    let (mut count, mut nonunique, mut impossible) = (0, 0, 0);
    for p in [2u16, 3] {
        for code in 0..(p as usize).pow(4) {
            let mut code = code;
            let coefficients: Vec<_> = (0..4)
                .map(|_| {
                    let x = (code % p as usize) as u16;
                    code /= p as usize;
                    x
                })
                .collect();
            for b0 in 0..p {
                for b1 in 0..p {
                    let r = input(
                        p,
                        vec![coefficients[..2].to_vec(), coefficients[2..].to_vec()],
                        vec![b0, b1],
                    );
                    let expected: BTreeSet<_> = (0..p)
                        .flat_map(|x| (0..p).map(move |y| vec![x, y]))
                        .filter(|v| {
                            r.problem
                                .system
                                .rows
                                .iter()
                                .zip(&r.problem.system.right)
                                .all(|(row, &b)| {
                                    row.iter()
                                        .zip(v)
                                        .map(|(&a, &z)| a as u32 * z as u32)
                                        .sum::<u32>()
                                        % p as u32
                                        == b as u32
                                })
                        })
                        .collect();
                    let out = checked(&r);
                    assert_eq!(
                        out.points
                            .iter()
                            .map(|point| point.values.clone())
                            .collect::<BTreeSet<_>>(),
                        expected
                    );
                    assert_eq!(out.complete, !expected.is_empty());
                    if expected.len() > 1 {
                        nonunique += 1;
                    }
                    if expected.is_empty() {
                        impossible += 1;
                    }
                    count += 1;
                }
            }
        }
    }
    assert_eq!(count, 793);
    assert!(nonunique > 0 && impossible > 0);
}

#[test]
fn nonzero_particular_solution_and_parameter_order_remain_explicit() {
    let r = input(3, vec![vec![1, 1, 1]], vec![1]);
    let out = checked(&r);
    assert!(out.complete);
    assert_eq!(out.cardinality, Some(9));
    assert_eq!(
        out.points
            .iter()
            .map(|p| p.weights.clone())
            .collect::<Vec<_>>(),
        vec![
            vec![0, 0],
            vec![1, 0],
            vec![2, 0],
            vec![0, 1],
            vec![1, 1],
            vec![2, 1],
            vec![0, 2],
            vec![1, 2],
            vec![2, 2]
        ]
    );
    assert_eq!(out.work.points, 9);
    assert_eq!(out.work.parameter_digits, 18);
    assert_eq!(out.work.coordinate_products, 54);
    assert_eq!(out.work.original_products, 27);
    assert_eq!(out.points[0].values, vec![1, 0, 0]);
    assert_eq!(out.work.ce_gradients, 0);
}

#[test]
fn all_enumeration_and_elimination_caps_keep_the_actual_prefix() {
    let original = input(3, vec![vec![1, 1, 1]], vec![1]);
    let full = checked(&original);
    for (cap, len) in [(0, 0), (1, 1), (5, 5), (9, 9), (10, 9)] {
        let mut r = original.clone();
        r.policy.points = cap;
        let out = checked(&r);
        assert_eq!(out.points, full.points[..len]);
        assert_eq!(out.complete, len == 9);
    }
    for coordinates in [false, true] {
        let mut r = original.clone();
        if coordinates {
            r.policy.coordinate_products = 13;
        } else {
            r.policy.original_products = 7;
        }
        let out = checked(&r);
        assert_eq!(out.points, full.points[..2]);
        assert_eq!(
            out.reason,
            if coordinates {
                "coordinate_product_cap"
            } else {
                "original_product_cap"
            }
        );
    }
    let mut r = original.clone();
    r.problem.policy.nullity = 1;
    let out = checked(&r);
    assert!(out.points.is_empty());
    assert!(out.elimination.space.is_none());
    let mut r = original;
    r.problem.policy.scalar_updates = 0;
    let out = checked(&r);
    assert!(out.points.is_empty());
    assert!(out.elimination.space.is_none());
}

#[test]
fn original_matrix_point_tuple_work_and_stop_mutations_are_rejected() {
    let r = input(3, vec![vec![1, 1, 1]], vec![1]);
    let out = checked(&r);
    for slot in 0..11 {
        let mut bad = out.clone();
        match slot {
            0 => bad.points[1].ordinal = 2,
            1 => bad.points[1].weights[0] = 2,
            2 => bad.points[1].values[0] = (bad.points[1].values[0] + 1) % 3,
            3 => bad.cardinality = Some(8),
            4 => bad.cardinality_overflow = true,
            5 => bad.complete = false,
            6 => bad.reason = "point_cap".into(),
            7 => bad.work.original_products -= 1,
            8 => bad.work.ce_gradients = 1,
            9 => bad.elimination.augmented[0][0] = 0,
            _ => {
                bad.points.pop();
            }
        }
        assert!(check(&r, &bad).is_err(), "mutation{slot}");
        assert!(replay(&r, &bad).is_err());
    }
    let mut changed = r.clone();
    changed.problem.system.right[0] = 2;
    assert!(check(&changed, &out).is_err());
    let mut invalid = r;
    invalid.policy.points = 65537;
    assert!(run(invalid).is_err());
}

#[test]
fn large_cardinality_is_bounded_without_overflow_or_false_completeness() {
    let r = input(257, vec![vec![0; 8]], vec![0]);
    let out = checked(&r);
    assert!(out.cardinality.is_none() && out.cardinality_overflow);
    assert_eq!(out.points.len(), 256);
    assert_eq!(out.reason, "point_cap");
    assert!(!out.complete);
    assert_eq!(out.points[255].weights, vec![255, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(out.points[255].values, out.points[255].weights);
    let r = input(257, vec![vec![1, 1, 1]], vec![256]);
    let out = checked(&r);
    assert_eq!(out.cardinality, Some(66049));
    assert_eq!(out.points.len(), 256);
    assert!(out.buffer_preflight_bytes <= 8u64 * 1024 * 1024 * 1024);
}
