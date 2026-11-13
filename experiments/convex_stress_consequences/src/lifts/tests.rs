use super::*;
use std::collections::BTreeSet;
fn request() -> Request {
    Request {
        system: System {
            variables: 2,
            rows: vec![vec![2, 0], vec![0, 0]],
            right: vec![2, 0],
        },
        modulus: 2,
        prime: 2,
        base: vec![1, 0],
        elimination: finite_system::Policy {
            nullity: 8,
            ..finite_system::Policy::default()
        },
        enumeration: affine::Policy::default(),
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
fn all_nonunique_digits_are_lifted_in_the_declared_order() {
    let r = request();
    let out = checked(&r);
    assert_eq!(
        out.branches
            .iter()
            .map(|b| b.point.clone())
            .collect::<Vec<_>>(),
        vec![vec![1, 0], vec![3, 0], vec![1, 2], vec![3, 2]]
    );
    assert!(out.entire_digit_space);
    assert_eq!(out.residual, vec![0, 0]);
    assert_eq!(out.work.base_products, 4);
    assert_eq!(out.work.lifted_products, 16);
    assert_eq!(out.work.ce_gradients, 0);
}
#[test]
fn bounded_original_lifts_match_direct_exhaustive_representatives() {
    let mut cases = 0;
    for m in 1u16..=6 {
        for p in [2u16, 3, 5, 7] {
            for a in [-2i64, 0, 1, 2] {
                let mut r = request();
                r.modulus = m;
                r.prime = p;
                r.system.rows = vec![vec![a, 1], vec![0, 2]];
                for x in 0..m {
                    for y in 0..m {
                        r.base = vec![x, y];
                        r.system.right = vec![
                            a * x as i64 + y as i64 + 3 * m as i64,
                            2 * y as i64 - m as i64,
                        ];
                        let out = checked(&r);
                        let q = m * p;
                        let expected: BTreeSet<_> = (0..q)
                            .flat_map(|x| (0..q).map(move |y| vec![x, y]))
                            .filter(|v| {
                                v[0] % m == r.base[0]
                                    && v[1] % m == r.base[1]
                                    && r.system.rows.iter().zip(&r.system.right).all(|(row, &b)| {
                                        (b - row
                                            .iter()
                                            .zip(v)
                                            .map(|(&a, &x)| a * x as i64)
                                            .sum::<i64>())
                                        .rem_euclid(q as i64)
                                            == 0
                                    })
                            })
                            .collect();
                        assert_eq!(
                            out.branches
                                .iter()
                                .map(|b| b.point.clone())
                                .collect::<BTreeSet<_>>(),
                            expected
                        );
                        if !expected.is_empty() {
                            assert!(out.entire_digit_space);
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 1456);
}
#[test]
fn failed_lifts_and_partial_digit_banks_remain_distinct() {
    let mut r = request();
    r.enumeration.points = 2;
    let out = checked(&r);
    assert_eq!(out.branches.len(), 2);
    assert!(!out.entire_digit_space);
    assert_eq!(out.reason, "digits_point_cap");
    let mut r = request();
    r.system.rows = vec![vec![2, 0]];
    r.system.right = vec![2];
    r.base = vec![0, 0];
    let out = checked(&r);
    assert!(out.branches.is_empty());
    assert!(!out.entire_digit_space);
    assert!(out.digits.elimination.contradiction.is_some());
    let mut r = request();
    r.system.right[0] = 1;
    assert!(run(r).is_err());
}
#[test]
fn changed_residuals_digits_branches_and_work_are_rejected() {
    let r = request();
    let out = checked(&r);
    for slot in 0..8 {
        let mut bad = out.clone();
        match slot {
            0 => bad.residual[0] = 1,
            1 => bad.branches[1].point[0] = 1,
            2 => bad.branches[0].original_difference[0] = 4,
            3 => bad.entire_digit_space = false,
            4 => bad.work.lifted_products -= 1,
            5 => bad.digits.points[0].values[0] = 1,
            6 => bad.input.base[0] = 0,
            _ => {
                bad.branches.pop();
            }
        }
        assert!(check(&r, &bad).is_err(), "mutation{slot}");
        assert!(replay(&r, &bad).is_err());
    }
    let mut invalid = r;
    invalid.modulus = 256;
    invalid.prime = 2;
    assert!(run(invalid).is_err());
}
