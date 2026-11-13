use super::*;
use std::collections::BTreeSet;

fn matrix() -> System {
    System {
        variables: 2,
        rows: vec![vec![1, 2], vec![3, 4]],
        right: vec![9, 23],
    }
}
fn fixture() -> lift::Input {
    lift::Input {
        system: matrix(),
        modulus: 4,
        prime: 2,
        base: vec![1, 2],
        policy: finite_system::Policy::default(),
    }
}

#[test]
fn all_small_scalar_lifts_match_exhaustive_original_integer_solutions() {
    let mut cases = 0;
    for p in [2u16, 3] {
        let q = p * p;
        for a in 0..q {
            for b in 0..q {
                for x in 0..p {
                    if (b as i64 - a as i64 * x as i64) % (p as i64) != 0 {
                        continue;
                    }
                    cases += 1;
                    let input = lift::Input {
                        system: System {
                            variables: 1,
                            rows: vec![vec![a as i64]],
                            right: vec![b as i64],
                        },
                        modulus: p,
                        prime: p,
                        base: vec![x],
                        policy: finite_system::Policy::default(),
                    };
                    let out = lift::run(input.clone()).unwrap();
                    lift::check(&input, &out).unwrap();
                    lift::replay(&input, &out).unwrap();
                    let actual: BTreeSet<_> = (0..p)
                        .map(|y| x + p * y)
                        .filter(|&z| (b as i64 - a as i64 * z as i64) % (q as i64) == 0)
                        .collect();
                    let mut represented = BTreeSet::new();
                    if let Some(space) = &out.elimination.space {
                        for weights in 0..p.pow(space.directions.len() as u32) {
                            let mut t = weights;
                            let mut y = space.particular[0] as u32;
                            for d in &space.directions {
                                y += (t % p) as u32 * d[0] as u32;
                                t /= p;
                            }
                            represented.insert(x + p * (y % p as u32) as u16);
                        }
                    }
                    assert_eq!(actual, represented);
                    assert_eq!(out.reason == "inconsistent", actual.is_empty());
                    if let Some(next) = out.next {
                        assert!(actual.contains(&next[0]));
                    }
                }
            }
        }
    }
    assert_eq!(cases, 97);
}

#[test]
fn actual_matrix_residuals_nonunique_digits_and_failed_lean_lift_are_explicit() {
    let input = fixture();
    let out = lift::run(input.clone()).unwrap();
    let audit = lift::check(&input, &out).unwrap();
    assert_eq!(out.residual, vec![1, 3]);
    assert_eq!(out.next, Some(vec![5, 2]));
    assert_eq!(audit["unique_digit"], false);
    assert_eq!(out.elimination.space.as_ref().unwrap().directions.len(), 1);
    assert_eq!(
        serde_json::from_slice::<lift::Step>(&serde_json::to_vec(&out).unwrap()).unwrap(),
        out
    );
    let failed = lift::Input {
        system: System {
            variables: 1,
            rows: vec![vec![2]],
            right: vec![0],
        },
        modulus: 2,
        prime: 2,
        base: vec![1],
        policy: finite_system::Policy::default(),
    };
    let out = lift::run(failed.clone()).unwrap();
    assert_eq!(out.residual, vec![-1]);
    assert_eq!(out.reason, "inconsistent");
    assert!(out.next.is_none());
    lift::check(&failed, &out).unwrap();
    lift::replay(&failed, &out).unwrap();
}

#[test]
fn coprime_recombination_matches_every_small_residue_pair_and_actual_matrix() {
    let mut checked = 0;
    for m in 2..=13u16 {
        for n in 2..=13u16 {
            if m * n > 65 {
                continue;
            }
            for x in 0..m {
                for y in 0..n {
                    let input = recombine::Input {
                        system: System {
                            variables: 1,
                            rows: vec![vec![0]],
                            right: vec![0],
                        },
                        left_modulus: m,
                        right_modulus: n,
                        left: vec![x],
                        right: vec![y],
                    };
                    let out = recombine::run(input.clone()).unwrap();
                    recombine::check(&input, &out).unwrap();
                    recombine::replay(&input, &out).unwrap();
                    let actual: Vec<_> = (0..m * n).filter(|z| z % m == x && z % n == y).collect();
                    if let Some(point) = out.point {
                        assert_eq!(actual, point);
                    } else {
                        assert_ne!(actual.len(), 1);
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 2000);
    let input = recombine::Input {
        system: matrix(),
        left_modulus: 4,
        right_modulus: 3,
        left: vec![1, 2],
        right: vec![2, 2],
    };
    let out = recombine::run(input.clone()).unwrap();
    assert_eq!(out.bezout, Some([1, -1]));
    assert_eq!(out.point, Some(vec![5, 2]));
    assert_eq!(out.residuals, Some(vec![0, 0]));
    recombine::check(&input, &out).unwrap();
}

#[test]
fn caps_invalid_base_and_bounds_do_not_produce_a_lift() {
    for which in 0..3 {
        let mut input = fixture();
        match which {
            0 => input.policy.scalar_updates = 0,
            1 => input.policy.row_operations = 0,
            _ => input.policy.nullity = 0,
        }
        let out = lift::run(input.clone()).unwrap();
        lift::check(&input, &out).unwrap();
        lift::replay(&input, &out).unwrap();
        assert!(out.next.is_none());
        assert_ne!(out.reason, "inconsistent");
    }
    let mut input = fixture();
    input.base[0] = 0;
    assert!(lift::run(input).is_err());
    input = fixture();
    input.prime = 4;
    assert!(lift::run(input).is_err());
    input = fixture();
    input.modulus = 256;
    assert!(lift::run(input).is_err());
    input = fixture();
    input.system.rows[0][0] = i64::MIN;
    assert!(lift::run(input).is_err());
    input = fixture();
    input.base.pop();
    assert!(lift::run(input).is_err());
}

#[test]
fn changed_residual_equations_representatives_bezout_and_work_are_rejected() {
    let input = fixture();
    let out = lift::run(input.clone()).unwrap();
    let mut wrong = out.clone();
    wrong.residual[0] += 1;
    assert!(lift::check(&input, &wrong).is_err());
    wrong = out.clone();
    wrong.elimination.problem.system.rows[0][0] = 0;
    assert!(lift::check(&input, &wrong).is_err());
    wrong = out.clone();
    wrong.next.as_mut().unwrap()[0] += 1;
    assert!(lift::check(&input, &wrong).is_err());
    wrong = out.clone();
    wrong.next_residuals.as_mut().unwrap()[0] += 8;
    assert!(lift::check(&input, &wrong).is_err());
    wrong = out;
    wrong.work.base_products += 1;
    assert!(lift::check(&input, &wrong).is_err());
    assert!(lift::replay(&input, &wrong).is_err());
    let input = recombine::Input {
        system: matrix(),
        left_modulus: 4,
        right_modulus: 3,
        left: vec![1, 2],
        right: vec![2, 2],
    };
    let out = recombine::run(input.clone()).unwrap();
    let mut wrong = out.clone();
    wrong.bezout.as_mut().unwrap()[0] += 1;
    assert!(recombine::check(&input, &wrong).is_err());
    wrong = out.clone();
    wrong.point.as_mut().unwrap()[0] += 1;
    assert!(recombine::check(&input, &wrong).is_err());
    wrong = out.clone();
    wrong.inverse_tests += 1;
    assert!(recombine::check(&input, &wrong).is_err());
    let mut changed = input.clone();
    changed.system.rows[0][0] += 1;
    assert!(recombine::replay(&changed, &out).is_err());
}
