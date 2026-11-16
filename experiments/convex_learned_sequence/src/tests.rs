use crate::domain::{DomainLimits, DomainPlan, RawDomain};
use crate::projection::{Projection, ProjectionProblem};
use crate::{ProjectionLimits, Rational, Refusal, checked_dot};
use num_bigint::BigInt;
use num_traits::{One, Zero};

fn integer(x: i64) -> Rational {
    Rational::from_integer(BigInt::from(x))
}
fn vector(values: &[i64]) -> Vec<Rational> {
    values.iter().map(|&x| integer(x)).collect()
}
fn ratio(n: i64, d: i64) -> Rational {
    Rational::new(BigInt::from(n), BigInt::from(d))
}
fn limits() -> ProjectionLimits {
    ProjectionLimits {
        max_active_sets: 256,
        max_matrix_entries: 10_000,
        max_coefficient_bits: 4096,
    }
}
fn domain_limits() -> DomainLimits {
    DomainLimits {
        max_prefixes: 8,
        max_patterns: 256,
        max_real_parameters: 100_000,
        max_row_entries: 1000,
    }
}

#[test]
fn halfline_and_unconstrained_projection_match_closed_forms() {
    let problem = ProjectionProblem::new(1, vec![vector(&[1])], limits()).unwrap();
    let result = problem.project(&vector(&[-1])).unwrap();
    assert_eq!(result.point, vector(&[0]));
    assert_eq!(result.multipliers, vector(&[1]));
    problem.verify(&vector(&[-1]), &result).unwrap();
    let unconstrained = ProjectionProblem::new(2, vec![], limits()).unwrap();
    assert_eq!(
        unconstrained.project(&vector(&[3, -2])).unwrap().point,
        vector(&[3, -2])
    );
}

#[test]
fn orthant_grid_matches_independent_coordinate_clipping() {
    let problem = ProjectionProblem::new(
        3,
        vec![vector(&[1, 0, 0]), vector(&[0, 1, 0]), vector(&[0, 0, 1])],
        limits(),
    )
    .unwrap();
    for x in -2..=2 {
        for y in -2..=2 {
            for z in -2..=2 {
                let proposal = vector(&[x, y, z]);
                let result = problem.project(&proposal).unwrap();
                assert_eq!(result.point, vector(&[x.max(0), y.max(0), z.max(0)]));
                problem.verify(&proposal, &result).unwrap();
            }
        }
    }
}

#[test]
fn slanted_projection_keeps_nonbinary_rational_coordinates() {
    let problem = ProjectionProblem::new(2, vec![vector(&[1, 2])], limits()).unwrap();
    let result = problem.project(&vector(&[-1, 0])).unwrap();
    assert_eq!(result.point, vec![ratio(-4, 5), ratio(2, 5)]);
    assert_eq!(result.multipliers, vec![ratio(1, 5)]);
    problem.verify(&vector(&[-1, 0]), &result).unwrap();
    assert_eq!(problem.project(&result.point).unwrap().point, result.point);
}

#[test]
fn redundant_and_degenerate_constraints_remain_checked() {
    let normals = vec![
        vector(&[1, 0]),
        vector(&[-1, 0]),
        vector(&[0, 1]),
        vector(&[0, 1]),
        vector(&[0, 0]),
    ];
    let problem = ProjectionProblem::new(2, normals.clone(), limits()).unwrap();
    assert_eq!(problem.normals(), &normals);
    let result = problem.project(&vector(&[-2, -3])).unwrap();
    assert_eq!(result.point, vector(&[0, 0]));
    assert_eq!(result.multipliers.len(), 5);
    problem.verify(&vector(&[-2, -3]), &result).unwrap();
    let mut forged = result.clone();
    forged.point[0] = Rational::one();
    forged.multipliers.fill(Rational::zero());
    assert_eq!(
        problem.verify(&vector(&[-2, -3]), &forged),
        Err(Refusal::Infeasible)
    );
}

#[test]
fn checker_rejects_each_failed_certificate_condition() {
    let problem = ProjectionProblem::new(1, vec![vector(&[1])], limits()).unwrap();
    let proposal = vector(&[-1]);
    let corrupt = |point, multiplier| Projection {
        point: vector(&[point]),
        multipliers: vector(&[multiplier]),
    };
    assert_eq!(
        problem.verify(&proposal, &corrupt(0, -1)),
        Err(Refusal::NegativeMultiplier)
    );
    assert_eq!(
        problem.verify(&proposal, &corrupt(-1, 0)),
        Err(Refusal::Infeasible)
    );
    assert_eq!(
        problem.verify(&proposal, &corrupt(1, 1)),
        Err(Refusal::Complementarity)
    );
    assert_eq!(
        problem.verify(&proposal, &corrupt(0, 0)),
        Err(Refusal::Stationarity)
    );
    let wrong_shape = Projection {
        point: vec![],
        multipliers: vector(&[1]),
    };
    assert_eq!(problem.verify(&proposal, &wrong_shape), Err(Refusal::Shape));
}

#[test]
fn finite_binary64_proposals_are_imported_without_rounding() {
    assert_eq!(limits().from_f64(0.125).unwrap(), ratio(1, 8));
    assert_eq!(limits().from_f64(-0.0).unwrap(), Rational::zero());
    assert_eq!(limits().from_f64(f64::INFINITY), Err(Refusal::NonFinite));
    assert_eq!(limits().from_f64(f64::NAN), Err(Refusal::NonFinite));
    let small = ProjectionLimits {
        max_coefficient_bits: 8,
        ..limits()
    };
    assert_eq!(small.from_f64(1e-100), Err(Refusal::Budget));
}

#[test]
fn complete_domain_rows_match_raw_bias_and_position_object_lookup() {
    let domain = RawDomain::new(2, 2, domain_limits()).unwrap();
    assert_eq!(
        domain.prefixes(),
        &[
            vec![],
            vec![0],
            vec![1],
            vec![0, 0],
            vec![0, 1],
            vec![1, 0],
            vec![1, 1]
        ]
    );
    assert_eq!(domain.plan().patterns, 128);
    assert_eq!(domain.plan().real_parameters, 5120);
    let weights = vector(&[3, -2, 5, 7, -11]);
    for (index, tokens) in domain.prefixes().iter().enumerate() {
        let mut expected = weights[0].clone();
        for (position, &object) in tokens.iter().enumerate() {
            expected += &weights[1 + position * 2 + object];
        }
        assert_eq!(
            checked_dot(&domain.rows()[index], &weights, limits()).unwrap(),
            expected
        );
        assert_eq!(domain.input_index(tokens).unwrap(), index);
    }
    assert_eq!(domain.input_index(&[0, 1, 0]), Err(Refusal::ContextTooLong));
    assert_eq!(domain.input_index(&[2]), Err(Refusal::InvalidObject));
}

#[test]
fn every_two_object_two_position_pattern_receives_a_checked_projection() {
    let domain = RawDomain::new(2, 2, domain_limits()).unwrap();
    let proposals = [vector(&[2, -3, 1, 4, -2]), vector(&[-1, 2, -4, 1, 3])];
    let mut certificates = 0;
    for mask in 0..domain.plan().patterns {
        let cone = domain.cone(mask, limits()).unwrap();
        assert_eq!(cone.normals().len(), domain.plan().prefixes);
        for proposal in &proposals {
            let result = cone.project(proposal).unwrap();
            cone.verify(proposal, &result).unwrap();
            certificates += 1;
        }
    }
    assert_eq!(certificates, 256);
}

#[test]
fn complete_three_object_domain_retains_control_and_both_other_objects() {
    let domain = RawDomain::new(3, 1, domain_limits()).unwrap();
    assert_eq!(domain.plan().prefixes, 4);
    assert_eq!(domain.plan().patterns, 16);
    assert_eq!(domain.plan().coordinates, 4);
    assert_eq!(domain.plan().real_parameters, 768);
    for mask in 0..16 {
        let cone = domain.cone(mask, limits()).unwrap();
        let proposal = vector(&[-1, 4, -3, 2]);
        let result = cone.project(&proposal).unwrap();
        cone.verify(&proposal, &result).unwrap();
    }
}

#[test]
fn budgets_refuse_whole_problem_instead_of_truncating_it() {
    let tiny = ProjectionLimits {
        max_active_sets: 1,
        ..limits()
    };
    assert!(matches!(
        ProjectionProblem::new(1, vec![vector(&[1])], tiny),
        Err(Refusal::Budget)
    ));
    let tiny = ProjectionLimits {
        max_matrix_entries: 1,
        ..limits()
    };
    assert!(matches!(
        ProjectionProblem::new(1, vec![vector(&[1])], tiny),
        Err(Refusal::Budget)
    ));
    assert!(matches!(
        DomainPlan::new(2, 3, domain_limits()),
        Err(Refusal::Budget)
    ));
    let tiny = DomainLimits {
        max_real_parameters: 767,
        ..domain_limits()
    };
    assert!(matches!(DomainPlan::new(3, 1, tiny), Err(Refusal::Budget)));
    assert!(matches!(
        DomainPlan::new(0, 1, domain_limits()),
        Err(Refusal::EmptyObjects)
    ));
    assert!(matches!(
        DomainPlan::new(usize::MAX, usize::MAX, domain_limits()),
        Err(Refusal::Overflow)
    ));
    let domain = RawDomain::new(3, 1, domain_limits()).unwrap();
    assert!(matches!(
        domain.cone(16, limits()),
        Err(Refusal::InvalidPattern)
    ));
}
