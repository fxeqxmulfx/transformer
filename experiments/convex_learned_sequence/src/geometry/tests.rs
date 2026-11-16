use super::{ConeGeometry, GeometryLimits};
use crate::domain::{DomainLimits, RawDomain};
use crate::model::{ComplexModel, ModelLimits};
use crate::projection::ProjectionProblem;
use crate::{ProjectionLimits, Rational, Refusal};
use num_traits::{One, Zero};

fn integer(value: i64) -> Rational {
    Rational::from_integer(value.into())
}
fn vector(values: &[i64]) -> Vec<Rational> {
    values.iter().map(|&v| integer(v)).collect()
}
fn arithmetic() -> ProjectionLimits {
    ProjectionLimits {
        max_active_sets: 256,
        max_matrix_entries: 10_000,
        max_coefficient_bits: 4096,
    }
}
fn limits() -> GeometryLimits {
    GeometryLimits {
        max_normal_projections: 8,
        max_active_candidates: 2048,
        max_rational_entries: 100_000,
        max_working_bytes: 128 * 1024 * 1024,
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
fn diagonal() -> ConeGeometry {
    ConeGeometry::new(
        ProjectionProblem::new(2, vec![vector(&[1, -1]), vector(&[-1, 1])], arithmetic()).unwrap(),
        limits(),
    )
    .unwrap()
}

#[test]
fn diagonal_queries_move_both_coordinates_when_individual_probes_fail() {
    let geometry = diagonal();
    assert_eq!(geometry.centre(), vector(&[0, 0]));
    assert_eq!(geometry.certificate().forced, vec![true, true]);
    assert_eq!(geometry.basis().vectors, vec![vector(&[1, 1])]);
    assert_eq!(geometry.basis().free_coordinates, vec![1]);
    assert_eq!(geometry.equalities().len(), 2);
    assert!(
        geometry
            .certificate()
            .normal_projections
            .iter()
            .all(|p| p.point.iter().all(Rational::is_zero))
    );
    let direction = geometry.combine(&vector(&[3])).unwrap();
    let cap = Rational::new(1.into(), 10.into());
    let query = geometry
        .symmetric_queries(geometry.centre(), &direction, &cap)
        .unwrap();
    assert_eq!(query.step, cap);
    assert_eq!(query.plus, vec![Rational::new(3.into(), 10.into()); 2]);
    assert_eq!(query.minus, vec![Rational::new((-3).into(), 10.into()); 2]);
    for direction in [vector(&[1, 0]), vector(&[0, 1])] {
        assert_eq!(
            geometry.verify_direction(&direction),
            Err(Refusal::QueryDirection)
        );
    }
}

#[test]
fn centre_mixture_opens_nonforced_boundaries_and_keeps_lineality() {
    let geometry = ConeGeometry::new(
        ProjectionProblem::new(
            3,
            vec![vector(&[1, 0, 0]), vector(&[0, 1, 0]), vector(&[0, 0, 0])],
            arithmetic(),
        )
        .unwrap(),
        limits(),
    )
    .unwrap();
    assert_eq!(geometry.centre(), vector(&[1, 1, 0]));
    assert_eq!(geometry.certificate().forced, vec![false, false, true]);
    assert_eq!(geometry.basis().vectors.len(), 3);
    let point = vector(&[0, 0, 0]);
    let direction = vector(&[2, -3, 4]);
    geometry.verify_direction(&direction).unwrap();
    assert_eq!(
        geometry.symmetric_queries(&point, &direction, &integer(1)),
        Err(Refusal::QueryDirection)
    );
    let mixed = geometry
        .mixture(&point, &Rational::new(1.into(), 4.into()))
        .unwrap();
    let query = geometry
        .symmetric_queries(&mixed, &direction, &integer(1))
        .unwrap();
    assert_eq!(query.step, Rational::new(1.into(), 16.into()));
    geometry.feasible(&query.plus).unwrap();
    geometry.feasible(&query.minus).unwrap();
    assert_ne!(query.plus[2], query.minus[2]);
    assert_eq!(geometry.equalities().len(), geometry.normals().len());
}

#[test]
fn zero_cone_has_a_certified_empty_basis_and_no_lost_rows() {
    let geometry = ConeGeometry::new(
        ProjectionProblem::new(
            2,
            vec![
                vector(&[1, 0]),
                vector(&[-1, 0]),
                vector(&[0, 1]),
                vector(&[0, -1]),
            ],
            arithmetic(),
        )
        .unwrap(),
        limits(),
    )
    .unwrap();
    assert_eq!(geometry.normals().len(), 4);
    assert_eq!(geometry.equalities().len(), 4);
    assert_eq!(geometry.certificate().normal_projections.len(), 4);
    assert!(geometry.basis().vectors.is_empty());
    assert_eq!(geometry.basis().recovery.len(), 2);
    assert!(geometry.basis().recovery.iter().all(|r| r.len() == 4));
    assert_eq!(geometry.combine(&[]).unwrap(), vector(&[0, 0]));
    assert_eq!(
        geometry.verify_direction(&vector(&[1, 0])),
        Err(Refusal::QueryDirection)
    );
    geometry.verify(geometry.certificate()).unwrap();
}

#[test]
fn no_constraints_retains_every_coordinate_as_a_direction() {
    let geometry = ConeGeometry::new(
        ProjectionProblem::new(3, vec![], arithmetic()).unwrap(),
        limits(),
    )
    .unwrap();
    assert_eq!(geometry.basis().free_coordinates, vec![0, 1, 2]);
    assert_eq!(
        geometry.combine(&vector(&[2, -3, 4])).unwrap(),
        vector(&[2, -3, 4])
    );
    let query = geometry
        .symmetric_queries(&vector(&[1, 2, 3]), &vector(&[2, -3, 4]), &integer(2))
        .unwrap();
    assert_eq!(query.plus, vector(&[5, -4, 11]));
    assert_eq!(query.minus, vector(&[-3, 8, -5]));
}

#[test]
fn slanted_plane_retains_rational_nonorthogonal_basis_and_recovery() {
    let geometry = ConeGeometry::new(
        ProjectionProblem::new(
            3,
            vec![vector(&[2, 1, 1]), vector(&[-2, -1, -1])],
            arithmetic(),
        )
        .unwrap(),
        limits(),
    )
    .unwrap();
    let half = Rational::new((-1).into(), 2.into());
    assert_eq!(
        geometry.basis().vectors,
        vec![
            vec![half.clone(), integer(1), integer(0)],
            vec![half, integer(0), integer(1)],
        ]
    );
    assert_eq!(
        geometry.basis().recovery[0],
        vec![Rational::new(1.into(), 2.into()), integer(0)]
    );
    assert_eq!(
        crate::checked_dot(
            &geometry.basis().vectors[0],
            &geometry.basis().vectors[1],
            arithmetic()
        )
        .unwrap(),
        Rational::new(1.into(), 4.into())
    );
    let direction = geometry.combine(&vector(&[2, 4])).unwrap();
    assert_eq!(direction, vector(&[-3, 2, 4]));
    let query = geometry
        .symmetric_queries(geometry.centre(), &direction, &integer(1))
        .unwrap();
    assert_eq!(query.plus, direction);
    geometry.verify(geometry.certificate()).unwrap();
}

#[test]
fn independent_geometry_checker_rejects_missing_or_corrupt_certificates() {
    let geometry = diagonal();
    let original = geometry.certificate().clone();
    let mut bad = original.clone();
    bad.normal_projections.pop();
    assert_eq!(geometry.verify(&bad), Err(Refusal::Shape));
    let mut bad = original.clone();
    bad.normal_projections[0].point[0] += Rational::one();
    assert!(geometry.verify(&bad).is_err());
    let mut bad = original.clone();
    bad.centre[0] += Rational::one();
    assert_eq!(geometry.verify(&bad), Err(Refusal::GeometryCertificate));
    let mut bad = original.clone();
    bad.forced[0] = false;
    assert_eq!(geometry.verify(&bad), Err(Refusal::GeometryCertificate));
    let mut bad = original.clone();
    bad.basis.vectors[0][0] = Rational::zero();
    assert_eq!(geometry.verify(&bad), Err(Refusal::BasisKernel));
    let mut bad = original.clone();
    bad.basis.vectors[0] = vector(&[2, 2]);
    assert_eq!(geometry.verify(&bad), Err(Refusal::BasisCoordinates));
    let mut bad = original.clone();
    bad.basis.recovery[0].fill(Rational::zero());
    assert_eq!(geometry.verify(&bad), Err(Refusal::BasisRecovery));
    let mut bad = original.clone();
    bad.basis.vectors.clear();
    bad.basis.free_coordinates.clear();
    assert_eq!(geometry.verify(&bad), Err(Refusal::BasisRecovery));
    let mut bad = original.clone();
    bad.basis.vectors.push(vector(&[1, 1]));
    bad.basis.free_coordinates.push(1);
    assert_eq!(geometry.verify(&bad), Err(Refusal::BasisCoordinates));
    geometry.verify(&original).unwrap();
}

#[test]
fn nonorthogonal_pairing_recovery_uses_the_complete_gram_system() {
    let geometry = ConeGeometry::new(
        ProjectionProblem::new(
            3,
            vec![vector(&[2, 1, 1]), vector(&[-2, -1, -1])],
            arithmetic(),
        )
        .unwrap(),
        limits(),
    )
    .unwrap();
    let pairings = vector(&[2, 4]);
    let recovered = geometry.recover_pairings(&pairings).unwrap();
    assert_eq!(recovered, vector(&[-2, 1, 3]));
    geometry.verify_pairings(&recovered, &pairings).unwrap();
    let wrong = geometry.combine(&pairings).unwrap();
    assert_eq!(wrong, vector(&[-3, 2, 4]));
    assert_eq!(
        geometry.verify_pairings(&wrong, &pairings),
        Err(Refusal::QueryPairing)
    );
    assert_eq!(
        geometry.recover_pairings(&vector(&[2])),
        Err(Refusal::Shape)
    );
    assert_eq!(
        geometry.verify_pairings(&vector(&[0, 1, 3]), &pairings),
        Err(Refusal::QueryDirection)
    );
}

#[test]
fn certified_empty_basis_pairing_recovery_keeps_the_full_zero_vector() {
    let geometry = ConeGeometry::new(
        ProjectionProblem::new(
            2,
            vec![
                vector(&[1, 0]),
                vector(&[-1, 0]),
                vector(&[0, 1]),
                vector(&[0, -1]),
            ],
            arithmetic(),
        )
        .unwrap(),
        limits(),
    )
    .unwrap();
    assert_eq!(geometry.recover_pairings(&[]).unwrap(), vector(&[0, 0]));
    geometry.verify_pairings(&vector(&[0, 0]), &[]).unwrap();
    assert_eq!(
        geometry.verify_pairings(&vector(&[1, 0]), &[]),
        Err(Refusal::QueryDirection)
    );
}

#[test]
fn geometry_limits_and_malformed_queries_refuse_explicitly() {
    let problem =
        ProjectionProblem::new(2, vec![vector(&[1, -1]), vector(&[-1, 1])], arithmetic()).unwrap();
    for limited in [
        GeometryLimits {
            max_normal_projections: 1,
            ..limits()
        },
        GeometryLimits {
            max_active_candidates: 1,
            ..limits()
        },
        GeometryLimits {
            max_rational_entries: 1,
            ..limits()
        },
        GeometryLimits {
            max_working_bytes: 1,
            ..limits()
        },
    ] {
        assert!(matches!(
            ConeGeometry::new(problem.clone(), limited),
            Err(Refusal::Budget)
        ));
    }
    let geometry = diagonal();
    assert_eq!(geometry.feasible(&[]), Err(Refusal::Shape));
    assert_eq!(geometry.combine(&[]), Err(Refusal::Shape));
    assert_eq!(
        geometry.mixture(&vector(&[0, 0]), &integer(0)),
        Err(Refusal::QueryStep)
    );
    assert_eq!(
        geometry.mixture(&vector(&[0, 0]), &integer(2)),
        Err(Refusal::QueryStep)
    );
    assert_eq!(
        geometry.symmetric_queries(&vector(&[0, 0]), &vector(&[1, 1]), &integer(0)),
        Err(Refusal::QueryStep)
    );
    assert_eq!(
        geometry.symmetric_queries(&vector(&[1, 0]), &vector(&[1, 1]), &integer(1)),
        Err(Refusal::Infeasible)
    );
}

#[test]
fn every_pattern_has_complete_geometry_and_feasible_basis_queries() {
    let mut degenerate = 0;
    let mut full = 0;
    for (objects, context) in [(2, 2), (3, 1)] {
        let domain = RawDomain::new(objects, context, domain_limits()).unwrap();
        for mask in 0..domain.plan().patterns {
            let problem = domain.cone(mask, arithmetic()).unwrap();
            let proposal: Vec<_> = (0..domain.plan().coordinates)
                .map(|d| integer(d as i64 - 2))
                .collect();
            let point = problem.project(&proposal).unwrap().point;
            let geometry = ConeGeometry::new(problem, limits()).unwrap();
            geometry.verify(geometry.certificate()).unwrap();
            assert_eq!(geometry.normals().len(), domain.plan().prefixes);
            if geometry.basis().vectors.len() < domain.plan().coordinates {
                degenerate += 1;
            } else {
                full += 1;
            }
            let mixed = geometry
                .mixture(&point, &Rational::new(1.into(), 3.into()))
                .unwrap();
            let pairings: Vec<_> = (0..geometry.basis().vectors.len())
                .map(|i| integer(2 * i as i64 - 3))
                .collect();
            let recovered = geometry.recover_pairings(&pairings).unwrap();
            geometry.verify_pairings(&recovered, &pairings).unwrap();
            geometry
                .symmetric_queries(&mixed, &recovered, &Rational::new(1.into(), 8.into()))
                .unwrap();
            for direction in &geometry.basis().vectors {
                let query = geometry
                    .symmetric_queries(&mixed, direction, &Rational::new(1.into(), 8.into()))
                    .unwrap();
                assert!(query.step > Rational::zero());
                geometry.feasible(&query.plus).unwrap();
                geometry.feasible(&query.minus).unwrap();
            }
        }
    }
    assert!(degenerate > 0);
    assert!(full > 0);
}

#[test]
fn certified_queries_execute_actual_recovered_complex_forwards() {
    let model = ComplexModel::zero(
        3,
        1,
        ModelLimits {
            domain: domain_limits(),
            projection: arithmetic(),
            max_working_bytes: 1024 * 1024 * 1024,
        },
    )
    .unwrap();
    let mut parameters = model.parameters().to_vec();
    let mut geometries = Vec::new();
    for mask in 0..model.domain().plan().patterns {
        let geometry =
            ConeGeometry::new(model.domain().cone(mask, arithmetic()).unwrap(), limits()).unwrap();
        for class in 0..3 {
            for sign in 0..2 {
                for component in 0..2 {
                    let first = model
                        .parameter_index(mask, class, sign, component, 0)
                        .unwrap();
                    parameters[first..first + 4].clone_from_slice(geometry.centre());
                }
            }
        }
        geometries.push(geometry);
    }
    let model = model.with_parameters(parameters).unwrap();
    let mask = model.domain().plan().patterns - 1;
    let geometry = &geometries[mask];
    let query = geometry
        .symmetric_queries(
            geometry.centre(),
            &geometry.basis().vectors[0],
            &Rational::new(1.into(), 16.into()),
        )
        .unwrap();
    let first = model.parameter_index(mask, 1, 0, 0, 0).unwrap();
    let mut plus = model.parameters().to_vec();
    let mut minus = plus.clone();
    plus[first..first + 4].clone_from_slice(&query.plus);
    minus[first..first + 4].clone_from_slice(&query.minus);
    let plus = model.with_parameters(plus).unwrap();
    let minus = model.with_parameters(minus).unwrap();
    for input in model.domain().prefixes() {
        let base = model.forward(input).unwrap();
        let upper = plus.forward(input).unwrap();
        let lower = minus.forward(input).unwrap();
        assert!(upper.cross_entropy(1).unwrap() < base.cross_entropy(1).unwrap());
        assert!(base.cross_entropy(1).unwrap() < lower.cross_entropy(1).unwrap());
        assert_eq!(upper.object(), 1);
        assert!((upper.complex_scores()[1].re - 1.0 / 16.0).abs() < 1e-12);
        assert!((upper.complex_scores()[1].im - 1.0 / 16.0).abs() < 1e-12);
    }
}
