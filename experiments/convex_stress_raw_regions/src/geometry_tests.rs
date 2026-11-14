use crate::*;

fn p(x: i64, y: i64, state: usize) -> Point {
    Point { x, y, state }
}

#[test]
fn exact_triangle_lattice_membership_has_independent_coordinate_certificates() -> Result<()> {
    let points = vec![p(0, 0, 0), p(6, 0, 1), p(0, 6, 2), p(1, 1, 3), p(2, 2, 4)];
    let mut work = GeometryWork::default();
    let hull = Hull::build(3, points.clone(), &mut work)?;
    assert_eq!(hull.vertices.len(), 3);
    let mut covered = 0;
    for x in -1..=7 {
        for y in -1..=7 {
            let query = p(x, y, usize::MAX);
            let certificate = hull.certify(query, &mut work)?;
            // Analytic triangle, independent of hull orientation/fan code.
            assert_eq!(certificate.is_some(), x >= 0 && y >= 0 && x + y <= 6);
            if let Some(c) = certificate {
                covered += 1;
                geometry::verify(&c, query, &points, &[3; 5], &mut work)?;
                assert!(c.weights.len() <= 3);
                assert!(c.weights.iter().all(|w| w.numerator > 0));
            }
        }
    }
    assert_eq!(covered, 28);
    assert_eq!(work.certificates_verified, 28);
    Ok(())
}

#[test]
fn collinear_points_duplicate_coordinates_and_exact_boundaries() -> Result<()> {
    let mut work = GeometryWork::default();
    let points = vec![p(0, 0, 0), p(3, 3, 1), p(6, 6, 2), p(3, 3, 3)];
    let hull = Hull::build(2, points.clone(), &mut work)?;
    assert_eq!(hull.vertices, vec![points[0], points[2]]);
    for x in 0..=6 {
        let query = p(x, x, 999);
        let c = hull.certify(query, &mut work)?.unwrap();
        geometry::verify(&c, query, &points, &[2; 4], &mut work)?;
    }
    assert!(hull.certify(p(3, 4, 0), &mut work)?.is_none());
    assert!(hull.certify(p(7, 7, 0), &mut work)?.is_none());
    let singleton = Hull::build(1, vec![p(3, 3, 0)], &mut work)?;
    assert!(geometry::overlaps(&hull, &singleton, &mut work)?);
    assert!(singleton.certify(p(3, 4, 0), &mut work)?.is_none());
    assert!(Hull::build(0, vec![], &mut work).is_err());
    assert!(
        geometry::overlaps(
            &hull,
            &Hull {
                label: 0,
                vertices: vec![]
            },
            &mut work
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn intersecting_edges_without_contained_vertices_and_touching_hulls_are_refused() -> Result<()> {
    let mut work = GeometryWork::default();
    let a = Hull::build(
        0,
        vec![p(-3, -1, 0), p(3, -1, 1), p(3, 1, 2), p(-3, 1, 3)],
        &mut work,
    )?;
    let b = Hull::build(
        2,
        vec![p(-1, -3, 0), p(1, -3, 1), p(1, 3, 2), p(-1, 3, 3)],
        &mut work,
    )?;
    for &point in &a.vertices {
        assert!(b.certify(point, &mut work)?.is_none());
    }
    for &point in &b.vertices {
        assert!(a.certify(point, &mut work)?.is_none());
    }
    assert!(geometry::overlaps(&a, &b, &mut work)?);
    let touching = Hull::build(
        3,
        vec![p(3, -1, 0), p(5, -1, 1), p(5, 1, 2), p(3, 1, 3)],
        &mut work,
    )?;
    assert!(geometry::overlaps(&a, &touching, &mut work)?);
    let disjoint = Hull::build(
        1,
        vec![p(4, -1, 0), p(6, -1, 1), p(6, 1, 2), p(4, 1, 3)],
        &mut work,
    )?;
    assert!(!geometry::overlaps(&a, &disjoint, &mut work)?);
    Ok(())
}

#[test]
fn independent_certificate_verification_rejects_mass_label_state_and_coordinate_changes()
-> Result<()> {
    let mut work = GeometryWork::default();
    let points = vec![p(0, 0, 0), p(6, 0, 1), p(0, 6, 2)];
    let hull = Hull::build(4, points.clone(), &mut work)?;
    let query = p(1, 2, 99);
    let certificate = hull.certify(query, &mut work)?.unwrap();
    for variant in 0..5 {
        let mut bad = certificate.clone();
        match variant {
            0 => bad.denominator += 1,
            1 => bad.label += 1,
            2 => bad.weights[0].numerator += 1,
            3 => bad.weights[0].state = points.len(),
            _ => bad.weights.push(bad.weights[0].clone()),
        }
        assert!(geometry::verify(&bad, query, &points, &[4; 3], &mut work).is_err());
    }
    assert!(geometry::verify(&certificate, p(2, 2, 99), &points, &[4; 3], &mut work).is_err());
    let mut wrong_points = points;
    wrong_points[0].x += 1;
    assert!(geometry::verify(&certificate, query, &wrong_points, &[4; 3], &mut work).is_err());
    Ok(())
}

#[test]
fn extreme_integer_geometry_refuses_overflow() {
    let mut work = GeometryWork::default();
    let points = vec![
        p(i64::MIN, i64::MIN, 0),
        p(i64::MAX, i64::MIN, 1),
        p(i64::MIN, i64::MAX, 2),
    ];
    assert!(Hull::build(0, points, &mut work).is_err());
}
