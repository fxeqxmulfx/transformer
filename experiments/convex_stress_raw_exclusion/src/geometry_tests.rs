use crate::*;

fn line() -> (Vec<Point>, Vec<usize>) {
    (
        [-3, -1, 1, 3]
            .into_iter()
            .enumerate()
            .map(|(state, x)| Point { x, y: 0, state })
            .collect(),
        vec![0, 0, 2, 2],
    )
}

#[test]
fn exterior_identification_and_missing_class_limit_match_lean() -> Result<()> {
    let (points, labels) = line();
    for (x, expected, excluded) in [(-4, 0, 2), (4, 2, 0)] {
        let query = Point { x, y: 0, state: 4 };
        let result = exclusion::decide(&points, &labels, query, &[0, 2], &mut Work::default())?;
        assert_eq!(result.survivors, [expected]);
        assert_eq!(result.exclusions.len(), 1);
        assert_eq!(result.exclusions[0].candidate, excluded);
        let ex = &result.exclusions[0];
        let cert = &ex.certificate;
        assert_eq!(cert.weights.len(), 2);
        // Canonical hull endpoints can give a different valid rational
        // combination than Lean's chosen witness. Check the actual identity.
        let numerator_x: i128 = cert
            .weights
            .iter()
            .map(|w| {
                let p = if w.state == query.state {
                    query
                } else {
                    points[w.state]
                };
                i128::from(w.numerator) * i128::from(p.x)
            })
            .sum();
        assert_eq!(
            numerator_x,
            i128::from(cert.denominator) * i128::from(points[ex.witness_state].x)
        );
        exclusion::verify(
            &result.exclusions[0],
            &points,
            &labels,
            query,
            &mut Work::default(),
        )?;
        // The original vocabulary also admits an unobserved class. Its singleton
        // augmented hull has no conflict, so full-vocabulary identification fails.
        let full = exclusion::decide(&points, &labels, query, &[0, 1, 2, 3], &mut Work::default())?;
        assert_eq!(full.survivors.len(), 3);
        assert!(full.survivors.contains(&1));
        assert!(full.survivors.contains(&3));
    }
    Ok(())
}

#[test]
fn gap_and_off_line_queries_do_not_invent_exclusions() -> Result<()> {
    let (points, labels) = line();
    for (x, y) in [(0, 0), (0, 1), (-4, 1)] {
        let d = exclusion::decide(
            &points,
            &labels,
            Point { x, y, state: 4 },
            &[0, 2],
            &mut Work::default(),
        )?;
        assert_eq!(d.survivors, [0, 2]);
        assert!(d.exclusions.is_empty());
    }
    Ok(())
}

#[test]
fn noncollinear_positive_triangle_certificate_is_exact() -> Result<()> {
    let points = vec![
        Point {
            x: 0,
            y: 0,
            state: 0,
        },
        Point {
            x: 2,
            y: 0,
            state: 1,
        },
        Point {
            x: 1,
            y: 1,
            state: 2,
        },
    ];
    let labels = [0, 0, 2];
    let query = Point {
        x: 1,
        y: 3,
        state: 3,
    };
    let d = exclusion::decide(&points, &labels, query, &[0, 2], &mut Work::default())?;
    assert_eq!(d.survivors, [2]);
    assert_eq!(d.exclusions[0].certificate.weights.len(), 3);
    exclusion::verify(
        &d.exclusions[0],
        &points,
        &labels,
        query,
        &mut Work::default(),
    )?;
    Ok(())
}

#[test]
fn conflict_certificates_reject_wrong_query_label_mass_state_and_coordinate() -> Result<()> {
    let (points, labels) = line();
    let q = Point {
        x: -4,
        y: 0,
        state: 4,
    };
    let good = exclusion::decide(&points, &labels, q, &[0, 2], &mut Work::default())?
        .exclusions
        .remove(0);
    for mutation in 0..8 {
        let mut ex = good.clone();
        let mut query = q;
        match mutation {
            0 => ex.candidate = 0,
            1 => ex.witness_state = 2,
            2 => ex.certificate.denominator += 1,
            3 => ex.certificate.weights[0].numerator += 1,
            4 => ex.certificate.weights[0].state = 0,
            5 => ex.certificate.weights.pop().map(|_| ()).unwrap(),
            6 => query.x -= 1,
            _ => query.state = 9,
        }
        assert!(exclusion::verify(&ex, &points, &labels, query, &mut Work::default()).is_err());
    }
    assert!(exclusion::decide(&points, &labels, q, &[2, 0], &mut Work::default()).is_err());
    assert!(exclusion::decide(&points, &labels, q, &[0, 0], &mut Work::default()).is_err());
    assert!(exclusion::decide(&points, &labels, points[0], &[0, 2], &mut Work::default()).is_err());
    Ok(())
}

#[test]
fn signed_extreme_coordinates_refuse_overflow() {
    let points = [
        Point {
            x: i64::MIN,
            y: i64::MIN,
            state: 0,
        },
        Point {
            x: i64::MAX,
            y: i64::MIN,
            state: 1,
        },
        Point {
            x: 0,
            y: 0,
            state: 2,
        },
    ];
    assert!(
        exclusion::decide(
            &points,
            &[0, 0, 2],
            Point {
                x: i64::MIN,
                y: i64::MAX,
                state: 3
            },
            &[0, 2],
            &mut Work::default()
        )
        .is_err()
    );
}

#[test]
fn incompatible_geometry_can_exclude_every_declared_candidate() -> Result<()> {
    let points = [-3, -1, 1]
        .into_iter()
        .enumerate()
        .map(|(state, x)| Point { x, y: 0, state })
        .collect::<Vec<_>>();
    let result = exclusion::decide(
        &points,
        &[0, 2, 0],
        Point {
            x: -4,
            y: 0,
            state: 3,
        },
        &[0, 2],
        &mut Work::default(),
    )?;
    assert!(result.survivors.is_empty());
    assert_eq!(result.exclusions.len(), 2);
    Ok(())
}

#[test]
fn prediction_weights_and_candidate_completeness_have_separate_checks() -> Result<()> {
    let (points, labels) = line();
    let q = Point {
        x: -4,
        y: 0,
        state: 4,
    };
    let result = exclusion::decide(&points, &labels, q, &[0, 2], &mut Work::default())?;
    let row = Row {
        block: 0,
        label: 0,
        pool: vec![0, 2],
        exclusions: result.exclusions,
        denominator: 2,
        weights: vec![
            RationalWeight {
                state: 0,
                numerator: 1,
            },
            RationalWeight {
                state: 1,
                numerator: 1,
            },
        ],
    };
    exclusion::verify_row(&row, &points, &labels, q, &mut Work::default())?;
    // This feature mixture has input barycenter -2, not the query -4. The
    // independent identifying certificate justifies the raw label instead.
    assert_ne!(points[0].x + points[1].x, 2 * q.x);
    for mutation in 0..4 {
        let mut bad = row.clone();
        match mutation {
            0 => bad.pool.pop().map(|_| ()).unwrap(),
            1 => bad.exclusions.clear(),
            2 => bad.weights[0].state = 2,
            _ => bad.denominator += 1,
        }
        assert!(exclusion::verify_row(&bad, &points, &labels, q, &mut Work::default()).is_err());
    }
    Ok(())
}
