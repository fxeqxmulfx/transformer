use crate::*;

fn points(coords: &[(i64, i64)]) -> Vec<Point> {
    coords
        .iter()
        .enumerate()
        .map(|(state, &(x, y))| Point { x, y, state })
        .collect()
}
fn crossing() -> Vec<Point> {
    points(&[(-1, -1), (-1, 1), (1, -1), (1, 1)])
}

#[test]
fn strict_crossing_improves_the_old_point_test_with_exact_lean_witnesses() -> Result<()> {
    let p = crossing();
    for (x, label) in [(-2, 0), (2, 2)] {
        let q = Point { x, y: 0, state: 4 };
        let old =
            parent::exclusion::decide(&p, &[0, 0, 2, 2], q, &[0, 2], &mut parent::Work::default())?;
        assert_eq!(old.survivors, [0, 2]);
        let mut work = Work::default();
        let new = exclusion::decide(&p, &[0, 0, 2, 2], q, &[0, 2], &mut work)?;
        assert_eq!(new.survivors, [label]);
        assert!(work.edge_pair_tests > 0);
        let ex = &new.exclusions[0];
        assert_eq!(ex.candidate_mixture.denominator, 3);
        assert_eq!(ex.other_mixture.denominator, 3);
        assert_eq!(
            ex.candidate_mixture
                .weights
                .iter()
                .find(|w| w.state == 4)
                .unwrap()
                .numerator,
            2
        );
        exclusion::verify(ex, &p, &[0, 0, 2, 2], q, &mut Work::default())?;
        // Adding unobserved classes keeps multiple survivors, as in CM49.
        let full = exclusion::decide(&p, &[0, 0, 2, 2], q, &[0, 1, 2, 3], &mut Work::default())?;
        assert_eq!(full.survivors.len(), 3);
        assert!(full.survivors.contains(&1) && full.survivors.contains(&3));
    }
    Ok(())
}

#[test]
fn old_point_and_segment_certificates_remain_valid() -> Result<()> {
    for (p, labels, q) in [
        (
            points(&[(-3, 0), (-1, 0), (1, 0), (3, 0)]),
            vec![0, 0, 2, 2],
            Point {
                x: -4,
                y: 0,
                state: 4,
            },
        ),
        (
            points(&[(0, 0), (2, 0), (1, 1)]),
            vec![0, 0, 2],
            Point {
                x: 1,
                y: 3,
                state: 3,
            },
        ),
    ] {
        let old = parent::exclusion::decide(&p, &labels, q, &[0, 2], &mut parent::Work::default())?;
        let new = exclusion::decide(&p, &labels, q, &[0, 2], &mut Work::default())?;
        assert_eq!(old.survivors, new.survivors);
        assert_eq!(old.exclusions.len(), 1);
        for ex in old.exclusions {
            exclusion::verify(
                &Exclusion {
                    candidate: ex.candidate,
                    candidate_mixture: ex.certificate,
                    other_mixture: certificates::dirac(labels[ex.witness_state], ex.witness_state),
                },
                &p,
                &labels,
                q,
                &mut Work::default(),
            )?;
        }
    }
    Ok(())
}

#[test]
fn polygons_and_query_inside_other_triangle_have_exact_certificates() -> Result<()> {
    let p = points(&[
        (-3, -1),
        (-1, -1),
        (-1, 1),
        (-3, 1),
        (1, -1),
        (3, -1),
        (3, 1),
        (1, 1),
    ]);
    let q = Point {
        x: -4,
        y: 0,
        state: 8,
    };
    let labels = [0, 0, 0, 0, 2, 2, 2, 2];
    assert_eq!(
        parent::exclusion::decide(&p, &labels, q, &[0, 2], &mut parent::Work::default())?.survivors,
        [0, 2]
    );
    let new = exclusion::decide(&p, &labels, q, &[0, 2], &mut Work::default())?;
    assert_eq!(new.survivors, [0]);
    exclusion::verify(&new.exclusions[0], &p, &labels, q, &mut Work::default())?;
    let p = points(&[(0, 0), (4, 0), (0, 4), (10, 0)]);
    let q = Point {
        x: 1,
        y: 1,
        state: 4,
    };
    let new = exclusion::decide(&p, &[0, 0, 0, 2], q, &[0, 2], &mut Work::default())?;
    assert_eq!(new.survivors, [0]);
    assert_eq!(
        new.exclusions[0].candidate_mixture.weights,
        [RationalWeight {
            state: 4,
            numerator: 1
        }]
    );
    assert_eq!(new.exclusions[0].other_mixture.weights.len(), 3);
    Ok(())
}

#[test]
fn gap_off_line_queries_and_inconsistent_observations_are_not_identified() -> Result<()> {
    let p = points(&[(-3, 0), (-1, 0), (1, 0), (3, 0)]);
    for (x, y) in [(0, 0), (0, 1), (-4, 1)] {
        let new = exclusion::decide(
            &p,
            &[0, 0, 2, 2],
            Point { x, y, state: 4 },
            &[0, 2],
            &mut Work::default(),
        )?;
        assert_eq!(new.survivors, [0, 2]);
        assert!(new.exclusions.is_empty());
    }
    let p = points(&[(-3, 0), (-1, 0), (1, 0)]);
    assert!(
        exclusion::decide(
            &p,
            &[0, 2, 0],
            Point {
                x: -4,
                y: 0,
                state: 3
            },
            &[0, 2],
            &mut Work::default()
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn both_mixtures_reject_label_query_mass_state_coordinate_and_order_mutations() -> Result<()> {
    let p = crossing();
    let q = Point {
        x: -2,
        y: 0,
        state: 4,
    };
    let good = exclusion::decide(&p, &[0, 0, 2, 2], q, &[0, 2], &mut Work::default())?
        .exclusions
        .remove(0);
    for mutation in 0..16 {
        let mut ex = good.clone();
        let mut q = q;
        match mutation {
            0 => ex.candidate = 0,
            1 => ex.candidate_mixture.label = 0,
            2 => ex.other_mixture.label = 2,
            3 => ex.candidate_mixture.denominator += 1,
            4 => ex.other_mixture.denominator += 1,
            5 => ex.candidate_mixture.weights[0].numerator = 0,
            6 => ex.other_mixture.weights[0].numerator += 1,
            7 => ex.candidate_mixture.weights[0].state = 0,
            8 => ex.other_mixture.weights[0].state = 2,
            9 => ex.candidate_mixture.weights.pop().map(|_| ()).unwrap(),
            10 => ex.other_mixture.weights[1].state = 4,
            11 => ex.candidate_mixture.weights.reverse(),
            12 => ex.other_mixture.weights.reverse(),
            13 => ex.other_mixture.weights[1].state = ex.other_mixture.weights[0].state,
            14 => q.x -= 1,
            _ => q.state = 9,
        }
        assert!(
            exclusion::verify(&ex, &p, &[0, 0, 2, 2], q, &mut Work::default()).is_err(),
            "mutation {mutation}"
        );
    }
    for pool in [vec![], vec![2, 0], vec![0, 0]] {
        assert!(exclusion::decide(&p, &[0, 0, 2, 2], q, &pool, &mut Work::default()).is_err());
    }
    assert!(exclusion::decide(&p, &[0, 0, 2, 2], p[0], &[0, 2], &mut Work::default()).is_err());
    Ok(())
}

#[test]
fn determinant_and_rational_comparison_overflow_are_refused() {
    let p = points(&[(i64::MIN, i64::MIN), (i64::MAX, i64::MIN), (0, 0)]);
    assert!(
        exclusion::decide(
            &p,
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
    let p = points(&[(i64::MAX - 2, 0), (i64::MAX - 1, 0)]);
    let q = Point {
        x: i64::MAX,
        y: 0,
        state: 2,
    };
    let d = u64::MAX - 1;
    let ex = Exclusion {
        candidate: 2,
        candidate_mixture: Certificate {
            label: 2,
            denominator: d,
            weights: vec![
                RationalWeight {
                    state: 0,
                    numerator: d / 2,
                },
                RationalWeight {
                    state: 2,
                    numerator: d / 2,
                },
            ],
        },
        other_mixture: Certificate {
            label: 0,
            denominator: u64::MAX,
            weights: vec![RationalWeight {
                state: 1,
                numerator: u64::MAX,
            }],
        },
    };
    let error = exclusion::verify(&ex, &p, &[2, 0], q, &mut Work::default()).unwrap_err();
    assert!(error.to_string().contains("rational comparison overflow"));
}

#[test]
fn row_weights_and_candidate_pool_are_separate_from_geometric_mixtures() -> Result<()> {
    let p = crossing();
    let q = Point {
        x: -2,
        y: 0,
        state: 4,
    };
    let d = exclusion::decide(&p, &[0, 0, 2, 2], q, &[0, 2], &mut Work::default())?;
    let row = Row {
        block: 0,
        label: 0,
        pool: vec![0, 2],
        exclusions: d.exclusions,
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
    exclusion::verify_row(&row, &p, &[0, 0, 2, 2], q, &mut Work::default())?;
    assert_ne!(p[0].x + p[1].x, 2 * q.x);
    for mutation in 0..5 {
        let mut bad = row.clone();
        match mutation {
            0 => bad.pool.pop().map(|_| ()).unwrap(),
            1 => bad.exclusions.clear(),
            2 => bad.weights[0].state = 2,
            3 => bad.denominator += 1,
            _ => bad.exclusions.push(bad.exclusions[0].clone()),
        }
        assert!(exclusion::verify_row(&bad, &p, &[0, 0, 2, 2], q, &mut Work::default()).is_err());
    }
    Ok(())
}

#[test]
fn finite_grid_contacts_agree_with_independent_reviewed_boolean_geometry() -> Result<()> {
    // Oblique edges, containment, vertex contacts and outside queries. The
    // reviewed Boolean oracle uses orientation signs instead of intersection
    // parameter ratios, and checks all candidate vertices in both directions.
    let p = points(&[(-3, -2), (-1, 0), (-4, 2), (1, -2), (4, 0), (2, 3)]);
    let labels = [0, 0, 0, 2, 2, 2];
    let observed = [
        Hull::build(0, p[..3].to_vec(), &mut GeometryWork::default())?,
        Hull::build(2, p[3..].to_vec(), &mut GeometryWork::default())?,
    ];
    for x in -5..=5 {
        for y in -4..=4 {
            if p.iter().any(|p| (p.x, p.y) == (x, y)) {
                continue;
            }
            let q = Point { x, y, state: 6 };
            let decision = exclusion::decide(&p, &labels, q, &[0, 1, 2], &mut Work::default())?;
            for candidate in [0, 1, 2] {
                let mut same: Vec<_> = p
                    .iter()
                    .zip(labels)
                    .filter(|(_, c)| *c == candidate)
                    .map(|(p, _)| *p)
                    .collect();
                same.push(q);
                let augmented = Hull::build(candidate, same, &mut GeometryWork::default())?;
                let mut expected = false;
                for other in observed.iter().filter(|h| h.label != candidate) {
                    expected |= regions::geometry::overlaps(
                        &augmented,
                        other,
                        &mut GeometryWork::default(),
                    )?;
                }
                assert_eq!(
                    decision
                        .exclusions
                        .iter()
                        .any(|ex| ex.candidate == candidate),
                    expected,
                    "query ({x},{y}) candidate {candidate}"
                );
            }
            for ex in &decision.exclusions {
                exclusion::verify(ex, &p, &labels, q, &mut Work::default())?;
            }
        }
    }
    Ok(())
}
