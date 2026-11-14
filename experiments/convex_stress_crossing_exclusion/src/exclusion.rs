//! CM50 finite hull conflicts at2b17dc86, extending CM49 at26ec2faa. Two
//! independently verified mixtures replace an integer observed witness point.
use crate::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Exclusion {
    pub candidate: usize,
    pub candidate_mixture: Certificate,
    pub other_mixture: Certificate,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Decision {
    pub survivors: Vec<usize>,
    pub exclusions: Vec<Exclusion>,
}

fn input(points: &[Point], labels: &[usize], query: Point, pool: &[usize]) -> Result<()> {
    if points.len() != labels.len()
        || points.iter().enumerate().any(|(i, p)| i != p.state)
        || query.state != points.len()
        || pool.is_empty()
        || pool.windows(2).any(|p| p[0] >= p[1])
        || points.iter().any(|p| (p.x, p.y) == (query.x, query.y))
    {
        return Err("invalid intersection points, query identity or candidate pool".into());
    }
    Ok(())
}

/// Exact CM50 hypothesis checks, including the strictly positive query role.
pub fn verify(
    ex: &Exclusion,
    points: &[Point],
    labels: &[usize],
    query: Point,
    work: &mut Work,
) -> Result<()> {
    input(points, labels, query, &[ex.candidate])?;
    let (a, b) = (&ex.candidate_mixture, &ex.other_mixture);
    if a.label != ex.candidate
        || b.label == ex.candidate
        || !a
            .weights
            .iter()
            .any(|w| w.state == query.state && w.numerator > 0)
    {
        return Err("intersection has invalid class or query role".into());
    }
    let (ax, ay) = certificates::coordinates(a, points, labels, Some(query))?;
    let (bx, by) = certificates::coordinates(b, points, labels, None)?;
    if !certificates::equal(ax, a.denominator, bx, b.denominator)?
        || !certificates::equal(ay, a.denominator, by, b.denominator)?
    {
        return Err("two mixtures do not meet at the same exact coordinate".into());
    }
    work.geometry.certificates_verified += 2;
    work.exclusion_certificates += 1;
    Ok(())
}

fn conflict(a: &Hull, b: &Hull, query: Point, work: &mut Work) -> Result<Option<Exclusion>> {
    work.class_pair_tests += 1;
    for p in &b.vertices {
        if let Some(candidate_mixture) = a.certify(*p, &mut work.geometry)? {
            return Ok(Some(Exclusion {
                candidate: a.label,
                candidate_mixture,
                other_mixture: certificates::dirac(b.label, p.state),
            }));
        }
    }
    // The only additional candidate vertex is the query. Original hulls have
    // been checked disjoint, so no other candidate vertex can be inside b.
    if let Some(other_mixture) = b.certify(query, &mut work.geometry)? {
        return Ok(Some(Exclusion {
            candidate: a.label,
            candidate_mixture: certificates::dirac(a.label, query.state),
            other_mixture,
        }));
    }
    let edges = |h: &Hull| match h.vertices.len() {
        0 | 1 => 0,
        2 => 1,
        n => n,
    };
    for i in 0..edges(a) {
        for j in 0..edges(b) {
            if let Some(ex) = certificates::intersection(
                a.label,
                b.label,
                a.vertices[i],
                a.vertices[(i + 1) % a.vertices.len()],
                b.vertices[j],
                b.vertices[(j + 1) % b.vertices.len()],
                work,
            )? {
                return Ok(Some(ex));
            }
        }
    }
    Ok(None)
}

pub fn decide(
    points: &[Point],
    labels: &[usize],
    query: Point,
    pool: &[usize],
    work: &mut Work,
) -> Result<Decision> {
    input(points, labels, query, pool)?;
    let observed: std::collections::BTreeSet<_> = labels.iter().copied().collect();
    let hulls: Vec<_> = observed
        .iter()
        .map(|&label| {
            Hull::build(
                label,
                points
                    .iter()
                    .zip(labels)
                    .filter(|(_, y)| **y == label)
                    .map(|(p, _)| *p)
                    .collect(),
                &mut work.geometry,
            )
        })
        .collect::<Result<_>>()?;
    for (i, a) in hulls.iter().enumerate() {
        for b in &hulls[i + 1..] {
            if regions::geometry::overlaps(a, b, &mut work.geometry)? {
                return Err("original observed class hulls already intersect".into());
            }
        }
    }
    let mut out = Decision {
        survivors: vec![],
        exclusions: vec![],
    };
    for &candidate in pool {
        work.candidate_tests += 1;
        let mut same: Vec<_> = points
            .iter()
            .zip(labels)
            .filter(|(_, y)| **y == candidate)
            .map(|(p, _)| *p)
            .collect();
        same.push(query);
        let augmented = Hull::build(candidate, same, &mut work.geometry)?;
        let mut exclusion = None;
        for other in hulls.iter().filter(|h| h.label != candidate) {
            if let Some(ex) = conflict(&augmented, other, query, work)? {
                verify(&ex, points, labels, query, work)?;
                exclusion = Some(ex);
                break;
            }
        }
        if let Some(ex) = exclusion {
            out.exclusions.push(ex);
        } else {
            out.survivors.push(candidate);
        }
    }
    Ok(out)
}

/// Verifies complete alternative exclusions and independent prediction weights.
pub fn verify_row(
    row: &Row,
    points: &[Point],
    labels: &[usize],
    query: Point,
    work: &mut Work,
) -> Result<()> {
    input(points, labels, query, &row.pool)?;
    let expected_pool: std::collections::BTreeSet<_> = labels.iter().copied().collect();
    if row.pool != expected_pool.into_iter().collect::<Vec<_>>()
        || !row.pool.contains(&row.label)
        || row.pool.len() < 2
        || row
            .exclusions
            .iter()
            .map(|ex| ex.candidate)
            .collect::<Vec<_>>()
            != row
                .pool
                .iter()
                .copied()
                .filter(|&c| c != row.label)
                .collect::<Vec<_>>()
    {
        return Err("row omits an observed candidate or an alternative exclusion".into());
    }
    for ex in &row.exclusions {
        verify(ex, points, labels, query, work)?;
    }
    let states: Vec<_> = labels
        .iter()
        .enumerate()
        .filter(|(_, label)| **label == row.label)
        .map(|(i, _)| i)
        .collect();
    if states.is_empty()
        || row.denominator != u64::try_from(states.len())?
        || row.weights
            != states
                .into_iter()
                .map(|state| RationalWeight {
                    state,
                    numerator: 1,
                })
                .collect::<Vec<_>>()
    {
        return Err("row changed the fixed positive same-label feature mixture".into());
    }
    work.rows_verified += 1;
    Ok(())
}
