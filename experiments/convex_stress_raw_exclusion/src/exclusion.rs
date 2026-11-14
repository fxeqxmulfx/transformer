//! Port of CM49 excluded_of_observed_point and target_eq_of_exclusions at26ec2faa.
//! Each certificate is verified without trusting hull order or float tolerances.
//! This sufficient observed-point test need not find every possible hull conflict.
use crate::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Exclusion {
    pub candidate: usize,
    pub witness_state: usize,
    pub certificate: Certificate,
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
        return Err("invalid exclusion points, query identity or candidate pool".into());
    }
    Ok(())
}

/// The query is an explicitly tagged additional candidate-class source. Its
/// positive mass proves a conflict with a different original observed class.
pub fn verify(
    ex: &Exclusion,
    points: &[Point],
    labels: &[usize],
    query: Point,
    work: &mut Work,
) -> Result<()> {
    input(points, labels, query, &[ex.candidate])?;
    let witness = *points
        .get(ex.witness_state)
        .ok_or("exclusion witness outside observations")?;
    if labels[ex.witness_state] == ex.candidate
        || ex.certificate.label != ex.candidate
        || !ex
            .certificate
            .weights
            .iter()
            .any(|w| w.state == query.state && w.numerator > 0)
    {
        return Err("invalid exclusion witness label or query role".into());
    }
    let mut augmented = points.to_vec();
    augmented.push(query);
    let mut augmented_labels = labels.to_vec();
    augmented_labels.push(ex.candidate);
    parent::geometry::verify(
        &ex.certificate,
        witness,
        &augmented,
        &augmented_labels,
        &mut work.geometry,
    )?;
    work.exclusion_certificates += 1;
    Ok(())
}

pub fn decide(
    points: &[Point],
    labels: &[usize],
    query: Point,
    pool: &[usize],
    work: &mut Work,
) -> Result<Decision> {
    input(points, labels, query, pool)?;
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
        let hull = Hull::build(candidate, same, &mut work.geometry)?;
        let mut exclusion = None;
        for (&p, &label) in points.iter().zip(labels) {
            if label == candidate {
                continue;
            }
            if let Some(certificate) = hull.certify(p, &mut work.geometry)? {
                let ex = Exclusion {
                    candidate,
                    witness_state: p.state,
                    certificate,
                };
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

/// Verifies every alternative's positive conflict and the separate prediction
/// weights. It never treats the input barycentric weights as feature weights.
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
