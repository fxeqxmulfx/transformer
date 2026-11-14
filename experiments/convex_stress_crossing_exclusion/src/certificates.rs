//! CM50 `excluded_of_barycenters` at2b17dc86. These exact rational identities
//! are independent of hull construction, vertex ordering and floating point.
use crate::*;

pub(crate) fn dirac(label: usize, state: usize) -> Certificate {
    Certificate {
        label,
        denominator: 1,
        weights: vec![RationalWeight {
            state,
            numerator: 1,
        }],
    }
}

/// The candidate side may use the tagged query; the opposite side may not.
/// Zero masses are omitted. At most three sources suffice for the planar search.
pub(crate) fn coordinates(
    cert: &Certificate,
    points: &[Point],
    labels: &[usize],
    query: Option<Point>,
) -> Result<(i128, i128)> {
    if cert.denominator == 0
        || cert.weights.is_empty()
        || cert.weights.len() > 3
        || cert.weights.windows(2).any(|w| w[0].state >= w[1].state)
    {
        return Err("invalid finite rational mixture shape".into());
    }
    let (mut mass, mut x, mut y) = (0u128, 0i128, 0i128);
    for w in &cert.weights {
        let p = if let Some(q) = query.filter(|q| q.state == w.state) {
            q
        } else {
            let p = *points
                .get(w.state)
                .ok_or("mixture source outside observations")?;
            if p.state != w.state || labels.get(w.state) != Some(&cert.label) {
                return Err("mixture source identity or raw label differs".into());
            }
            p
        };
        if w.numerator == 0 {
            return Err("zero mass must be omitted from the finite support".into());
        }
        mass += u128::from(w.numerator);
        x = i128::from(w.numerator)
            .checked_mul(i128::from(p.x))
            .and_then(|v| x.checked_add(v))
            .ok_or("rational x sum overflow")?;
        y = i128::from(w.numerator)
            .checked_mul(i128::from(p.y))
            .and_then(|v| y.checked_add(v))
            .ok_or("rational y sum overflow")?;
    }
    if mass != u128::from(cert.denominator) {
        return Err("finite mixture does not have unit mass".into());
    }
    Ok((x, y))
}

pub(crate) fn equal(a: i128, ad: u64, b: i128, bd: u64) -> Result<bool> {
    Ok(a.checked_mul(i128::from(bd))
        .ok_or("rational comparison overflow")?
        == b.checked_mul(i128::from(ad))
            .ok_or("rational comparison overflow")?)
}

fn cross(a: (i128, i128), b: (i128, i128)) -> Result<i128> {
    a.0.checked_mul(b.1)
        .and_then(|x| a.1.checked_mul(b.0).and_then(|y| x.checked_sub(y)))
        .ok_or_else(|| "edge determinant overflow".into())
}

fn delta(a: Point, b: Point) -> (i128, i128) {
    (
        i128::from(b.x) - i128::from(a.x),
        i128::from(b.y) - i128::from(a.y),
    )
}

fn segment(
    label: usize,
    a: Point,
    b: Point,
    numerator: i128,
    denominator: i128,
) -> Result<Certificate> {
    let gcd = |mut a: u128, mut b: u128| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let divisor = gcd(u128::try_from(numerator)?, u128::try_from(denominator)?);
    let denominator = u64::try_from(u128::try_from(denominator)? / divisor)?;
    let numerator = u64::try_from(u128::try_from(numerator)? / divisor)?;
    let mut weights: Vec<_> = [(a.state, denominator - numerator), (b.state, numerator)]
        .into_iter()
        .filter(|(_, n)| *n > 0)
        .map(|(state, numerator)| RationalWeight { state, numerator })
        .collect();
    weights.sort_by_key(|w| w.state);
    Ok(Certificate {
        label,
        denominator,
        weights,
    })
}

/// Exact nonparallel edge intersection. Collinear contact is found by the
/// separate vertex containment checks. Overflow refuses the construction.
pub(crate) fn intersection(
    candidate: usize,
    other: usize,
    a: Point,
    b: Point,
    c: Point,
    d: Point,
    work: &mut Work,
) -> Result<Option<Exclusion>> {
    work.edge_pair_tests += 1;
    let (r, s, q) = (delta(a, b), delta(c, d), delta(a, c));
    let mut denominator = cross(r, s)?;
    if denominator == 0 {
        return Ok(None);
    }
    let (mut t, mut u) = (cross(q, s)?, cross(q, r)?);
    if denominator < 0 {
        denominator = denominator.checked_neg().ok_or("edge sign overflow")?;
        t = t.checked_neg().ok_or("edge sign overflow")?;
        u = u.checked_neg().ok_or("edge sign overflow")?;
    }
    if t < 0 || t > denominator || u < 0 || u > denominator {
        return Ok(None);
    }
    Ok(Some(Exclusion {
        candidate,
        candidate_mixture: segment(candidate, a, b, t, denominator)?,
        other_mixture: segment(other, c, d, u, denominator)?,
    }))
}
