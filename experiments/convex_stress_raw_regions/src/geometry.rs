//! CM48 at 71d96c93: input-space hull certificates, independent of the head.
//! Checked integer operations decide membership; floating-point tolerances
//! never grant coverage. Rational weights are converted only during inference.
use crate::{GeometryWork, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Point {
    pub x: i64,
    pub y: i64,
    pub state: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RationalWeight {
    pub state: usize,
    pub numerator: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Certificate {
    pub label: usize,
    pub denominator: u64,
    pub weights: Vec<RationalWeight>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hull {
    pub label: usize,
    pub vertices: Vec<Point>,
}

pub(crate) fn orientation(a: Point, b: Point, c: Point, w: &mut GeometryWork) -> Result<i128> {
    w.orientations += 1;
    let (bx, by) = (
        i128::from(b.x) - i128::from(a.x),
        i128::from(b.y) - i128::from(a.y),
    );
    let (cx, cy) = (
        i128::from(c.x) - i128::from(a.x),
        i128::from(c.y) - i128::from(a.y),
    );
    bx.checked_mul(cy)
        .and_then(|left| by.checked_mul(cx).and_then(|right| left.checked_sub(right)))
        .ok_or_else(|| "integer orientation overflow".into())
}

fn between(a: Point, b: Point, q: Point) -> bool {
    a.x.min(b.x) <= q.x && q.x <= a.x.max(b.x) && a.y.min(b.y) <= q.y && q.y <= a.y.max(b.y)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn certificate(label: usize, denominator: i128, terms: &[(Point, i128)]) -> Result<Certificate> {
    if denominator <= 0 || terms.iter().any(|(_, n)| *n < 0) {
        return Err("invalid barycentric sign".into());
    }
    let mut denominator = u64::try_from(denominator)?;
    let mut weights: Vec<_> = terms
        .iter()
        .filter(|(_, n)| *n != 0)
        .map(|(point, n)| {
            Ok(RationalWeight {
                state: point.state,
                numerator: u64::try_from(*n)?,
            })
        })
        .collect::<Result<_>>()?;
    weights.sort_by_key(|w| w.state);
    let divisor = weights.iter().fold(denominator, |d, w| gcd(d, w.numerator));
    denominator /= divisor;
    for weight in &mut weights {
        weight.numerator /= divisor;
    }
    Ok(Certificate {
        label,
        denominator,
        weights,
    })
}

impl Hull {
    pub fn build(label: usize, mut points: Vec<Point>, work: &mut GeometryWork) -> Result<Self> {
        if points.is_empty() {
            return Err("empty observed class".into());
        }
        work.sorted_points += points.len();
        points.sort_by_key(|p| (p.x, p.y, p.state));
        points.dedup_by_key(|p| (p.x, p.y));
        if points.len() == 1 {
            return Ok(Self {
                label,
                vertices: points,
            });
        }
        let chain = |items: Vec<Point>, work: &mut GeometryWork| -> Result<Vec<Point>> {
            let mut result = Vec::new();
            for p in items {
                while result.len() >= 2
                    && orientation(result[result.len() - 2], result[result.len() - 1], p, work)?
                        <= 0
                {
                    result.pop();
                }
                result.push(p);
            }
            result.pop();
            Ok(result)
        };
        let mut vertices = chain(points.clone(), work)?;
        vertices.extend(chain(points.into_iter().rev().collect(), work)?);
        Ok(Self { label, vertices })
    }

    pub fn certify(&self, query: Point, work: &mut GeometryWork) -> Result<Option<Certificate>> {
        work.membership_tests += 1;
        match self.vertices.as_slice() {
            [] => Err("empty hull".into()),
            [a] => {
                if (a.x, a.y) == (query.x, query.y) {
                    Ok(Some(certificate(self.label, 1, &[(*a, 1)])?))
                } else {
                    Ok(None)
                }
            }
            [a, b] => {
                if orientation(*a, *b, query, work)? != 0 || !between(*a, *b, query) {
                    return Ok(None);
                }
                let (av, bv, qv) = if a.x != b.x {
                    (a.x, b.x, query.x)
                } else {
                    (a.y, b.y, query.y)
                };
                let mut d = i128::from(bv) - i128::from(av);
                let mut n = i128::from(qv) - i128::from(av);
                if d < 0 {
                    d = -d;
                    n = -n;
                }
                Ok(Some(certificate(self.label, d, &[(*a, d - n), (*b, n)])?))
            }
            vertices => {
                for pair in vertices[1..].windows(2) {
                    let (a, b, c) = (vertices[0], pair[0], pair[1]);
                    let area = orientation(a, b, c, work)?;
                    if area <= 0 {
                        return Err("nonpositive canonical hull triangle".into());
                    }
                    let terms = [
                        (a, orientation(b, c, query, work)?),
                        (b, orientation(c, a, query, work)?),
                        (c, orientation(a, b, query, work)?),
                    ];
                    if terms.iter().all(|(_, n)| *n >= 0) {
                        return Ok(Some(certificate(self.label, area, &terms)?));
                    }
                }
                Ok(None)
            }
        }
    }
}

/// Independent exact input-space identity; does not trust hull order or containment.
pub fn verify(
    cert: &Certificate,
    query: Point,
    points: &[Point],
    labels: &[usize],
    work: &mut GeometryWork,
) -> Result<()> {
    work.certificates_verified += 1;
    if cert.denominator == 0
        || cert.weights.is_empty()
        || cert.weights.len() > 3
        || cert.weights.windows(2).any(|p| p[0].state >= p[1].state)
    {
        return Err("invalid rational certificate shape".into());
    }
    let (mut mass, mut x, mut y) = (0u128, 0i128, 0i128);
    for w in &cert.weights {
        let p = points
            .get(w.state)
            .ok_or("certificate state outside original block")?;
        if p.state != w.state || labels.get(w.state) != Some(&cert.label) || w.numerator == 0 {
            return Err("certificate source identity or label differs".into());
        }
        mass += u128::from(w.numerator);
        x = i128::from(w.numerator)
            .checked_mul(i128::from(p.x))
            .and_then(|v| x.checked_add(v))
            .ok_or("certificate x overflow")?;
        y = i128::from(w.numerator)
            .checked_mul(i128::from(p.y))
            .and_then(|v| y.checked_add(v))
            .ok_or("certificate y overflow")?;
    }
    let dx = i128::from(cert.denominator)
        .checked_mul(i128::from(query.x))
        .ok_or("query x overflow")?;
    let dy = i128::from(cert.denominator)
        .checked_mul(i128::from(query.y))
        .ok_or("query y overflow")?;
    if mass != u128::from(cert.denominator) || x != dx || y != dy {
        return Err("certificate does not reproduce the original input coordinate".into());
    }
    Ok(())
}

fn intersect(a: Point, b: Point, c: Point, d: Point, w: &mut GeometryWork) -> Result<bool> {
    let (ac, ad, ca, cb) = (
        orientation(a, b, c, w)?,
        orientation(a, b, d, w)?,
        orientation(c, d, a, w)?,
        orientation(c, d, b, w)?,
    );
    let opposite = |u: i128, v: i128| (u < 0 && v > 0) || (u > 0 && v < 0);
    Ok((opposite(ac, ad) && opposite(ca, cb))
        || (ac == 0 && between(a, b, c))
        || (ad == 0 && between(a, b, d))
        || (ca == 0 && between(c, d, a))
        || (cb == 0 && between(c, d, b)))
}

pub fn overlaps(a: &Hull, b: &Hull, work: &mut GeometryWork) -> Result<bool> {
    if a.vertices.is_empty() || b.vertices.is_empty() {
        return Err("empty hull in overlap check".into());
    }
    work.class_pairs += 1;
    for point in &a.vertices {
        if b.certify(*point, work)?.is_some() {
            return Ok(true);
        }
    }
    for point in &b.vertices {
        if a.certify(*point, work)?.is_some() {
            return Ok(true);
        }
    }
    for i in 0..a.vertices.len() {
        for j in 0..b.vertices.len() {
            if intersect(
                a.vertices[i],
                a.vertices[(i + 1) % a.vertices.len()],
                b.vertices[j],
                b.vertices[(j + 1) % b.vertices.len()],
                work,
            )? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
