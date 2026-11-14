//! Geometry selected by the true label is diagnostic only. Affine membership
//! is not a new prediction or proof that a target class extends beyond its hull.
use super::*;
type Point = (i64, i64);
fn orient(a: Point, b: Point, c: Point) -> Result<i128> {
    let (x, y, u, v) = (
        i128::from(b.0) - i128::from(a.0),
        i128::from(b.1) - i128::from(a.1),
        i128::from(c.0) - i128::from(a.0),
        i128::from(c.1) - i128::from(a.1),
    );
    x.checked_mul(v)
        .and_then(|l| y.checked_mul(u).and_then(|r| l.checked_sub(r)))
        .ok_or_else(|| "diagnostic integer overflow".into())
}
pub(crate) fn describe(points: &[(Point, usize)], label: usize, query: Point) -> Result<Value> {
    let class: Vec<_> = points
        .iter()
        .filter(|(_, c)| *c == label)
        .map(|(p, _)| *p)
        .collect();
    let Some(&a) = class.first() else {
        return Ok(json!({"observed_points":0,"affine_rank":null,"query_in_affine_span":false}));
    };
    let Some(&b) = class.iter().find(|&&p| p != a) else {
        return Ok(
            json!({"observed_points":class.len(),"affine_rank":0,"query_in_affine_span":query==a}),
        );
    };
    let mut rank = 1;
    for &p in &class {
        if orient(a, b, p)? != 0 {
            rank = 2;
            break;
        }
    }
    if rank == 2 {
        return Ok(
            json!({"observed_points":class.len(),"affine_rank":2,"query_in_affine_span":true}),
        );
    }
    let mut signs = BTreeMap::<usize, (bool, bool, bool)>::new();
    for &(p, c) in points.iter().filter(|(_, c)| *c != label) {
        let z = orient(a, b, p)?;
        let s = signs.entry(c).or_default();
        s.0 |= z < 0;
        s.1 |= z > 0;
        s.2 |= z == 0;
    }
    let conflicts: Vec<_> = signs
        .into_iter()
        .filter(|(_, s)| s.2 || (s.0 && s.1))
        .map(|(c, _)| c)
        .collect();
    Ok(
        json!({"observed_points":class.len(),"affine_rank":1,"query_in_affine_span":orient(a,b,query)?==0,
        "line_intersects_other_observed_class_hulls":conflicts}),
    )
}
