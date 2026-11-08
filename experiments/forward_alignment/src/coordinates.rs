//! Pullback and owner coordinates for the source's constrained real expansion.
//! Norms use 143360 independent coordinates; scalar pairings agree with the
//! expanded derivative. A gradient is pulled back, never merely sliced.
use crate::domain::{
    data::Random,
    model::{self, Arm, E, H, W},
};
use serde_json::{Value, json};
use std::ops::Range;

pub const COUNT: usize = 143360;
pub fn groups() -> [(&'static str, Range<usize>); 4] {
    [
        ("E", 0..E),
        ("A", E..E + 2 * H * W),
        ("B", E + 2 * H * W..COUNT),
        ("all", 0..COUNT),
    ]
}
pub fn expanded_groups(arm: Arm) -> [(&'static str, Range<usize>); 4] {
    let b = E + 2 * arm.hidden() * W;
    [
        ("E", 0..E),
        ("A", E..b),
        ("B", b..arm.count()),
        ("all", 0..arm.count()),
    ]
}
pub fn pullback(arm: Arm, g: &[f64]) -> Vec<f64> {
    assert_eq!(g.len(), arm.count());
    if arm == Arm::Swiglu {
        return g.to_vec();
    }
    model::pairs()
        .into_iter()
        .flat_map(|[ar, bi, br, ai, kind]| {
            if kind == 1 {
                [g[ar as usize], g[bi as usize]]
            } else {
                [
                    g[ar as usize] + g[ai as usize],
                    g[bi as usize] - g[br as usize],
                ]
            }
        })
        .collect()
}
pub fn owners(arm: Arm, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), arm.count());
    if arm == Arm::Swiglu {
        return x.to_vec();
    }
    model::pairs()
        .into_iter()
        .flat_map(|[ar, bi, _, _, _]| [x[ar as usize], x[bi as usize]])
        .collect()
}
pub fn expand(arm: Arm, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), COUNT);
    if arm == Arm::Swiglu {
        return x.to_vec();
    }
    let mut result = vec![0.; arm.count()];
    for ([ar, bi, br, ai, kind], v) in model::pairs().iter().zip(x.as_chunks::<2>().0) {
        result[*ar as usize] = v[0];
        result[*bi as usize] = v[1];
        if *kind == 0 {
            result[*ai as usize] = v[0];
            result[*br as usize] = -v[1];
        }
    }
    result
}
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
pub fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
pub fn direction(group: usize, seed: u64) -> Vec<f64> {
    let mut rng = Random(seed);
    let mut d = vec![0.; COUNT];
    for i in groups()[group].1.clone() {
        d[i] = if rng.next() & 1 == 0 { 1. } else { -1. };
    }
    let n = norm(&d);
    for x in &mut d {
        *x /= n;
    }
    d
}
pub fn compare(g: &[f64], d: &[f64], displacement: bool) -> Value {
    let mut out = json!({});
    for (name, range) in groups() {
        let (g, d) = (&g[range.clone()], &d[range]);
        let pairing = dot(g, d);
        let gn = norm(g);
        let dn = norm(d);
        let cosine = if gn * dn > 0. {
            Some(if displacement { -pairing } else { pairing } / (gn * dn))
        } else {
            None
        };
        let class = match cosine {
            Some(c) if c > 1e-3 => "descent",
            Some(c) if c < -1e-3 => "ascent",
            Some(_) => "near-orthogonal",
            None => "zero-norm",
        };
        out[name] = json!({"pairing":pairing,"gradient_norm":gn,"direction_norm":dn,"cosine":cosine,"classification":class});
    }
    out
}
pub fn pairing_check(arm: Arm, g: &[f64], d: &[f64]) -> Value {
    let gi = pullback(arm, g);
    let di = owners(arm, d);
    let mut rows = vec![];
    for ((name, a), (_, b)) in groups().into_iter().zip(expanded_groups(arm)) {
        let independent = dot(&gi[a.clone()], &di[a]);
        let expanded = dot(&g[b.clone()], &d[b]);
        let error = (independent - expanded).abs();
        let tolerance = 1e-12 + 1e-9 * independent.abs().max(expanded.abs());
        rows.push(json!({"group":name,"independent":independent,"expanded":expanded,"error":error,"tolerance":tolerance,"passed":error.is_finite()&&error<=tolerance}));
    }
    json!({"passed":rows.iter().all(|r|r["passed"]==true),"groups":rows})
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_pullback_preserves_pairing_and_owner_roundtrip() {
        for arm in Arm::ALL {
            let d = direction(3, 19);
            let expanded = expand(arm, &d);
            assert_eq!(owners(arm, &expanded), d);
            let g: Vec<_> = (0..arm.count()).map(|i| (i % 13) as f64 - 6.).collect();
            assert_eq!(pullback(arm, &g).len(), COUNT);
            assert_eq!(pairing_check(arm, &g, &expanded)["passed"], true);
        }
    }
    #[test]
    fn descent_sign_and_zero_norm_are_not_confused() {
        let g = direction(0, 3);
        let d: Vec<_> = g.iter().map(|v| -3. * v).collect();
        assert!((compare(&g, &d, true)["E"]["cosine"].as_f64().unwrap() - 1.).abs() < 1e-12);
        assert_eq!(compare(&g, &d, true)["E"]["classification"], "descent");
        assert_eq!(compare(&g, &g, true)["E"]["classification"], "ascent");
        assert!(compare(&g, &d, true)["B"]["cosine"].is_null());
    }
}
