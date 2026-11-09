//! Same scalar/complex bilinear algebra and nominal maxima as task #188.
//! Apply the three tightened incoming radii without dropping mixed terms.
use crate::{
    forecast_jet::{Product, norm},
    jet::Tight,
};
fn apply(kind: Product, a: &[f64], b: &[f64]) -> Vec<f64> {
    let h = a.len() / 2;
    match kind {
        Product::Real | Product::Diagonal => a.iter().zip(b).map(|(x, y)| x * y).collect(),
        Product::Gate => a.iter().enumerate().map(|(i, x)| x * b[i % h]).collect(),
        Product::Complex => (0..2 * h)
            .map(|i| {
                if i < h {
                    (a[i] * b[i] - a[i + h] * b[i + h]) / 2f64.sqrt()
                } else {
                    (a[i - h] * b[i] + a[i] * b[i - h]) / 2f64.sqrt()
                }
            })
            .collect(),
    }
}
fn max_block(x: &[f64], complex: bool) -> f64 {
    if complex {
        let h = x.len() / 2;
        (0..h).map(|i| x[i].hypot(x[i + h])).fold(0., f64::max)
    } else {
        x.iter().map(|v| v.abs()).fold(0., f64::max)
    }
}
pub fn product(kind: Product, a: &Tight, b: &Tight) -> Tight {
    let mut out = Tight::initial(kind.jets(&a.base, &b.base));
    let scale = if matches!(kind, Product::Complex) {
        1. / 2f64.sqrt()
    } else {
        1.
    };
    for q in 0..a.base.rows() {
        let av = &a.base.v[q * a.base.width..(q + 1) * a.base.width];
        let bv = &b.base.v[q * b.base.width..(q + 1) * b.base.width];
        out.defects[q] = norm(&apply(kind, av, bv));
        let an: Vec<_> = av
            .iter()
            .enumerate()
            .map(|(k, v)| a.base.a[q * a.base.width + k] + v)
            .collect();
        let bn: Vec<_> = bv
            .iter()
            .enumerate()
            .map(|(k, v)| b.base.a[q * b.base.width + k] + v)
            .collect();
        let am = max_block(&an, !matches!(kind, Product::Real));
        let bm = max_block(&bn, matches!(kind, Product::Complex | Product::Diagonal));
        for m in 0..4 {
            let ra = a.radii[m][q];
            let rb = b.radii[m][q];
            let transported = scale * (am * rb + bm * ra + ra * rb);
            out.gains[m][q] = if ra + rb > 0. {
                transported / (ra + rb)
            } else {
                0.
            };
            if m > 0 {
                out.radii[m][q] = out.defects[q] + transported;
            }
        }
    }
    out
}
