//! Source equations from rotate90_complex b70811e, batched independently in FP64.
use crate::algebra::linear;
pub const P: usize = 97;
pub const W: usize = 128;
pub const E: usize = P * W;
pub const H: usize = 341;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    Swiglu,
    Diagonal,
    Product,
    Plain,
}
impl Arm {
    pub const ALL: [Self; 4] = [Self::Swiglu, Self::Diagonal, Self::Product, Self::Plain];
    pub fn name(self) -> &'static str {
        match self {
            Self::Swiglu => "real-swiglu",
            Self::Diagonal => "complex-diagonal",
            Self::Product => "complex-product-radial",
            Self::Plain => "complex-product-plain",
        }
    }
    pub fn hidden(self) -> usize {
        if self == Self::Swiglu { H } else { 2 * H }
    }
    pub fn count(self) -> usize {
        E + 3 * W * self.hidden()
    }
}

fn normalize(x: &mut [f64], width: usize) -> Vec<f64> {
    x.chunks_exact_mut(width)
        .map(|row| {
            let d = (row.iter().map(|v| v * v).sum::<f64>() / width as f64 + f32::EPSILON as f64)
                .sqrt();
            for v in row {
                *v /= d;
            }
            d
        })
        .collect()
}

pub struct Forward {
    pub states: Vec<f64>,
    pub raw: Vec<f64>,
    pub logits: Vec<f64>,
    pub residual: Vec<f64>,
    pub denominators: Vec<f64>,
}

pub fn forward(arm: Arm, p: &[f64], rows: &[u32]) -> Forward {
    assert_eq!(p.len(), arm.count());
    assert_eq!(rows.len() % 3, 0);
    let batch = rows.len() / 3;
    let h = arm.hidden();
    let mut x = Vec::with_capacity(batch * W);
    for row in rows.as_chunks::<3>().0 {
        for c in 0..W {
            x.push(
                p[row[0] as usize * W + c]
                    + if c % 2 == 0 {
                        -p[row[1] as usize * W + c + 1]
                    } else {
                        p[row[1] as usize * W + c - 1]
                    },
            );
        }
    }
    normalize(&mut x, W);
    let residual = linear(&x, batch, W, &p[..E], P);
    let mut inner = x.clone();
    normalize(&mut inner, W);
    let gv = linear(&inner, batch, W, &p[E..E + 2 * h * W], 2 * h);
    let mut hidden = vec![0.; batch * h];
    for r in 0..batch {
        let g = &gv[r * 2 * h..(r + 1) * 2 * h];
        if arm == Arm::Swiglu {
            for c in 0..h {
                hidden[r * h + c] = g[c] / (1. + (-g[c]).exp()) * g[c + h];
            }
        } else {
            for c in 0..h / 2 {
                let (a, b, u, v) = (g[c], g[c + h / 2], g[c + h], g[c + 3 * h / 2]);
                let gate = if arm == Arm::Plain {
                    1.
                } else {
                    1. / (1. + (-(a * a + b * b + f32::EPSILON as f64).sqrt()).exp())
                };
                let (re, im) = if arm == Arm::Diagonal {
                    (a * u, b * v)
                } else {
                    ((a * u - b * v) / 2f64.sqrt(), (a * v + b * u) / 2f64.sqrt())
                };
                hidden[r * h + c] = gate * re;
                hidden[r * h + c + h / 2] = gate * im;
            }
        }
    }
    let mut branch = linear(&hidden, batch, h, &p[E + 2 * h * W..], W);
    normalize(&mut branch, W);
    let states: Vec<_> = x.iter().zip(branch).map(|(x, b)| x + b).collect();
    let raw = linear(&states, batch, W, &p[..E], P);
    let mut normalized = states.clone();
    let denominators = normalize(&mut normalized, W);
    let logits = linear(&normalized, batch, W, &p[..E], P);
    Forward {
        states,
        raw,
        logits,
        residual,
        denominators,
    }
}
