//! EM5 with the fixed numerical allowances from future_forcing/forecast.rs.
//! This is a conditional numerical bound; the guard is not a CUDA theorem.
use crate::{Result, evidence::require};
use probability_tube::encoder::{Caps, Shape, norm};
use std::ops::Range;
const CONE: f64 = 19. / 14.;
const MEMORY: f64 = 81. / 95.;
const RELATIVE: f64 = 1e-6;
pub fn blocks(shape: Shape) -> Vec<Range<usize>> {
    (0..shape.classes)
        .map(|c| shape.row(c))
        .chain([shape.gate(), shape.value(), shape.output()])
        .collect()
}
pub fn caps(shape: Shape, values: &[f64]) -> Result<Caps> {
    require(
        values.len() == shape.classes + 3 && values.iter().all(|v| v.is_finite() && *v >= 0.),
        "parameter cap shape/value",
    )?;
    Ok(Caps {
        rows: values[..shape.classes].to_vec(),
        gate: values[shape.classes],
        value: values[shape.classes + 1],
        output: values[shape.classes + 2],
    })
}
pub struct Tube {
    norm: Vec<f64>,
    ratio: Vec<f64>,
    dimension: Vec<f64>,
    radius: Vec<f64>,
    maximum: f64,
    max_ratio: f64,
    scale: f64,
    pub step: usize,
}
impl Tube {
    pub fn new(shape: Shape, p: &[f64], m: &[f64], v: &[f64], step: usize) -> Result<Self> {
        require(
            p.len() == shape.count() && m.len() == p.len() && v.len() == p.len(),
            "moment dimensions",
        )?;
        require(
            p.iter().chain(m).chain(v).all(|x| x.is_finite()) && v.iter().all(|x| *x >= 0.),
            "moment finiteness",
        )?;
        require(
            m.iter().zip(v).all(|(m, v)| *v > 0. || *m == 0.),
            "zero variance with nonzero moment",
        )?;
        let ratios: Vec<_> = m
            .iter()
            .zip(v)
            .map(|(m, v)| if *v == 0. { 0. } else { (m * m / v).next_up() })
            .collect();
        let max_ratio = ratios.iter().copied().fold(0., f64::max);
        require(max_ratio <= CONE * (1. + RELATIVE), "anchor moment cone")?;
        let blocks = blocks(shape);
        Ok(Self {
            norm: blocks.iter().map(|r| norm(&p[r.clone()])).collect(),
            ratio: blocks
                .iter()
                .map(|r| ratios[r.clone()].iter().sum())
                .collect(),
            dimension: blocks.iter().map(|r| r.len() as f64).collect(),
            radius: vec![0.; blocks.len()],
            maximum: p.iter().map(|x| x.abs()).fold(0., f64::max),
            max_ratio,
            scale: 1.,
            step,
        })
    }
    pub fn advance(&mut self, h: usize, eta: f64) {
        assert!(h > 0 && eta >= 0. && 0.1 * eta <= 1.);
        let retention = 1. - 0.1 * eta;
        let memory = MEMORY.powf(h as f64);
        let time = (self.step + h) as f64;
        let bias = (1. - 0.95f64.powf(time)).sqrt() / (1. - 0.9f64.powf(time));
        let max_direction = (memory * self.max_ratio + (1. - memory) * CONE).sqrt() * bias;
        let rounding =
            8. * f32::EPSILON as f64 * (1. + self.maximum + eta * max_direction * (1. + RELATIVE));
        self.maximum = retention * self.maximum + eta * max_direction * (1. + RELATIVE) + rounding;
        self.scale *= retention;
        for i in 0..self.radius.len() {
            let direction =
                (memory * self.ratio[i] + (1. - memory) * CONE * self.dimension[i]).sqrt() * bias;
            self.radius[i] = retention * self.radius[i]
                + eta * direction * (1. + RELATIVE)
                + self.dimension[i].sqrt() * rounding;
        }
    }
    pub fn displacement(&self) -> Vec<f64> {
        self.norm
            .iter()
            .zip(&self.radius)
            .map(|(n, r)| (1. - self.scale).abs() * n + r)
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn guarded_full_blocks_cover_independent_fp32_updates() {
        let shape = Shape {
            classes: 5,
            width: 4,
            hidden_pairs: 3,
        };
        let p: Vec<f32> = (0..shape.count())
            .map(|i| 0.3 * (i as f32 * 0.7).sin())
            .collect();
        let m: Vec<f32> = p.iter().map(|x| x * 0.01).collect();
        let v: Vec<f32> = p.iter().map(|x| 0.0001 + x * x).collect();
        let cv = |x: &[f32]| x.iter().map(|x| *x as f64).collect::<Vec<_>>();
        let (op, om, ov) = (cv(&p), cv(&m), cv(&v));
        let mut tube = Tube::new(shape, &op, &om, &ov, 9000).unwrap();
        let (mut p, mut m, mut v) = (p, m, v);
        let mut rates = vec![];
        for h in 1..=24 {
            let eta = if h % 2 == 0 { 1e-5f32 } else { 0.001f32 };
            rates.push(eta as f64);
            for i in 0..p.len() {
                let g = ((i * 17 + h * 31) % 101) as f32 / 100. - 0.5;
                m[i] = 0.1f32.mul_add(g - m[i], m[i]);
                v[i] = 0.05f32.mul_add(g * g, 0.95 * v[i]);
                p[i] = p[i] * (1. - eta * 0.1) - eta * m[i] / (v[i].sqrt() + 1e-8);
            }
            tube.advance(h, eta as f64);
            let caps = tube.displacement();
            let exact = probability_tube::adam::Checkpoint {
                parameters: &op,
                first: &om,
                second: &ov,
                step: 9000,
            }
            .motion(shape, &rates, 0.1);
            let exact: Vec<_> = exact
                .rows
                .into_iter()
                .chain([exact.gate, exact.value, exact.output])
                .collect();
            for (b, r) in blocks(shape).iter().enumerate() {
                let actual = r
                    .clone()
                    .map(|i| (p[i] as f64 - op[i]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!(actual <= caps[b] && exact[b] <= caps[b]);
            }
        }
    }
}
