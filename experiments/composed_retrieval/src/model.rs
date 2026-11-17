//! Opaque objects address learned angles. Signed offsets address a learned
//! scalar table. The only constrained weights are the two convex gates (CM113).
use crate::Result;
use baseline::random::Random;
use serde::{Deserialize, Serialize};
use std::{f64::consts::PI, ops::Range};

pub const WIDTH: usize = 8;

pub fn sigmoid(z: f64) -> f64 {
    if z >= 0. {
        1. / (1. + (-z).exp())
    } else {
        let e = z.exp();
        e / (1. + e)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Model {
    pub vocabulary: usize,
    pub context: usize,
    pub capacity: f64,
    pub weights: Vec<f64>,
}

#[derive(Clone)]
pub struct Layout {
    pub key: Range<usize>,
    pub value: Range<usize>,
    pub phase: Range<usize>,
    pub start: Range<usize>,
    pub bias: Range<usize>,
    pub relation: Range<usize>,
    pub scale: usize,
    pub recency: usize,
    pub depth: usize,
    pub product: usize,
}

impl Layout {
    pub fn new(vocab: usize, context: usize) -> Result<Self> {
        if vocab == 0 || context < 2 {
            return Err("empty vocabulary or insufficient context".into());
        }
        let table = vocab.checked_mul(WIDTH).ok_or("table size overflow")?;
        let offsets = (context - 2)
            .checked_mul(2)
            .and_then(|v| v.checked_add(1))
            .ok_or("offset size overflow")?;
        let mut end = 0usize;
        let mut range = |length| -> Result<Range<usize>> {
            let first = end;
            end = end.checked_add(length).ok_or("model size overflow")?;
            Ok(first..end)
        };
        let key = range(table)?;
        let value = range(table)?;
        let phase = range(table)?;
        let start = range(WIDTH)?;
        let bias = range(vocab + 1)?;
        let relation = range(offsets)?;
        let scalars = range(4)?;
        Ok(Self {
            key,
            value,
            phase,
            start,
            bias,
            relation,
            scale: scalars.start,
            recency: scalars.start + 1,
            depth: scalars.start + 2,
            product: scalars.start + 3,
        })
    }
    pub fn offset(&self, j: usize, k: usize) -> usize {
        self.relation.start + self.relation.len() / 2 + k - j
    }
    pub fn project(&self, i: usize, weight: f64) -> f64 {
        if i == self.depth || i == self.product {
            weight.clamp(0., 1.)
        } else {
            weight
        }
    }
}

impl Model {
    pub fn new(
        vocabulary: usize,
        context: usize,
        p: &crate::data::Protocol,
        seed: u32,
    ) -> Result<Self> {
        let l = Layout::new(vocabulary, context)?;
        let mut rng = Random::new(seed);
        let mut weights = vec![0.; l.product + 1];
        for x in &mut weights[..l.start.end] {
            *x = (2. * rng.random() - 1.) * PI;
        }
        weights[l.scale] = p.initial_key_scale;
        weights[l.depth] = p.initial_depth;
        weights[l.product] = p.initial_product;
        let model = Self {
            vocabulary,
            context,
            capacity: p.capacity,
            weights,
        };
        model.validate()?;
        Ok(model)
    }
    pub fn validate(&self) -> Result<Layout> {
        let l = Layout::new(self.vocabulary, self.context)?;
        if self.weights.len() != l.product + 1
            || !self.capacity.is_finite()
            || self.capacity < 0.
            || self.weights.iter().any(|w| !w.is_finite())
            || [l.depth, l.product]
                .iter()
                .any(|&i| !(0. ..=1.).contains(&self.weights[i]))
        {
            return Err("invalid physical model".into());
        }
        Ok(l)
    }
    pub fn objects(&self, prefix: &[i64]) -> Result<Vec<usize>> {
        if prefix.len() > self.context {
            return Err("prefix exceeds declared context".into());
        }
        prefix
            .iter()
            .map(|&x| {
                let o = usize::try_from(x)?;
                if o >= self.vocabulary {
                    return Err("object outside original vocabulary".into());
                }
                Ok(o)
            })
            .collect()
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Wave {
    pub re: [f64; WIDTH],
    pub im: [f64; WIDTH],
}

impl Wave {
    pub fn angles(xs: &[f64]) -> Self {
        let mut v = Self::default();
        for (d, &theta) in xs.iter().enumerate() {
            v.set(d, theta);
        }
        v
    }
    pub fn set(&mut self, d: usize, theta: f64) {
        (self.im[d], self.re[d]) = theta.sin_cos();
    }
    #[inline]
    pub fn dot(&self, b: &Self) -> f64 {
        (0..WIDTH)
            .map(|d| self.re[d] * b.re[d] + self.im[d] * b.im[d])
            .sum()
    }
    #[inline]
    pub fn add(&mut self, b: &Self, weight: f64) {
        for d in 0..WIDTH {
            self.re[d] += weight * b.re[d];
            self.im[d] += weight * b.im[d];
        }
    }
    pub fn mix(a: &Self, b: &Self, t: f64) -> Self {
        let mut v = Self::default();
        v.add(a, 1. - t);
        v.add(b, t);
        v
    }
    pub fn times(&self, b: &Self, conjugate: bool) -> Self {
        let mut v = Self::default();
        for d in 0..WIDTH {
            let y = if conjugate { -b.im[d] } else { b.im[d] };
            v.re[d] = self.re[d] * b.re[d] - self.im[d] * y;
            v.im[d] = self.im[d] * b.re[d] + self.re[d] * y;
        }
        v
    }
    pub fn angle_gradient(&self, adj: &Self, g: &mut [f64], weight: f64) {
        for (d, entry) in g.iter_mut().enumerate() {
            *entry += weight * (-self.im[d] * adj.re[d] + self.re[d] * adj.im[d]);
        }
    }
}
