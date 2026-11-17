//! Factored CM113 evaluation sums every one- and two-hop path without
//! constructing an n-by-n-by-n-by-n tensor. No target enters this module.
use crate::{
    Result,
    model::{Layout, Model, WIDTH, Wave},
};

pub fn softmax(row: &mut [f64]) {
    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut total = 0.;
    for x in row.iter_mut() {
        *x = (*x - max).exp();
        total += *x;
    }
    for x in row {
        *x /= total;
    }
}
fn left_multiply(x: &[f64], matrix: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.; n];
    for (&w, row) in x.iter().zip(matrix.chunks_exact(n)) {
        for (z, &a) in out.iter_mut().zip(row) {
            *z += w * a;
        }
    }
    out
}

pub struct Tables {
    pub layout: Layout,
    pub keys: Vec<Wave>,
    pub values: Vec<Wave>,
    pub phases: Vec<Wave>,
    pub start: Wave,
    pub similarity: Vec<f64>,
    affinity: Vec<f64>,
    pub links: Vec<Vec<f64>>,
}
impl Tables {
    pub fn new(m: &Model, lengths: &[usize]) -> Result<Self> {
        let l = m.validate()?;
        let waves = |xs: &[f64]| {
            xs.as_chunks::<WIDTH>()
                .0
                .iter()
                .map(|row| Wave::angles(row))
                .collect()
        };
        let mut s = Self {
            keys: waves(&m.weights[l.key.clone()]),
            values: waves(&m.weights[l.value.clone()]),
            phases: waves(&m.weights[l.phase.clone()]),
            start: Wave::angles(&m.weights[l.start.clone()]),
            similarity: vec![
                0.;
                m.vocabulary
                    .checked_mul(m.vocabulary)
                    .ok_or("key kernel overflow")?
            ],
            affinity: vec![0.; m.vocabulary * m.vocabulary],
            links: vec![vec![]; m.context],
            layout: l,
        };
        for o in 0..m.vocabulary {
            s.key_row(m, o);
        }
        for &n in lengths {
            if n >= m.context {
                return Err("memory length outside declared context".into());
            }
            if n > 0 && s.links[n].is_empty() {
                s.links[n] = vec![0.; n * n];
                s.link_rows(m, n);
            }
        }
        Ok(s)
    }
    fn key_row(&mut self, m: &Model, o: usize) {
        let t = m.weights[self.layout.scale];
        for v in 0..m.vocabulary {
            let (i, j) = (o * m.vocabulary + v, v * m.vocabulary + o);
            let s = self.keys[o].dot(&self.keys[v]) / WIDTH as f64;
            let k = (t * s - t.abs()).exp();
            self.similarity[i] = s;
            self.similarity[j] = s;
            self.affinity[i] = k;
            self.affinity[j] = k;
        }
    }
    fn link_rows(&mut self, m: &Model, n: usize) {
        for (j, row) in self.links[n].chunks_exact_mut(n).enumerate() {
            let i = self.layout.offset(j, 0);
            row.copy_from_slice(&m.weights[i..i + n]);
            softmax(row);
        }
    }
    /// Refresh only tables whose inputs changed on a feasible forward query.
    pub fn refresh(&mut self, m: &Model, i: usize) {
        let l = &self.layout;
        let a = m.weights[i];
        if l.key.contains(&i) {
            let at = i - l.key.start;
            self.keys[at / WIDTH].set(at % WIDTH, a);
            self.key_row(m, at / WIDTH);
        } else if l.value.contains(&i) {
            let at = i - l.value.start;
            self.values[at / WIDTH].set(at % WIDTH, a);
        } else if l.phase.contains(&i) {
            let at = i - l.phase.start;
            self.phases[at / WIDTH].set(at % WIDTH, a);
        } else if l.start.contains(&i) {
            self.start.set(i - l.start.start, a);
        } else if l.relation.contains(&i) {
            for n in 1..self.links.len() {
                if !self.links[n].is_empty() {
                    self.link_rows(m, n);
                }
            }
        } else if i == l.scale {
            for (k, &s) in self.affinity.iter_mut().zip(&self.similarity) {
                *k = (a * s - a.abs()).exp();
            }
        }
    }
    fn attention(&self, m: &Model, x: &[usize], q: usize) -> Vec<f64> {
        let n = x.len();
        if n == 1 {
            return vec![1.; 2];
        }
        let l = &self.layout;
        let t = m.weights[l.scale];
        let prior: Vec<_> = (0..n)
            .map(|j| {
                m.weights[l.bias.start + if j == 0 { m.vocabulary } else { x[j - 1] }]
                    + m.weights[l.recency] * j as f64
            })
            .collect();
        let peak = prior.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let positional: Vec<_> = prior.iter().map(|p| (p - peak).exp()).collect();
        let mut a = vec![0.; (n + 1) * n];
        for (o, row) in x.iter().copied().chain([q]).zip(a.chunks_exact_mut(n)) {
            for ((slot, &id), &w) in row.iter_mut().zip(x).zip(&positional) {
                *slot = self.affinity[o * m.vocabulary + id] * w;
            }
            let z: f64 = row.iter().sum();
            if z.is_normal() && row.iter().all(|&v| v.is_normal()) {
                for v in row {
                    *v /= z;
                }
            } else {
                for ((slot, &id), &p) in row.iter_mut().zip(x).zip(&prior) {
                    *slot = t * self.similarity[o * m.vocabulary + id] + p;
                }
                softmax(row);
            }
        }
        a
    }
    pub fn reads(&self, x: &[usize], one: &[f64], two: &[f64]) -> (Wave, Wave) {
        let (mut first, mut second) = (Wave::default(), Wave::default());
        for ((&o, &w1), &w2) in x.iter().zip(one).zip(two) {
            first.add(&self.values[o], w1);
            second.add(&self.values[o], w2);
        }
        (first, second)
    }
    pub fn radius(&self, m: &Model, id: usize) -> f64 {
        if m.learned_product_radii {
            m.weights[self.layout.radii.start + id]
        } else {
            1.
        }
    }
    pub fn scaled_values(&self, x: &[usize], one: &[f64], radii: &[f64]) -> Wave {
        let mut read = Wave::default();
        for (&id, &weight) in x.iter().zip(one) {
            read.add(&self.values[id], weight * radii[id]);
        }
        read
    }
    pub fn product_read(&self, m: &Model, x: &[usize], one: &[f64], s1: &Wave) -> Wave {
        if m.learned_product_radii {
            self.scaled_values(x, one, &m.weights[self.layout.radii.clone()])
        } else {
            *s1
        }
    }
    pub fn hidden(&self, m: &Model, q: usize, one: &Wave, two: &Wave, product_read: &Wave) -> Wave {
        let l = &self.layout;
        Wave::mix(
            &Wave::mix(one, two, m.weights[l.depth]),
            &product_read
                .times(&self.phases[q], false)
                .scaled(self.radius(m, q)),
            m.weights[l.product],
        )
    }
    pub fn pass(&self, m: &Model, prefix: &[usize]) -> Pass {
        let Some((&q, x)) = prefix.split_last() else {
            return Pass {
                hidden: self.start,
                ..Pass::default()
            };
        };
        let n = x.len();
        if n == 0 {
            return Pass {
                hidden: self.start.times(&self.phases[q], false),
                ..Pass::default()
            };
        }
        let attention = self.attention(m, x, q);
        let links = &self.links[n];
        let one = left_multiply(&attention[n * n..], links, n);
        let second_keys = left_multiply(&one, &attention[..n * n], n);
        let two = left_multiply(&second_keys, links, n);
        let (s1, s2) = self.reads(x, &one, &two);
        let product_read = self.product_read(m, x, &one, &s1);
        let hidden = self.hidden(m, q, &s1, &s2, &product_read);
        Pass {
            attention,
            one,
            second_keys,
            two,
            s1,
            s2,
            product_read,
            hidden,
        }
    }
    pub fn class_logit(&self, m: &Model, h: &Wave, c: usize) -> f64 {
        let z = m.capacity / WIDTH as f64 * self.values[c].dot(h);
        let offsets = &m.weights[self.layout.output_offsets.clone()];
        offsets.get(c).map_or(z, |b| z + b)
    }
    pub fn logits(&self, m: &Model, h: &Wave) -> Vec<f64> {
        let scale = m.capacity / WIDTH as f64;
        let mut z: Vec<_> = self.values.iter().map(|w| scale * w.dot(h)).collect();
        for (s, b) in z
            .iter_mut()
            .zip(&m.weights[self.layout.output_offsets.clone()])
        {
            *s += b;
        }
        z
    }
}
#[derive(Default)]
pub struct Pass {
    pub attention: Vec<f64>,
    pub one: Vec<f64>,
    pub second_keys: Vec<f64>,
    pub two: Vec<f64>,
    pub s1: Wave,
    pub s2: Wave,
    pub product_read: Wave,
    pub hidden: Wave,
}
