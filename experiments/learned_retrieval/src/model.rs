//! CM111 with complete learned maps on opaque object IDs. Each adjacent pair
//! of entries stores the real and imaginary parts of one complex coordinate.
use crate::Result;
use baseline::random::Random;
use serde::{Deserialize, Serialize};

pub const WIDTH: usize = 8;
pub const DIM: usize = 2 * WIDTH;
pub type Row = [f64; DIM];

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn softmax(scores: &mut [f64]) {
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.;
    for x in &mut *scores {
        *x = (*x - maximum).exp();
        sum += *x;
    }
    for x in scores {
        *x /= sum;
    }
}

fn project(row: &mut [f64]) {
    let norm = dot(row, row).sqrt();
    if norm > 1. {
        for x in row {
            *x /= norm;
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub vocabulary: usize,
    pub context: usize,
    pub capacity: f64,
    pub weights: Vec<f64>,
}

pub struct Pass {
    pub attention: Vec<f64>,
    pub mass: Vec<f64>,
    pub state: Row,
    pub logits: Vec<f64>,
}

impl Model {
    fn checked_count(vocabulary: usize, context: usize) -> Result<usize> {
        if vocabulary == 0 || context < 2 {
            return Err("empty vocabulary or insufficient context".into());
        }
        let tables = vocabulary
            .checked_mul(2 * DIM)
            .ok_or("embedding size overflow")?;
        let offsets_and_scale = (context - 1)
            .checked_mul(2)
            .ok_or("context size overflow")?;
        tables
            .checked_add(offsets_and_scale)
            .ok_or_else(|| "model size overflow".into())
    }
    pub fn new(
        vocabulary: usize,
        context: usize,
        capacity: f64,
        radius: f64,
        key_scale: f64,
        seed: u32,
    ) -> Result<Self> {
        if vocabulary == 0
            || context < 2
            || !capacity.is_finite()
            || capacity <= 0.
            || !(0.0..=1.).contains(&radius)
            || !key_scale.is_finite()
        {
            return Err("invalid retrieval shape or initialization".into());
        }
        let mut model = Self {
            vocabulary,
            context,
            capacity,
            weights: vec![],
        };
        let count = Self::checked_count(vocabulary, context)?;
        model.weights.try_reserve_exact(count)?;
        model.weights.resize(count, 0.);
        let mut rng = Random::new(seed);
        let end = model.relative_start();
        for row in model.weights[..end].as_chunks_mut::<DIM>().0 {
            for x in &mut *row {
                *x = rng.gauss(0., 1.);
            }
            let norm = dot(row, row).sqrt();
            for x in row {
                *x *= radius / norm;
            }
        }
        let scale = model.scale_index();
        model.weights[scale] = key_scale;
        model.validate()?;
        Ok(model)
    }
    pub fn table_len(&self) -> usize {
        self.vocabulary * DIM
    }
    pub fn relative_start(&self) -> usize {
        2 * self.table_len()
    }
    pub fn scale_index(&self) -> usize {
        self.relative_start() + 2 * (self.context - 2) + 1
    }
    pub fn relative_index(&self, j: usize, k: usize) -> usize {
        self.relative_start() + (self.context - 2 + k - j)
    }
    pub fn scale(&self) -> f64 {
        self.weights[self.scale_index()]
    }
    pub fn key(&self, id: usize) -> &[f64] {
        &self.weights[id * DIM..(id + 1) * DIM]
    }
    pub fn value(&self, id: usize) -> &[f64] {
        let start = self.table_len() + id * DIM;
        &self.weights[start..start + DIM]
    }
    pub fn validate(&self) -> Result<()> {
        if self.weights.len() != Self::checked_count(self.vocabulary, self.context)?
            || !self.capacity.is_finite()
            || self.capacity <= 0.
            || self.weights.iter().any(|x| !x.is_finite())
            || self.weights[..self.relative_start()]
                .as_chunks::<DIM>()
                .0
                .iter()
                .any(|r| dot(r, r) > 1. + 2e-14)
        {
            return Err("invalid or infeasible retrieval parameters".into());
        }
        Ok(())
    }
    pub fn project_all(&mut self) {
        let end = self.relative_start();
        for row in self.weights[..end].as_chunks_mut::<DIM>().0 {
            project(row);
        }
    }
    pub fn perturb(&self, i: usize, step: f64) -> Self {
        let mut p = self.clone();
        p.weights[i] += step;
        if i < p.relative_start() {
            let start = i / DIM * DIM;
            project(&mut p.weights[start..start + DIM]);
        }
        p
    }
    pub fn objects(&self, raw: &[i64]) -> Result<Vec<usize>> {
        if raw.len() > self.context {
            return Err("prefix exceeds declared context".into());
        }
        raw.iter()
            .map(|&id| match usize::try_from(id) {
                Ok(o) if o < self.vocabulary => Ok(o),
                _ => Err("object outside vocabulary".into()),
            })
            .collect()
    }

    /// Row j is the complete normalized relative link from j to all k.
    pub fn links(&self, n: usize) -> Vec<f64> {
        let mut matrix = Vec::with_capacity(n * n);
        for j in 0..n {
            let start = matrix.len();
            matrix.extend((0..n).map(|k| self.weights[self.relative_index(j, k)]));
            softmax(&mut matrix[start..]);
        }
        matrix
    }
    pub fn pass(&self, prefix: &[usize], links: &[f64]) -> Pass {
        let n = prefix.len().saturating_sub(1);
        let mut attention = vec![];
        let mut mass = vec![0.; n];
        let mut state = [0.; DIM];
        if n > 0 {
            let query = self.key(prefix[n]);
            attention.extend(
                prefix[..n]
                    .iter()
                    .map(|&o| self.scale() * dot(query, self.key(o))),
            );
            softmax(&mut attention);
            for (a, row) in attention.iter().zip(links.chunks_exact(n)) {
                for (m, b) in mass.iter_mut().zip(row) {
                    *m += a * b;
                }
            }
            for (&w, &o) in mass.iter().zip(prefix) {
                for (s, v) in state.iter_mut().zip(self.value(o)) {
                    *s += w * v;
                }
            }
        }
        let logits = (0..self.vocabulary)
            .map(|c| self.capacity * dot(self.value(c), &state))
            .collect();
        Pass {
            attention,
            mass,
            state,
            logits,
        }
    }
    pub fn logits(&self, raw: &[i64]) -> Result<Vec<f64>> {
        let ids = self.objects(raw)?;
        let links = self.links(ids.len().saturating_sub(1));
        Ok(self.pass(&ids, &links).logits)
    }
}
