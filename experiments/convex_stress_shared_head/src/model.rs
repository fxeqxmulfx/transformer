//! Source routes and original record identities are fixed before fitting.
//! Only the complete realified complex tensor is optimized; old heads stay
//! external immutable references bound by their full artifact hashes.
use crate::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub parent_sha256: String,
    pub source_sha256: String,
    pub n: usize,
    pub vocabulary: usize,
    pub blocks: usize,
    pub route: Vec<usize>,
    pub records: Vec<Record>,
}
impl Input {
    pub fn from_source(
        source: &kernel::Model,
        records: Vec<Record>,
        parent_sha256: String,
    ) -> Result<Self> {
        source.bounds()?;
        convex_stress_shared_roles::records::encode(source, &records)?;
        let input = Self {
            parent_sha256,
            source_sha256: digest(source)?,
            n: source.n,
            vocabulary: source.vocabulary,
            blocks: source.blocks.len(),
            route: (0..source.n.pow(2))
                .map(|cell| source.location(cell).0)
                .collect(),
            records,
        };
        input.validate()?;
        Ok(input)
    }
    pub fn validate(&self) -> Result<()> {
        if !(2..=257).contains(&self.n)
            || !(self.n..=257).contains(&self.vocabulary)
            || !(1..=1024).contains(&self.blocks)
            || self.route.len() != self.n.pow(2)
            || self.route.iter().any(|&b| b >= self.blocks)
            || self.records.is_empty()
            || self.records.len() > 66_049
            || [&self.parent_sha256, &self.source_sha256]
                .iter()
                .any(|s| s.len() != 64 || !s.bytes().all(|c| c.is_ascii_hexdigit()))
        {
            return Err("invalid shared-head dimensions, route or source binding".into());
        }
        let mut ids = BTreeSet::new();
        let mut labels = BTreeMap::new();
        for r in &self.records {
            if r.cell >= self.route.len()
                || r.label >= self.vocabulary
                || !ids.insert(r.id)
                || labels
                    .insert(r.cell, r.label)
                    .is_some_and(|old| old != r.label)
            {
                return Err(
                    "invalid original record identity or contradictory raw observation".into(),
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub input: Input,
    pub protocol_sha256: String,
    pub coefficients: Vec<f64>,
}
impl Model {
    pub fn zero(input: Input) -> Result<Self> {
        input.validate()?;
        limits::forecast(&input, &training::Policy::forward())?;
        Ok(Self {
            coefficients: vec![0.; input.blocks * WIDTH * input.vocabulary],
            input,
            protocol_sha256: digest(&serde_json::from_str::<serde_json::Value>(PROTOCOL)?)?,
        })
    }
    pub fn validate(&self) -> Result<f64> {
        self.input.validate()?;
        if self.protocol_sha256 != digest(&serde_json::from_str::<serde_json::Value>(PROTOCOL)?)?
            || self.coefficients.len() != self.input.blocks * WIDTH * self.input.vocabulary
            || self.coefficients.iter().any(|x| !x.is_finite())
        {
            return Err("invalid complete shared complex tensor".into());
        }
        let mut masses = vec![0.; self.input.vocabulary];
        for row in self.coefficients.chunks_exact(self.input.vocabulary) {
            for (s, x) in masses.iter_mut().zip(row) {
                *s += x.abs();
            }
        }
        let largest = masses.into_iter().fold(0., f64::max);
        if largest > RADIUS * (1. + 1e-10) {
            return Err("global per-class coefficient budget exceeded".into());
        }
        Ok(largest)
    }
    pub fn project(&mut self) {
        project_columns(&mut self.coefficients, self.input.vocabulary, RADIUS);
    }
    pub fn logits_cell(&self, cell: usize) -> Result<Vec<f64>> {
        let block = *self
            .input
            .route
            .get(cell)
            .ok_or("input cell outside fixed route")?;
        Ok(self.logits_encoded(block, &features::encode(self.input.n, cell)))
    }
    pub fn logits_encoded(&self, block: usize, phi: &[C; STATES]) -> Vec<f64> {
        let v = self.input.vocabulary;
        let mut z = vec![0.; v];
        for (f, p) in phi.iter().enumerate() {
            let at = (block * WIDTH + 2 * f) * v;
            for (c, z) in z.iter_mut().enumerate() {
                *z += p.re * self.coefficients[at + c] - p.im * self.coefficients[at + v + c];
            }
        }
        z
    }
    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        if !(2..=4096).contains(&prefix.len())
            || prefix
                .iter()
                .any(|&t| t < 0 || t as usize >= self.input.vocabulary)
            || prefix[prefix.len() - 2..]
                .iter()
                .any(|&t| t as usize >= self.input.n)
        {
            return Err("prefix outside shared two-token domain".into());
        }
        self.logits_cell(
            prefix[prefix.len() - 2] as usize * self.input.n + prefix[prefix.len() - 1] as usize,
        )
    }
}
