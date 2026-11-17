//! Per-occurrence finite estimates, refreshed by exact old/new replacement.
//! This arithmetic identity supplies no unbiasedness or convergence assumption.
use crate::Result;
use bilinear::train::Record;
use serde::{Deserialize, Serialize};
#[derive(Clone, Serialize, Deserialize)]
pub struct Cache {
    pub identities: Vec<String>,
    pub values: Vec<Vec<f64>>,
    pub origins: Vec<Option<usize>>,
    pub mean: Vec<f64>,
    pub refreshes: usize,
}
impl Cache {
    pub fn new(records: &[Record<'_>], dimensions: usize) -> Result<Self> {
        if records.is_empty() || dimensions == 0 {
            return Err("empty estimate cache".into());
        }
        Ok(Self {
            identities: records.iter().map(retained::index::identity).collect(),
            values: vec![vec![0.; dimensions]; records.len()],
            origins: vec![None; records.len()],
            mean: vec![0.; dimensions],
            refreshes: 0,
        })
    }
    pub fn coverage(&self) -> usize {
        self.origins.iter().filter(|v| v.is_some()).count()
    }
    pub fn validate(&self) -> Result<()> {
        let n = self.identities.len();
        let d = self.mean.len();
        if n == 0
            || d == 0
            || self.values.len() != n
            || self.origins.len() != n
            || self
                .values
                .iter()
                .any(|v| v.len() != d || v.iter().any(|x| !x.is_finite()))
            || self.mean.iter().any(|x| !x.is_finite())
            || self.identities.iter().any(|s| s.len() != 64)
            || self.refreshes < self.coverage()
        {
            return Err("invalid complete estimate cache".into());
        }
        for (origin, v) in self.origins.iter().zip(&self.values) {
            if origin.is_none() && v.iter().any(|&x| x != 0.) {
                return Err("unobserved record has a cached estimate".into());
            }
        }
        Ok(())
    }
    pub fn observe(
        &mut self,
        ids: &[usize],
        records: &[Record<'_>],
        values: &[Vec<f64>],
        version: usize,
    ) -> Result<()> {
        if ids.len() != records.len() || ids.len() != values.len() || ids.is_empty() {
            return Err("estimate observation shape".into());
        }
        // Validate the whole observation before mutating any retained entry.
        for ((&id, r), v) in ids.iter().zip(records).zip(values) {
            if self.identities.get(id) != Some(&retained::index::identity(r))
                || v.len() != self.mean.len()
                || v.iter().any(|x| !x.is_finite())
                || self.origins[id].is_some_and(|old| old > version)
            {
                return Err("estimate identity, dimension or version mismatch".into());
            }
        }
        let n = self.identities.len() as f64;
        for (&id, v) in ids.iter().zip(values) {
            for ((mean, &new), &old) in self.mean.iter_mut().zip(v).zip(&self.values[id]) {
                *mean += (new - old) / n;
            }
            self.values[id] = v.clone();
            self.origins[id] = Some(version);
            self.refreshes += 1;
        }
        Ok(())
    }
    pub fn error(&self) -> Result<f64> {
        self.validate()?;
        let mut sum = vec![0.; self.mean.len()];
        for v in &self.values {
            for (s, &x) in sum.iter_mut().zip(v) {
                *s += x;
            }
        }
        Ok(sum
            .iter()
            .zip(&self.mean)
            .map(|(&s, &m)| (s / self.identities.len() as f64 - m).abs())
            .fold(0., f64::max))
    }
}
