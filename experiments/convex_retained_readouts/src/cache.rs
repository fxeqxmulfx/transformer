//! Complete mean of last observed per-record signals on fixed learned states.
//! Unvisited signals are zero; stale signals are not the current full gradient.
use crate::{Result, coefficient, index};
use bilinear::train::{Mode, Record};
use joint::capacity::{self, Table};
use rotary::model::Model;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Serialize, Deserialize)]
pub struct Cache {
    pub content_sha256: String,
    pub identities: Vec<String>,
    pub signals: Vec<Vec<f64>>,
    pub seen: Vec<bool>,
    pub refreshed: usize,
    pub mean: Table,
}
impl Cache {
    pub fn new(model: &Model, records: &[Record<'_>]) -> Result<Self> {
        model.validate()?;
        if records.is_empty() || records.iter().any(|r| r.target >= model.objects) {
            return Err("invalid training cache records".into());
        }
        Ok(Self {
            content_sha256: readouts::source::content_hash(model)?,
            identities: records.iter().map(index::identity).collect(),
            signals: vec![vec![0.; model.objects]; records.len()],
            seen: vec![false; records.len()],
            refreshed: 0,
            mean: capacity::zero(model),
        })
    }
    pub fn validate(&self, model: &Model) -> Result<()> {
        model.validate()?;
        capacity::masses(model, &self.mean)?;
        if self.content_sha256 != readouts::source::content_hash(model)?
            || self.identities.is_empty()
            || self.identities.len() != self.signals.len()
            || self.seen.len() != self.signals.len()
            || self
                .signals
                .iter()
                .any(|s| s.len() != model.objects || s.iter().any(|x| !x.is_finite()))
        {
            return Err("cache content or complete signal shape changed".into());
        }
        Ok(())
    }
    pub fn observe(
        &mut self,
        model: &Model,
        ids: &[usize],
        records: &[Record<'_>],
        mode: Mode,
        h: f64,
    ) -> Result<coefficient::Batch> {
        self.validate(model)?;
        if ids.len() != records.len() || ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
            return Err("duplicate or incomplete occurrence indices".into());
        }
        for (&id, r) in ids.iter().zip(records) {
            if self.identities.get(id) != Some(&index::identity(r)) {
                return Err("record index identity changed".into());
            }
        }
        let b = coefficient::batch(model, records, mode, h)?;
        let n = self.signals.len() as f64;
        for ((&id, state), signal) in ids.iter().zip(&b.states).zip(&b.signals) {
            for (rows, s) in self.mean.iter_mut().zip(state) {
                for ((a, &new), &old) in rows.iter_mut().zip(signal).zip(&self.signals[id]) {
                    for k in 0..2 {
                        a[k] += s[k] * ((new - old) / n);
                    }
                }
            }
            self.signals[id].clone_from(signal);
            self.seen[id] = true;
            self.refreshed += 1;
        }
        Ok(b)
    }
    pub fn coverage(&self) -> usize {
        self.seen.iter().filter(|&&x| x).count()
    }

    /// Independent reconstruction from stored signals, with no new CE query.
    pub fn recompute(&self, model: &Model, records: &[Record<'_>]) -> Result<Table> {
        self.validate(model)?;
        if records.len() != self.identities.len() {
            return Err("cache replay count changed".into());
        }
        let mut sum = capacity::zero(model);
        for (i, r) in records.iter().enumerate() {
            if self.identities[i] != index::identity(r) {
                return Err("cache replay order changed".into());
            }
            if !self.seen[i] && self.signals[i].iter().any(|&x| x != 0.) {
                return Err("unvisited nonzero signal".into());
            }
            for (a, rows) in model.atoms.iter().zip(&mut sum) {
                let state = a.state(r.prefix)?;
                for (row, &signal) in rows.iter_mut().zip(&self.signals[i]) {
                    for k in 0..2 {
                        row[k] += state[k] * signal;
                    }
                }
            }
        }
        joint::coefficient::mean(&mut sum, records.len())?;
        Ok(sum)
    }
}
