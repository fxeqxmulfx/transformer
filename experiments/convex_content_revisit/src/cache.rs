//! CM106 mean_refresh_both: replace old signal*state by new signal*state.
//! Every occurrence retains its historical states, signals and content version.
use crate::Result;
use bilinear::train::{Mode, Record};
use joint::capacity::{self, Table};
use retained::{coefficient, index};
use rotary::{atom::State, model::Model};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Serialize, Deserialize)]
pub struct Cache {
    pub identities: Vec<String>,
    pub signals: Vec<Vec<f64>>,
    pub states: Vec<Vec<State>>,
    pub origins: Vec<usize>,
    pub seen: Vec<bool>,
    pub refreshes: usize,
    pub starting_refreshes: usize,
    pub mean: Table,
}
impl Cache {
    pub fn from_parent(
        model: &Model,
        records: &[Record<'_>],
        parent: retained::cache::Cache,
    ) -> Result<Self> {
        parent.validate(model)?;
        if records.iter().map(index::identity).collect::<Vec<_>>() != parent.identities {
            return Err("parent cache occurrence identity changed".into());
        }
        let states = records
            .iter()
            .map(|r| {
                model
                    .atoms
                    .iter()
                    .map(|a| a.state(r.prefix))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let result = Self {
            origins: vec![0; records.len()],
            states,
            identities: parent.identities,
            signals: parent.signals,
            seen: parent.seen,
            refreshes: 0,
            starting_refreshes: parent.refreshed,
            mean: parent.mean,
        };
        result.validate(model)?;
        if result.recomputation_error(model)? > 1e-10 {
            return Err("parent mean differs from initial states/signals".into());
        }
        Ok(result)
    }
    pub fn validate(&self, model: &Model) -> Result<()> {
        model.validate()?;
        capacity::masses(model, &self.mean)?;
        let n = self.identities.len();
        if n == 0
            || self.signals.len() != n
            || self.states.len() != n
            || self.origins.len() != n
            || self.seen.len() != n
            || self
                .signals
                .iter()
                .any(|s| s.len() != model.objects || s.iter().any(|x| !x.is_finite()))
            || self.states.iter().any(|s| {
                s.len() != model.atoms.len()
                    || s.iter()
                        .flatten()
                        .any(|z| !z.re.is_finite() || !z.im.is_finite() || z.norm() > 1. + 2e-10)
            })
        {
            return Err("incomplete or unbounded historical cache".into());
        }
        if self.states.iter().any(|s| s[0] != [crate::C::default(); 2]) {
            return Err("structural zero cache state changed".into());
        }
        for (s, &seen) in self.signals.iter().zip(&self.seen) {
            if !seen && s.iter().any(|&x| x != 0.) {
                return Err("unvisited nonzero signal".into());
            }
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
        version: usize,
    ) -> Result<coefficient::Batch> {
        self.validate(model)?;
        if ids.len() != records.len()
            || ids.iter().collect::<BTreeSet<_>>().len() != ids.len()
            || self.origins.iter().any(|&v| v > version)
        {
            return Err("invalid historical occurrence/version indices".into());
        }
        for (&id, r) in ids.iter().zip(records) {
            if self.identities.get(id) != Some(&index::identity(r)) {
                return Err("record index identity changed".into());
            }
        }
        let b = coefficient::batch(model, records, mode, h)?;
        let n = self.signals.len() as f64;
        for ((&id, new_state), new_signal) in ids.iter().zip(&b.states).zip(&b.signals) {
            for ((rows, new), old) in self.mean.iter_mut().zip(new_state).zip(&self.states[id]) {
                for ((a, &new_s), &old_s) in rows.iter_mut().zip(new_signal).zip(&self.signals[id])
                {
                    for k in 0..2 {
                        a[k] += (new[k] * new_s - old[k] * old_s) / n;
                    }
                }
            }
            self.signals[id].clone_from(new_signal);
            self.states[id].clone_from(new_state);
            self.origins[id] = version;
            self.seen[id] = true;
            self.refreshes += 1;
        }
        Ok(b)
    }
    pub fn coverage(&self) -> usize {
        self.seen.iter().filter(|&&s| s).count()
    }
    pub fn recompute(&self, model: &Model) -> Result<Table> {
        self.validate(model)?;
        let mut sum = capacity::zero(model);
        for (states, signals) in self.states.iter().zip(&self.signals) {
            for (rows, state) in sum.iter_mut().zip(states) {
                for (a, &s) in rows.iter_mut().zip(signals) {
                    for k in 0..2 {
                        a[k] += state[k] * s;
                    }
                }
            }
        }
        joint::coefficient::mean(&mut sum, self.identities.len())?;
        Ok(sum)
    }
    pub fn recomputation_error(&self, model: &Model) -> Result<f64> {
        Ok(self
            .recompute(model)?
            .iter()
            .flatten()
            .flatten()
            .zip(self.mean.iter().flatten().flatten())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0f64, f64::max))
    }
}
