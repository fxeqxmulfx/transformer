//! CM25 categorical strip views (84cacc46) around CM27 checked code discovery
//! (a14b416c). Only observed class identities enter either constructor. The
//! wrapper retains every proposed, skipped, refused and duplicate view.
use crate::{Model, Result, coding};
use convex_stress_tagged::views;
use serde::{Deserialize, Serialize};
use serde_json::Value;
mod audit;
mod run;
pub use audit::check;
pub use run::run;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Input {
    pub categorical: coding::Input,
    pub minimum_strip_support: usize,
    pub max_views: usize,
}
impl Input {
    pub fn validate(&self) -> Result<u64> {
        let bound = self.categorical.validate()?;
        if !(1..=257).contains(&self.minimum_strip_support) || self.max_views > 4 {
            return Err("strip support or view policy exceeds declared bound".into());
        }
        let total = 2 * 1024 * 1024 * 1024 + self.max_views as u64 * bound;
        if total > 48 * 1024 * 1024 * 1024 {
            return Err("strip report/check/replay forecast exceeds48GiB".into());
        }
        Ok(total)
    }
    fn samples(&self) -> Vec<crate::Sample> {
        self.categorical
            .observations
            .iter()
            .map(|&[cell, label]| crate::Sample { cell, label })
            .collect()
    }
    fn inner(&self, v: &views::View) -> coding::Input {
        coding::Input {
            n: v.n,
            vocab: self.categorical.vocab,
            observations: v.observations.clone(),
            policy: self.categorical.policy.clone(),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    ViewCap,
    EmptyInterior,
    Refused,
    Duplicate { candidate: usize },
    Accepted { candidate: usize },
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    pub view: views::View,
    pub coding: Option<coding::Report>,
    pub audit: Option<Value>,
    pub decision: Decision,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub view: usize,
    pub model: Model,
    pub audit: Value,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub views_proposed: usize,
    pub views_attempted: usize,
    pub finite_stages: usize,
    pub finite_scalar_updates: u64,
    pub finite_row_operations: usize,
    pub interior_raw_frequencies: usize,
    pub full_raw_frequencies: usize,
    pub ce_gradients: usize,
}
impl Work {
    fn charge(&mut self, r: &coding::Report) {
        self.views_attempted += 1;
        self.finite_stages += r.work.finite_stages;
        self.finite_scalar_updates += r.work.finite_scalar_updates;
        self.finite_row_operations += r.work.finite_row_operations;
        self.interior_raw_frequencies += r.work.raw_frequencies;
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub input: Input,
    pub buffer_bound_bytes: u64,
    pub attempts: Vec<Attempt>,
    pub candidates: Vec<Candidate>,
    pub selected: Option<usize>,
    pub reason: String,
    pub work: Work,
}
pub fn replay(input: &Input, out: &Report) -> Result<()> {
    if &out.input != input || run(input.clone())? != *out {
        return Err("composite strip original-input/full-state/work replay differs".into());
    }
    Ok(())
}
