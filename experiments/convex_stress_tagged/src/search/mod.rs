//! Observed-only bounded CM25 proposal discovery. The same categorical policy
//! applies to every task. Full replay binds every guess, refusal and cost.
mod run;
use crate::{Model, Result, Sample, refinement, sketch, views};
pub use run::run;
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub minimum_strip_support: usize,
    pub max_views: usize,
    pub code_trials: usize,
    pub power_steps: usize,
    pub refinement_rounds: usize,
    pub seed_budget: usize,
    pub axis_budget: usize,
    pub seed: u32,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            minimum_strip_support: 8,
            max_views: 4,
            code_trials: 8,
            power_steps: 64,
            refinement_rounds: 24,
            seed_budget: 32,
            axis_budget: 1024,
            seed: 73,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Input {
    pub n: usize,
    pub vocab: usize,
    pub observations: Vec<[usize; 2]>,
    pub policy: Policy,
}
impl Input {
    pub fn samples(&self) -> Vec<Sample> {
        self.observations
            .iter()
            .map(|&[cell, label]| Sample { cell, label })
            .collect()
    }
    pub fn validate(&self) -> Result<u64> {
        crate::validate_observations(self.n, self.vocab, &self.samples())?;
        let p = &self.policy;
        if !(1..=257).contains(&p.minimum_strip_support)
            || !(1..=8).contains(&p.max_views)
            || !(1..=32).contains(&p.code_trials)
            || !(1..=128).contains(&p.power_steps)
            || !(1..=128).contains(&p.refinement_rounds)
            || p.seed_budget > 64
            || p.axis_budget > 2048
        {
            return Err("categorical search policy exceeds bounded views/trials/power/refinement/seeds/axes".into());
        }
        let (n, v) = (self.n as u64, self.vocab as u64);
        // Includes both a saved report and its replay: all possible strip views,
        // all power states, all coordinate states, candidate tables and raw audits.
        let views = 16 * n * n * (2 * n + 1);
        let power = p.max_views as u64
            * p.code_trials as u64
            * (32 * n * n + 128 * p.power_steps as u64 * n + 64 * v);
        let refinement =
            p.seed_budget as u64 * ((5 * p.refinement_rounds as u64 + 1) * 128 * n + 128 * n * n);
        let bound = 1024 * 1024 * 1024 + 2 * (views + power + refinement);
        if bound > 8 * 1024 * 1024 * 1024 {
            return Err("aggregate categorical search exceeds eight GiB forecast".into());
        }
        Ok(bound)
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SketchSlot {
    pub view: usize,
    pub trial: usize,
    pub reason: String,
    pub sketch: Option<sketch::Sketch>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    pub view: usize,
    pub trial: usize,
    pub q: usize,
    pub reason: String,
    pub refinement: Option<refinement::Outcome>,
    pub certificate_audit: Option<Value>,
    pub model: Option<Model>,
    pub audit: Option<Value>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub strip_observations: usize,
    pub sketches: usize,
    pub sketch_observations: usize,
    pub power_matvecs: usize,
    pub power_products: usize,
    pub decoder_observations: usize,
    pub decoder_comparisons: usize,
    pub seeds: usize,
    pub axis_steps: usize,
    pub axis_observation_comparisons: usize,
    pub coordinate_states: usize,
    pub raw_base_frequencies: usize,
    pub raw_full_frequencies: usize,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub input: Input,
    pub buffer_bound_bytes: u64,
    pub views: Vec<views::View>,
    pub sketches: Vec<SketchSlot>,
    pub attempts: Vec<Attempt>,
    pub work: Work,
    pub selected: Option<usize>,
}
pub fn replay(input: &Input, report: &Report) -> Result<()> {
    if &report.input != input || run(input.clone())? != *report {
        return Err("categorical search original-input/full-state/work replay mismatch".into());
    }
    Ok(())
}
