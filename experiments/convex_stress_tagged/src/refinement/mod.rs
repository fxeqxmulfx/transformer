//! CM25 / Lean 84cacc46: independent phase choices certify an optimal update
//! of one axis. Phases may repeat and n need not equal q. This is bounded
//! coordinate descent, not a joint global optimizer or an unseen-label proof.
mod audit;
mod certificate;
mod run;
use crate::{Model, Result, Sample};
pub use audit::check;
pub use run::{replay, run};
use serde::{Deserialize, Serialize};
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Input {
    pub seed: Model,
    pub observations: Vec<[usize; 2]>,
    pub rounds: usize,
    pub axis_budget: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct State {
    pub rows: Vec<usize>,
    pub columns: Vec<usize>,
    pub decoder: Vec<usize>,
    pub errors: usize,
    pub unseen_latents: Vec<usize>,
}
impl State {
    pub fn model(&self, vocab: usize) -> Result<Model> {
        Model::construct(
            vocab,
            self.rows.clone(),
            self.columns.clone(),
            self.decoder.clone(),
        )
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Seed,
    Decoder,
    Row,
    Column,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub kind: Kind,
    pub state: State,
    /// Actual minimal mismatch cost for each input category. The independent
    /// checker scans all q alternatives from the original observations.
    pub minimum_costs: Option<Vec<usize>>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cost {
    pub class_observations: usize,
    pub state_observations: usize,
    pub decoder_observations: usize,
    pub decoder_comparisons: usize,
    pub axis_steps: usize,
    pub axis_observation_comparisons: usize,
    pub axis_minimum_comparisons: usize,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub input: Input,
    pub steps: Vec<Step>,
    pub cost: Cost,
    pub best: Option<usize>,
    pub reason: String,
}
pub fn validate(input: &Input) -> Result<Vec<Sample>> {
    input.seed.validate()?;
    if input.seed.q > 257 || !(1..=128).contains(&input.rounds) || input.axis_budget > 256 {
        return Err("free refinement exceeds q257, rounds128 or axes256 bounds".into());
    }
    let samples: Vec<_> = input
        .observations
        .iter()
        .map(|&[cell, label]| Sample { cell, label })
        .collect();
    crate::validate_observations(input.seed.n, input.seed.vocab, &samples)?;
    Ok(samples)
}
