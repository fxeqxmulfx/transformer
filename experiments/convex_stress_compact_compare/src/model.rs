//! Model boundary ported from reduced_compare at f5015cec, using CM70 storage.
//! The model boundary contains only original training rows or a query pair.
//! Case identifiers, historical scores, generators and heldout labels stay in
//! the runner. Every consistent member of the fixed family remains present.
use crate::*;
use joint::{
    Parameters, Shape, Work,
    domain::Domain,
    head::{Audit, Head, Prediction},
    training::{Config, Mode, Step, Trained},
};
use serde::{Deserialize, Serialize};
mod predictor;
pub use predictor::Predictor;

pub const MODES: [Mode; 2] = crate::REQUIRED_MODES;
pub fn name(mode: Mode) -> &'static str {
    match mode {
        Mode::Forward => "forward",
        Mode::Full => "full",
        Mode::Mini => "mini",
    }
}
pub fn original_mode(mode: Mode) -> &'static str {
    match mode {
        Mode::Forward => "forward_only",
        Mode::Full => "full_batch",
        Mode::Mini => "minibatch",
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub shape: Shape,
    pub config: Config,
    pub tensor: disk::Artifact,
    pub tensor_body_sha256: [u8; 32],
    pub unknown_rows: usize,
    pub initial_ce: f64,
    pub final_ce: f64,
    pub audit: Audit,
    pub steps: Vec<Step>,
    pub training_operations: u64,
    pub total_operations: u64,
    pub feasibility_residual: f64,
    pub target_realizability_certified: bool,
    pub exact_real_projection_certified: bool,
}
impl State {
    pub fn from_trained(
        trained: Trained,
        domain: &Domain,
        tensor: tensor::Saved,
        total: u64,
        residual: f64,
    ) -> Self {
        Self {
            shape: trained.head.shape(),
            config: trained.config,
            tensor: tensor.artifact,
            tensor_body_sha256: tensor.body_sha256,
            unknown_rows: domain.unknown.len(),
            initial_ce: trained.initial_ce,
            final_ce: trained.final_ce,
            audit: trained.audit,
            steps: trained.steps,
            training_operations: trained.operations,
            total_operations: total,
            feasibility_residual: residual,
            target_realizability_certified: false,
            exact_real_projection_certified: false,
        }
    }
    pub fn check(
        &self,
        domain: &Domain,
        parameters: &Parameters,
        head: &Head,
        limit: u64,
        work: &mut Work,
    ) -> Result<()> {
        crate::memory::remaining(limit, 4096 + 8 * self.shape.classes as u128)?;
        if self.shape != domain.shape
            || self.shape != head.shape()
            || self.unknown_rows != domain.unknown.len()
            || self.config != pairs::policy(self.config.mode)
            || self.target_realizability_certified
            || self.exact_real_projection_certified
            || !self.initial_ce.is_finite()
            || !self.final_ce.is_finite()
            || self.final_ce < 0.
            || self.final_ce > self.initial_ce + 1e-8
            || self.total_operations > crate::work::ENDPOINT_LIMIT
            || self.training_operations > self.total_operations
        {
            return Err("invalid bounded model state or policy".into());
        }
        let violation = parameters.check(domain, 1e-8, work)?;
        let audit = head.audit(domain, parameters, work)?;
        if violation.to_bits() != self.feasibility_residual.to_bits()
            || audit != self.audit
            || audit
                .column_l1
                .iter()
                .any(|&v| v > 2. * domain.maximum_mass * self.shape.states as f64 + 1e-7)
        {
            return Err("saved complete model audit differs".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Answer {
    pub numerical: Prediction,
    /// This conditional anchor test is not a target-realizability certificate.
    pub all_latent_states_observed: bool,
    pub unanimous_observed_label: Option<usize>,
}
