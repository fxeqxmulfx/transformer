//! The model boundary contains only original training rows or a query pair.
//! Case identifiers, historical scores, generators and heldout labels stay in
//! the runner. Every consistent member of the fixed family remains present.
use crate::*;
use joint::{
    Parameters, Shape, Work,
    domain::Domain,
    head::{Audit, Head, Prediction, PredictionBudget},
    records::{Packed, Source},
    training::{Config, Mode, Step, Trained},
};
use pairs::{
    Training,
    family::{Evaluator, Program},
    selection::Selection,
};
use serde::{Deserialize, Serialize};

pub const MODES: [Mode; 3] = [Mode::Forward, Mode::Full, Mode::Mini];
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

pub struct Prepared {
    pub selection: Selection,
    pub domain: Domain,
    pub records: Packed,
}
impl Prepared {
    pub fn source(&self) -> Source<'_> {
        Source::Packed(&self.records)
    }
}
pub fn prepare(training: &Training, expected: &Selection, work: &mut Work) -> Result<Prepared> {
    // encode independently repeats the complete partition and streamed audit.
    let (domain, records) = pairs::encoding::encode_packed(training, expected, work)?;
    Ok(Prepared {
        selection: expected.clone(),
        domain,
        records,
    })
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
        trained: &Trained,
        domain: &Domain,
        tensor: tensor::Saved,
        total: u64,
        residual: f64,
    ) -> Self {
        Self {
            shape: trained.head.shape,
            config: trained.config.clone(),
            tensor: tensor.artifact,
            tensor_body_sha256: tensor.body_sha256,
            unknown_rows: domain.unknown.len(),
            initial_ce: trained.initial_ce,
            final_ce: trained.final_ce,
            audit: trained.audit.clone(),
            steps: trained.steps.clone(),
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
        work: &mut Work,
    ) -> Result<()> {
        if self.shape != domain.shape
            || self.shape != head.shape
            || self.unknown_rows != domain.unknown.len()
            || self.config != pairs::policy(self.config.mode)
            || self.target_realizability_certified
            || self.exact_real_projection_certified
            || !self.initial_ce.is_finite()
            || !self.final_ce.is_finite()
            || self.final_ce < 0.
            || self.final_ce > self.initial_ce + 1e-8
            || self.total_operations > crate::MAX_WORK
            || self.training_operations > self.total_operations
        {
            return Err("invalid bounded model state or policy".into());
        }
        let violation = domain.check(parameters, 1e-8, work)?;
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
pub struct Predictor<'a> {
    head: &'a Head,
    domain: &'a Domain,
    programs: Vec<Program>,
    evaluator: Evaluator,
    budget: PredictionBudget,
}
impl<'a> Predictor<'a> {
    pub fn new(
        head: &'a Head,
        domain: &'a Domain,
        selection: &Selection,
        queries: usize,
        work: &mut Work,
    ) -> Result<Self> {
        head.shape.validate()?;
        domain.validate(work)?;
        if head.shape != domain.shape || head.shape != selection.shape() {
            return Err("query model shape differs from selected support".into());
        }
        let bank = pairs::family::programs();
        let programs = selection
            .retained
            .iter()
            .map(|&i| {
                bank.get(i)
                    .copied()
                    .ok_or("program index outside fixed bank")
            })
            .collect::<std::result::Result<_, _>>()?;
        Ok(Self {
            head,
            domain,
            programs,
            evaluator: Evaluator::new(selection.vocabulary)?,
            budget: PredictionBudget::new(queries)?,
        })
    }
    pub fn predict(&mut self, input: [u32; 2], work: &mut Work) -> Result<Answer> {
        if input.iter().any(|&x| x as usize >= self.head.shape.classes) {
            return Err("query pair outside vocabulary".into());
        }
        work.charge(
            self.programs
                .iter()
                .map(|&p| 512 + self.evaluator.reference_cost(p))
                .sum(),
        )?;
        let states: Vec<_> = self
            .programs
            .iter()
            .map(|&p| {
                self.evaluator
                    .evaluate(p, input[0] as usize, input[1] as usize)
            })
            .collect();
        for (&program, &state) in self.programs.iter().zip(&states) {
            if self
                .evaluator
                .reference(program, input[0] as usize, input[1] as usize)
                != state
            {
                return Err("independent query-state interpreter differs".into());
            }
        }
        let first = self.domain.labels[states[0]];
        let observed = states
            .iter()
            .enumerate()
            .all(|(p, &s)| self.domain.labels[p * self.head.shape.states + s].is_some());
        let unanimous = observed
            && states
                .iter()
                .enumerate()
                .all(|(p, &s)| self.domain.labels[p * self.head.shape.states + s] == first);
        let numerical = self.head.predict(&states, &mut self.budget, work)?;
        Ok(Answer {
            numerical,
            all_latent_states_observed: observed,
            unanimous_observed_label: if unanimous { first } else { None },
        })
    }
}
