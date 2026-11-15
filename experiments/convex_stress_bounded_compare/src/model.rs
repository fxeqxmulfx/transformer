//! The model boundary contains only original training rows or a query pair.
//! Case identifiers, historical scores, generators and heldout labels stay in
//! the runner. Every consistent member of the fixed family remains present.
use crate::*;
use joint::{
    Parameters, Record, Shape, Work,
    domain::Domain,
    head::{Head, HeadAudit, Prediction, PredictionBudget},
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
    pub records: Vec<Record>,
}
pub fn prepare(training: &Training, expected: &Selection, work: &mut Work) -> Result<Prepared> {
    let selected = pairs::selection::select(training, work)?;
    if selected != *expected {
        return Err("training selection differs from the frozen probe".into());
    }
    let (domain, records) = pairs::selection::encode(training, &selected, work)?;
    Ok(Prepared {
        selection: selected,
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
    pub initial_ce: f64,
    pub final_ce: f64,
    pub audit: HeadAudit,
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
        tensor: disk::Artifact,
        total: u64,
        residual: f64,
    ) -> Self {
        Self {
            shape: trained.head.shape,
            config: trained.config.clone(),
            tensor,
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
            || self.config != pairs::policy(self.config.mode)
            || self.target_realizability_certified
            || self.exact_real_projection_certified
            || !self.initial_ce.is_finite()
            || !self.final_ce.is_finite()
            || self.final_ce < 0.
            || self.final_ce > self.initial_ce + 1e-8
            || self.total_operations > joint::MAX_WORK
            || self.training_operations > self.total_operations
        {
            return Err("invalid bounded model state or policy".into());
        }
        let violation = domain.check(parameters, 1e-8)?;
        let audit = head.audit(parameters, work)?;
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
    ) -> Result<Self> {
        head.validate()?;
        domain.validate()?;
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
        work.charge(16384 * self.programs.len() as u128)?;
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
