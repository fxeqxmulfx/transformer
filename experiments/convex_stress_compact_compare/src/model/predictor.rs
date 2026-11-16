//! Input-only prediction with the original independent interpreter and full
//! complex contractions. All selected programs, query states, exact-integer
//! scratch and both raw readback answers are reserved before allocation.
use super::*;
use crate::{memory, preparation::memory as preparation_memory};
use joint::head::PredictionBudget;
use pairs::{
    family::{Evaluator, Program},
    selection::Selection,
};

pub struct Predictor<'a> {
    head: &'a Head,
    domain: &'a Domain,
    programs: Vec<Program>,
    evaluator: Evaluator,
    budget: PredictionBudget,
    states: Vec<usize>,
}
impl<'a> Predictor<'a> {
    /// Limit excludes head, dictionary and caller-owned selection/records.
    /// The scratch reservation also covers one retained previous answer during
    /// replay, bounded JSON row strings and serializer/parser containers.
    pub fn new(
        head: &'a Head,
        domain: &'a Domain,
        selection: &Selection,
        queries: usize,
        limit: u64,
        work: &mut Work,
    ) -> Result<Self> {
        let shape = head.shape();
        shape.validate()?;
        let budget = PredictionBudget::new(queries)?;
        let retained_bytes = std::mem::size_of::<Self>() as u128
            + shape.programs as u128 * (std::mem::size_of::<Program>() + 8) as u128;
        // Integer products use at most8192 bits per class. 64KiB per class
        // covers both accumulators, moving limb growth, rational conversions,
        // intervals, logits and two raw answers; 8MiB covers fixed I/O scratch.
        let scratch = 8 * 1024 * 1024 + 65536 * shape.classes as u128;
        memory::remaining(
            limit,
            retained_bytes
                + scratch
                + preparation_memory::INTERPRETER_SCRATCH
                + preparation_memory::SELECTION_SCRATCH,
        )?;
        joint::domain::validate(domain, work)?;
        if shape != domain.shape
            || shape != selection.shape()
            || selection.retained.windows(2).any(|w| w[0] >= w[1])
        {
            return Err("query model shape or selected support order changed".into());
        }
        let bank = pairs::family::programs();
        if preparation_memory::vector(&bank) > preparation_memory::SELECTION_SCRATCH {
            return Err("query family exceeds admitted generator scratch".into());
        }
        let mut programs = memory::vector(shape.programs, limit)?;
        for &id in &selection.retained {
            programs.push(*bank.get(id).ok_or("program index outside fixed bank")?);
        }
        let states = memory::vector(shape.programs, limit)?;
        Ok(Self {
            head,
            domain,
            programs,
            states,
            evaluator: Evaluator::new(selection.vocabulary)?,
            budget,
        })
    }
    pub fn predict(&mut self, input: [u32; 2], work: &mut Work) -> Result<Answer> {
        let shape = self.head.shape();
        if input.iter().any(|&x| x as usize >= shape.classes) {
            return Err("query pair outside vocabulary".into());
        }
        work.charge(
            self.programs
                .iter()
                .map(|&p| 512 + self.evaluator.reference_cost(p))
                .sum(),
        )?;
        self.states.clear();
        self.states.extend(self.programs.iter().map(|&p| {
            self.evaluator
                .evaluate(p, input[0] as usize, input[1] as usize)
        }));
        for (&program, &state) in self.programs.iter().zip(&self.states) {
            if self
                .evaluator
                .reference(program, input[0] as usize, input[1] as usize)
                != state
            {
                return Err("independent query-state interpreter differs".into());
            }
        }
        let first = self.domain.labels[self.states[0]];
        let observed = self
            .states
            .iter()
            .enumerate()
            .all(|(p, &s)| self.domain.labels[p * shape.states + s].is_some());
        let unanimous = observed
            && self
                .states
                .iter()
                .enumerate()
                .all(|(p, &s)| self.domain.labels[p * shape.states + s] == first);
        let numerical = self.head.predict(&self.states, &mut self.budget, work)?;
        Ok(Answer {
            numerical,
            all_latent_states_observed: observed,
            unanimous_observed_label: if unanimous { first } else { None },
        })
    }
}
