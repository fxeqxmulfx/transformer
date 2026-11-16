//! Complete forward-only Mini, from CM79--CM82 (Lean ba4b4b93).
//!
//! Every certified basis direction is queried through actual recovered complex
//! forwards and ordinary indexed mean CE. There is no analytic pullback in this
//! implementation. Numeric central differences estimate the proved derivative;
//! no finite-step error or convergence bound is asserted. Exact rational
//! geometry, full Gram recovery and whole-model projection retain the complete
//! original domain. Receipts support independent full-query/update replay.

mod plan;
mod replay;
mod run;

pub use plan::{ForwardLimits, ForwardPlan};

use super::learning::Record;
use super::{ComplexModel, ProjectedModel, Renaming};
use crate::geometry::{ConeGeometry, GeometryLimits};
use crate::{Rational, Refusal};
use num_bigint::BigInt;
use num_traits::{One, Zero};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForwardConfig {
    pub rate: Rational,
    pub mixture: Rational,
    pub query_cap: Rational,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectionProbe {
    pub step: Rational,
    pub plus_loss: f64,
    pub minus_loss: f64,
    pub pairing: Rational,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CellProbes {
    pub mask: usize,
    pub class: usize,
    pub sign: usize,
    pub component: usize,
    /// One receipt per complete certified basis vector, in certificate order.
    /// An empty list is valid only for a certified zero-dimensional kernel.
    pub directions: Vec<DirectionProbe>,
}

#[derive(Clone, Debug)]
pub struct ForwardEstimate {
    pub mixed: ComplexModel,
    pub coordinates: Vec<Rational>,
    pub cells: Vec<CellProbes>,
    pub old_loss: f64,
    pub mixed_loss: f64,
    pub plan: ForwardPlan,
}

#[derive(Clone, Debug)]
pub struct ForwardUpdate {
    pub config: ForwardConfig,
    pub estimate: ForwardEstimate,
    pub proposal: Vec<Rational>,
    pub projected: ProjectedModel,
    pub final_loss: f64,
}

impl ComplexModel {
    fn forward_admit(
        &self,
        records: &[Record],
        renaming: Option<&Renaming>,
        config: &ForwardConfig,
        limits: ForwardLimits,
    ) -> Result<ForwardPlan, Refusal> {
        let arithmetic = self.limits.projection;
        for value in [&config.rate, &config.mixture, &config.query_cap] {
            arithmetic.check(value)?;
        }
        if config.rate <= Rational::zero() {
            return Err(Refusal::LearningRate);
        }
        if config.mixture <= Rational::zero()
            || config.mixture > Rational::one()
            || config.query_cap <= Rational::zero()
        {
            return Err(Refusal::QueryStep);
        }
        let plan = ForwardPlan::new(self, records.len(), renaming.is_some(), limits)?;
        if renaming.is_some_and(|r| r.permutation().len() != self.domain.plan().objects) {
            return Err(Refusal::InvalidRenaming);
        }
        for record in records {
            self.domain.input_index(&record.tokens)?;
            if record.target >= self.domain.plan().objects {
                return Err(Refusal::InvalidObject);
            }
        }
        Ok(plan)
    }

    fn forward_geometry(
        &self,
        mask: usize,
        limits: GeometryLimits,
    ) -> Result<ConeGeometry, Refusal> {
        ConeGeometry::new(self.domain.cone(mask, self.limits.projection)?, limits)
    }

    /// Loss-only evaluation. It never obtains or constructs a CE gradient.
    /// The calling update/replay has already admitted every record and pass.
    fn forward_mean_loss(
        &self,
        records: &[Record],
        renaming: Option<&Renaming>,
    ) -> Result<f64, Refusal> {
        if records.is_empty() {
            return Err(Refusal::EmptyBatch);
        }
        let mut sum = 0.0;
        for record in records {
            let output = match renaming {
                Some(r) => self.paired_forward(&record.tokens, r)?,
                None => self.forward(&record.tokens)?,
            };
            sum += output.cross_entropy(record.target)?;
            if !sum.is_finite() {
                return Err(Refusal::NonFinite);
            }
        }
        Ok(sum / records.len() as f64)
    }

    fn forward_query_loss(
        &self,
        first: usize,
        vector: &[Rational],
        records: &[Record],
        renaming: Option<&Renaming>,
    ) -> Result<f64, Refusal> {
        let mut parameters = self.parameters.clone();
        parameters[first..first + vector.len()].clone_from_slice(vector);
        // This checks every original inequality, recovers both complex factors,
        // executes actual inference, then drops the query model on return.
        self.with_parameters(parameters)?
            .forward_mean_loss(records, renaming)
    }

    fn forward_pairing(&self, step: &Rational, plus: f64, minus: f64) -> Result<Rational, Refusal> {
        let arithmetic = self.limits.projection;
        if step <= &Rational::zero() {
            return Err(Refusal::QueryStep);
        }
        let difference = arithmetic.from_f64(plus)? - arithmetic.from_f64(minus)?;
        arithmetic.check(&difference)?;
        let denominator = Rational::from_integer(BigInt::from(2)) * step;
        arithmetic.check(&denominator)?;
        let value = difference / denominator;
        arithmetic.check(&value)?;
        Ok(value)
    }
}

#[cfg(test)]
mod tests;
