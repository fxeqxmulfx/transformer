//! Complete indexed Mini updates, following CM82 (Lean ba4b4b93).
//!
//! Actual recovered complex forwards supply ordinary CE and full-class
//! residuals. Their binary64 probabilities are imported exactly as dyadics;
//! subsequent pullback/mean/proposal arithmetic is rational. Thus the numeric
//! residual approximates the Lean derivative, without an error enclosure.
//! Both shared passes and every record/coordinate remain present. All new
//! states pass the existing exact whole-model projection and replay checks.

use super::{ComplexModel, ProjectedModel, Renaming, offset};
use crate::{Rational, Refusal};
use num_bigint::BigInt;
use num_traits::{One, Zero};

/// Original immutable sequence and original next-object target. A borrowed
/// batch retains occurrence indices; equal records are never deduplicated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub tokens: Vec<usize>,
    pub target: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct MiniLimits {
    pub max_records: usize,
    pub max_pullback_terms: usize,
    pub max_projection_candidates: usize,
    pub max_working_bytes: usize,
}

/// Complete per-update admission, including retained gradient/proposal and
/// the existing model/projection reservation. This is conservative storage
/// accounting, not a measured allocator peak or a wall-clock guarantee.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MiniPlan {
    pub records: usize,
    pub passes: usize,
    pub pullback_terms: usize,
    pub projection_candidates: usize,
    pub reserved_bytes: usize,
}

impl MiniPlan {
    pub fn new(
        model: &ComplexModel,
        records: usize,
        paired: bool,
        limits: MiniLimits,
    ) -> Result<Self, Refusal> {
        if records == 0 {
            return Err(Refusal::EmptyBatch);
        }
        let family = model.domain.plan();
        let passes = if paired { 2 } else { 1 };
        let terms = records
            .checked_mul(passes)
            .and_then(|x| x.checked_mul(family.real_parameters))
            .ok_or(Refusal::Overflow)?;
        let candidates = (family.real_parameters / family.coordinates)
            .checked_mul(family.patterns)
            .ok_or(Refusal::Overflow)?;
        let entries = 2 * family.real_parameters as u128 + 8 * family.objects as u128;
        let per_entry =
            512 + 8 * u128::from(model.limits.projection.max_coefficient_bits).div_ceil(8);
        let record_bytes = (records as u128)
            .checked_mul(64 + 8 * family.context as u128)
            .ok_or(Refusal::Overflow)?;
        let bytes = entries
            .checked_mul(per_entry)
            .and_then(|x| x.checked_add(model.memory.reserved_bytes as u128))
            .and_then(|x| x.checked_add(record_bytes))
            .and_then(|x| x.checked_add(256 * family.objects as u128 + 16 * family.context as u128))
            .ok_or(Refusal::Overflow)?;
        let result = Self {
            records,
            passes,
            pullback_terms: terms,
            projection_candidates: candidates,
            reserved_bytes: usize::try_from(bytes).map_err(|_| Refusal::Overflow)?,
        };
        if records > limits.max_records
            || terms > limits.max_pullback_terms
            || candidates > limits.max_projection_candidates
            || result.reserved_bytes > limits.max_working_bytes
        {
            return Err(Refusal::Budget);
        }
        Ok(result)
    }
}

#[derive(Clone, Debug)]
pub struct BatchGradient {
    pub mean_loss: f64,
    pub coordinates: Vec<Rational>,
    pub plan: MiniPlan,
}

#[derive(Clone, Debug)]
pub struct MiniUpdate {
    pub projected: ProjectedModel,
    pub proposal: Vec<Rational>,
    pub gradient: BatchGradient,
}

impl ComplexModel {
    /// The exact CM82 coordinate formula with numeric actual-forward CE
    /// residuals. An empty batch is refused; short batches use their own size.
    pub fn mini_gradient(
        &self,
        records: &[Record],
        renaming: Option<&Renaming>,
        limits: MiniLimits,
    ) -> Result<BatchGradient, Refusal> {
        let plan = MiniPlan::new(self, records.len(), renaming.is_some(), limits)?;
        let family = self.domain.plan();
        if renaming.is_some_and(|r| r.permutation().len() != family.objects) {
            return Err(Refusal::InvalidRenaming);
        }
        // Validate every original record before computing any proposal.
        for record in records {
            self.domain.input_index(&record.tokens)?;
            if record.target >= family.objects {
                return Err(Refusal::InvalidObject);
            }
        }
        let arithmetic = self.limits.projection;
        let divisor = Rational::from_integer(BigInt::from(records.len()));
        arithmetic.check(&divisor)?;
        let pass_scale = if renaming.is_some() {
            Rational::new(BigInt::one(), BigInt::from(2))
        } else {
            Rational::one()
        };
        let mut coordinates = vec![Rational::zero(); family.real_parameters];
        let mut total_loss = 0.0;
        for record in records {
            let output = match renaming {
                Some(r) => self.paired_forward(&record.tokens, r)?,
                None => self.forward(&record.tokens)?,
            };
            total_loss += output.cross_entropy(record.target)?;
            if !total_loss.is_finite() {
                return Err(Refusal::NonFinite);
            }
            let mut residual = Vec::with_capacity(family.objects);
            for (class, &probability) in output.probabilities().iter().enumerate() {
                let mut value = arithmetic.from_f64(probability)?;
                if class == record.target {
                    value -= Rational::one();
                }
                value *= &pass_scale;
                arithmetic.check(&value)?;
                residual.push(value);
            }
            self.add_pullback(&record.tokens, &residual, None, &mut coordinates)?;
            if let Some(r) = renaming {
                let renamed: Vec<_> = record.tokens.iter().map(|&x| r.permutation()[x]).collect();
                // Renaming is checked involutive, so its inverse equals itself.
                // The residual still comes from the one averaged prediction.
                self.add_pullback(&renamed, &residual, Some(r), &mut coordinates)?;
            }
        }
        for value in &mut coordinates {
            *value /= &divisor;
            arithmetic.check(value)?;
        }
        Ok(BatchGradient {
            mean_loss: total_loss / records.len() as f64,
            coordinates,
            plan,
        })
    }

    fn add_pullback(
        &self,
        tokens: &[usize],
        residual: &[Rational],
        alignment: Option<&Renaming>,
        gradient: &mut [Rational],
    ) -> Result<(), Refusal> {
        let family = self.domain.plan();
        let input = self.domain.input_index(tokens)?;
        let row = &self.domain.rows()[input];
        for mask in 0..family.patterns {
            let active = mask & (1usize << input) != 0;
            for class in 0..family.objects {
                let source = alignment.map_or(class, |r| r.permutation()[class]);
                for sign in 0..2 {
                    for component in 0..2 {
                        let first = offset(family, mask, class, sign, component);
                        for (coordinate, value) in row.iter().enumerate() {
                            let mut term = if active {
                                &residual[source] * value
                            } else {
                                Rational::zero()
                            };
                            if sign != component {
                                term = -term;
                            }
                            self.limits.projection.check(&term)?;
                            gradient[first + coordinate] += term;
                            self.limits
                                .projection
                                .check(&gradient[first + coordinate])?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Full rational proposal followed by atomic whole-model projection.
    /// The caller can independently replay every returned certificate.
    pub fn mini_step(
        &self,
        records: &[Record],
        renaming: Option<&Renaming>,
        rate: &Rational,
        limits: MiniLimits,
    ) -> Result<MiniUpdate, Refusal> {
        if rate <= &Rational::zero() {
            return Err(Refusal::LearningRate);
        }
        self.limits.projection.check(rate)?;
        let gradient = self.mini_gradient(records, renaming, limits)?;
        let mut proposal = Vec::with_capacity(self.parameters.len());
        for (old, direction) in self.parameters.iter().zip(&gradient.coordinates) {
            let change = rate * direction;
            self.limits.projection.check(&change)?;
            let next = old - change;
            self.limits.projection.check(&next)?;
            proposal.push(next);
        }
        let projected = self.project_parameters(&proposal)?;
        Ok(MiniUpdate {
            projected,
            proposal,
            gradient,
        })
    }
}

#[cfg(test)]
mod tests;
