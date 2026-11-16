//! Actual CM75--CM78 learned complex inference and CM79 certified updates.
//!
//! The full rational joint array is authoritative. Derived floating factors
//! execute the original physical computation; they are not a fixed-feature
//! classifier. Exact cone feasibility does not make floating inference exact.
//! All constructors check the complete domain and refuse numerical failures.

mod forward;
pub mod learning;
mod physical;
mod update;

use crate::domain::{DomainLimits, DomainPlan, RawDomain};
use crate::{ProjectionLimits, Rational, Refusal, checked_dot};
use num_traits::Zero;
use std::sync::Arc;

pub use forward::{Output, RecoveredNeuron, Renaming};
pub use num_complex::Complex64;
pub use physical::PhysicalNeuron;
pub use update::ProjectedModel;

#[derive(Clone, Copy, Debug)]
pub struct ModelLimits {
    pub domain: DomainLimits,
    pub projection: ProjectionLimits,
    pub max_working_bytes: usize,
}

/// Conservative reservation for old/proposed/new state, every certificate,
/// rational work arrays and two floating factor caches. This is not a claim
/// about an allocator's measured peak; external aggregate RSS guards remain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryPlan {
    pub rational_entries: usize,
    pub floating_bytes: usize,
    pub reserved_bytes: usize,
}

impl MemoryPlan {
    pub fn new(plan: &DomainPlan, limits: ModelLimits) -> Result<Self, Refusal> {
        let p = plan.real_parameters as u128;
        let d = plan.coordinates as u128;
        let k = plan.prefixes as u128;
        let cells = p / d;
        let entries = 4 * p + cells * k + 3 * k * d + 3 * k * k + 8 * (k + d);
        let bits = u128::from(limits.projection.max_coefficient_bits);
        let rational_bytes = 512 + 8 * bits.div_ceil(8);
        let floating =
            16 * p + 32 * cells + 192 * plan.objects as u128 + 8 * k * (plan.context as u128 + 3);
        let bytes = entries
            .checked_mul(rational_bytes)
            .and_then(|x| x.checked_add(floating))
            .ok_or(Refusal::Overflow)?;
        let result = Self {
            rational_entries: usize::try_from(entries).map_err(|_| Refusal::Overflow)?,
            floating_bytes: usize::try_from(floating).map_err(|_| Refusal::Overflow)?,
            reserved_bytes: usize::try_from(bytes).map_err(|_| Refusal::Overflow)?,
        };
        if result.reserved_bytes > limits.max_working_bytes {
            return Err(Refusal::Budget);
        }
        Ok(result)
    }
}

#[derive(Clone, Debug)]
pub struct ComplexModel {
    domain: Arc<RawDomain>,
    limits: ModelLimits,
    memory: MemoryPlan,
    parameters: Vec<Rational>,
    recovered: Vec<RecoveredNeuron>,
}

impl ComplexModel {
    pub fn zero(objects: usize, context: usize, limits: ModelLimits) -> Result<Self, Refusal> {
        let plan = DomainPlan::new(objects, context, limits.domain)?;
        let memory = MemoryPlan::new(&plan, limits)?;
        // Admit each complete component search before allocating model state.
        let domain = Arc::new(RawDomain::new(objects, context, limits.domain)?);
        domain.cone(0, limits.projection)?;
        Self::from_array(
            domain,
            limits,
            memory,
            vec![Rational::zero(); plan.real_parameters],
        )
    }

    /// Same domain, all new learned coordinates. Feasibility is checked before
    /// any derived factor cache is accepted; the old model is not modified.
    pub fn with_parameters(&self, parameters: Vec<Rational>) -> Result<Self, Refusal> {
        Self::from_array(
            Arc::clone(&self.domain),
            self.limits,
            self.memory.clone(),
            parameters,
        )
    }

    fn from_array(
        domain: Arc<RawDomain>,
        limits: ModelLimits,
        memory: MemoryPlan,
        parameters: Vec<Rational>,
    ) -> Result<Self, Refusal> {
        let plan = domain.plan();
        if parameters.len() != plan.real_parameters {
            return Err(Refusal::Shape);
        }
        for value in &parameters {
            limits.projection.check(value)?;
        }
        for mask in 0..plan.patterns {
            for class in 0..plan.objects {
                for sign in 0..2 {
                    for component in 0..2 {
                        let offset = offset(plan, mask, class, sign, component);
                        let vector = &parameters[offset..offset + plan.coordinates];
                        for (row_index, row) in domain.rows().iter().enumerate() {
                            let response = checked_dot(row, vector, limits.projection)?;
                            let oriented = if mask & (1usize << row_index) != 0 {
                                response
                            } else {
                                -response
                            };
                            if oriented < Rational::zero() {
                                return Err(Refusal::Infeasible);
                            }
                        }
                    }
                }
            }
        }
        let mut recovered = Vec::with_capacity(plan.real_parameters / (2 * plan.coordinates));
        for mask in 0..plan.patterns {
            for class in 0..plan.objects {
                for sign in 0..2 {
                    let real = offset(plan, mask, class, sign, 0);
                    let imag = offset(plan, mask, class, sign, 1);
                    recovered.push(RecoveredNeuron::new(
                        &parameters[real..real + plan.coordinates],
                        &parameters[imag..imag + plan.coordinates],
                    )?);
                }
            }
        }
        Ok(Self {
            domain,
            limits,
            memory,
            parameters,
            recovered,
        })
    }

    pub fn domain(&self) -> &RawDomain {
        &self.domain
    }
    pub fn parameters(&self) -> &[Rational] {
        &self.parameters
    }
    pub fn recovered(&self) -> &[RecoveredNeuron] {
        &self.recovered
    }
    pub fn memory_plan(&self) -> &MemoryPlan {
        &self.memory
    }

    /// Native order: sign 0 is Lean true (positive); component 0 is Lean true
    /// (real). Every mask/class/sign/component has all original coordinates.
    pub fn parameter_index(
        &self,
        mask: usize,
        class: usize,
        sign: usize,
        component: usize,
        coordinate: usize,
    ) -> Result<usize, Refusal> {
        let plan = self.domain.plan();
        if mask >= plan.patterns
            || class >= plan.objects
            || sign >= 2
            || component >= 2
            || coordinate >= plan.coordinates
        {
            return Err(Refusal::Shape);
        }
        Ok(offset(plan, mask, class, sign, component) + coordinate)
    }
}

fn offset(plan: &DomainPlan, mask: usize, class: usize, sign: usize, component: usize) -> usize {
    (((mask * plan.objects + class) * 2 + sign) * 2 + component) * plan.coordinates
}

#[cfg(test)]
mod tests;
