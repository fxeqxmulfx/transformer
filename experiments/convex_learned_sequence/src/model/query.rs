//! CM83 local replacement, Lean commit ce6970ab: `Replacement.replace_mem`
//! and `Replacement.replace_forward`. The old model remains immutable and
//! feasible. Check every original row of the changed component, recover the
//! entire changed complex neuron, and retain all other learned factors.
//! Native shape/arithmetic checks and binary64 comparison remain additional
//! obligations; the Lean equality is over exact real and complex numbers.

use super::{ComplexModel, RecoveredNeuron};
use crate::{Rational, Refusal, checked_dot};
use num_traits::Zero;

/// A checked replacement tied to its original immutable model. Private fields
/// prevent constructing an unchecked patch or applying it to another model.
pub(super) struct ComponentQuery<'a> {
    model: &'a ComplexModel,
    index: usize,
    recovered: RecoveredNeuron,
}

impl ComplexModel {
    pub(super) fn component_query(
        &self,
        first: usize,
        vector: &[Rational],
    ) -> Result<ComponentQuery<'_>, Refusal> {
        let plan = self.domain.plan();
        let d = plan.coordinates;
        if vector.len() != d || !first.is_multiple_of(d) || first >= plan.real_parameters {
            return Err(Refusal::Shape);
        }
        for value in vector {
            self.limits.projection.check(value)?;
        }
        let cell = first / d;
        let mask = cell / (plan.objects * 4);
        for (row_index, row) in self.domain.rows().iter().enumerate() {
            let response = checked_dot(row, vector, self.limits.projection)?;
            let oriented = if mask & (1usize << row_index) != 0 {
                response
            } else {
                -response
            };
            if oriented < Rational::zero() {
                return Err(Refusal::Infeasible);
            }
        }
        let index = cell / 2;
        let real_start = index * 2 * d;
        let imag_start = real_start + d;
        let (real, imag) = if cell.is_multiple_of(2) {
            (vector, &self.parameters[imag_start..imag_start + d])
        } else {
            (&self.parameters[real_start..real_start + d], vector)
        };
        // Both factors depend on the norm of the whole complex vector. Never
        // preserve an old opposite-component factor inside the changed neuron.
        let recovered = RecoveredNeuron::new(real, imag)?;
        Ok(ComponentQuery {
            model: self,
            index,
            recovered,
        })
    }
}

impl ComponentQuery<'_> {
    pub(super) fn model(&self) -> &ComplexModel {
        self.model
    }

    pub(super) fn neuron(&self, index: usize) -> &RecoveredNeuron {
        if index == self.index {
            &self.recovered
        } else {
            &self.model.recovered[index]
        }
    }
}

#[cfg(test)]
mod tests;
