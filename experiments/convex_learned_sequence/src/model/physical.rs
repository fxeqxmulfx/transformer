//! CM76 physical-to-reduced coverage map, using every original raw prefix.
//! Real and imaginary component patterns are aggregated separately. The full
//! signed class array remains present, including every zero coordinate.

use super::{ComplexModel, offset};
use crate::{Rational, Refusal, checked_dot};
use num_traits::{Signed, Zero};

#[derive(Clone, Debug)]
pub struct PhysicalNeuron {
    pub real: Vec<Rational>,
    pub imaginary: Vec<Rational>,
    pub class_readouts: Vec<Rational>,
}

impl ComplexModel {
    /// Construct the exact CM76 joint point of a supplied physical network.
    /// This is a representation map or initialization, not a training result.
    pub fn with_physical(&self, neurons: &[PhysicalNeuron]) -> Result<Self, Refusal> {
        let plan = self.domain.plan();
        let input_entries = (2u128 * plan.coordinates as u128 + plan.objects as u128)
            .checked_mul(neurons.len() as u128)
            .ok_or(Refusal::Overflow)?;
        let bytes = input_entries
            .checked_mul(
                512 + 8 * u128::from(self.limits.projection.max_coefficient_bits).div_ceil(8),
            )
            .and_then(|x| x.checked_add(self.memory.reserved_bytes as u128))
            .ok_or(Refusal::Overflow)?;
        if bytes > self.limits.max_working_bytes as u128 {
            return Err(Refusal::Budget);
        }
        let mut parameters = vec![Rational::zero(); plan.real_parameters];
        for neuron in neurons {
            if neuron.real.len() != plan.coordinates
                || neuron.imaginary.len() != plan.coordinates
                || neuron.class_readouts.len() != plan.objects
            {
                return Err(Refusal::Shape);
            }
            for value in neuron
                .real
                .iter()
                .chain(&neuron.imaginary)
                .chain(&neuron.class_readouts)
            {
                self.limits.projection.check(value)?;
            }
            for (component, weights) in [&neuron.real, &neuron.imaginary].into_iter().enumerate() {
                let mut mask = 0usize;
                for (index, row) in self.domain.rows().iter().enumerate() {
                    if checked_dot(row, weights, self.limits.projection)? >= Rational::zero() {
                        mask |= 1usize << index;
                    }
                }
                for (class, readout) in neuron.class_readouts.iter().enumerate() {
                    let sign = usize::from(readout.is_negative());
                    let magnitude = readout.abs();
                    let first = offset(plan, mask, class, sign, component);
                    for (coordinate, value) in weights.iter().enumerate() {
                        let contribution = &magnitude * value;
                        self.limits.projection.check(&contribution)?;
                        parameters[first + coordinate] += contribution;
                        self.limits
                            .projection
                            .check(&parameters[first + coordinate])?;
                    }
                }
            }
        }
        self.with_parameters(parameters)
    }
}
