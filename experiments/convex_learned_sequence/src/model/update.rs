//! Atomic whole-model CM79 projection, retaining every component certificate.
//! A refusal leaves the old model untouched. The independent replay checks
//! both the component certificates and their correspondence to the new array.

use super::{ComplexModel, offset};
use crate::{Rational, Refusal, projection::Projection};

#[derive(Clone, Debug)]
pub struct ProjectedModel {
    pub model: ComplexModel,
    pub certificates: Vec<Projection>,
}

impl ComplexModel {
    pub fn project_parameters(&self, proposal: &[Rational]) -> Result<ProjectedModel, Refusal> {
        let plan = self.domain.plan();
        if proposal.len() != plan.real_parameters {
            return Err(Refusal::Shape);
        }
        let mut parameters = Vec::with_capacity(plan.real_parameters);
        let mut certificates = Vec::with_capacity(plan.real_parameters / plan.coordinates);
        for mask in 0..plan.patterns {
            // Only one pattern's Gram/workspace is live at a time. The complete
            // parameter and certificate arrays retain every pattern and class.
            let cone = self.domain.cone(mask, self.limits.projection)?;
            for class in 0..plan.objects {
                for sign in 0..2 {
                    for component in 0..2 {
                        let first = offset(plan, mask, class, sign, component);
                        let certificate =
                            cone.project(&proposal[first..first + plan.coordinates])?;
                        cone.verify(&proposal[first..first + plan.coordinates], &certificate)?;
                        parameters.extend_from_slice(&certificate.point);
                        certificates.push(certificate);
                    }
                }
            }
        }
        let model = self.with_parameters(parameters)?;
        model.verify_projection(proposal, &certificates)?;
        Ok(ProjectedModel {
            model,
            certificates,
        })
    }

    pub fn verify_projection(
        &self,
        proposal: &[Rational],
        certificates: &[Projection],
    ) -> Result<(), Refusal> {
        let plan = self.domain.plan();
        if proposal.len() != plan.real_parameters
            || certificates.len() != plan.real_parameters / plan.coordinates
        {
            return Err(Refusal::Shape);
        }
        for mask in 0..plan.patterns {
            let cone = self.domain.cone(mask, self.limits.projection)?;
            for class in 0..plan.objects {
                for sign in 0..2 {
                    for component in 0..2 {
                        let first = offset(plan, mask, class, sign, component);
                        let last = first + plan.coordinates;
                        let certificate = &certificates[first / plan.coordinates];
                        cone.verify(&proposal[first..last], certificate)?;
                        if self.parameters[first..last] != certificate.point {
                            return Err(Refusal::Stationarity);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
