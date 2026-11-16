//! Replay every receipt against actual queried models, without running the
//! update generator or an analytic gradient. Kernel/pairing and projection
//! checks are independent certificate acceptance paths.

use super::{ComplexModel, ForwardLimits, ForwardUpdate, Record, Renaming};
use crate::Refusal;
use crate::model::offset;

fn same_loss(actual: f64, recorded: f64) -> Result<(), Refusal> {
    if !recorded.is_finite() || recorded < 0.0 || actual.to_bits() != recorded.to_bits() {
        return Err(Refusal::ForwardTrace);
    }
    Ok(())
}

impl ComplexModel {
    pub fn verify_forward_update(
        &self,
        records: &[Record],
        renaming: Option<&Renaming>,
        update: &ForwardUpdate,
        limits: ForwardLimits,
    ) -> Result<(), Refusal> {
        let plan = self.forward_admit(records, renaming, &update.config, limits)?;
        let family = self.domain.plan();
        let estimate = &update.estimate;
        if estimate.plan != plan
            || estimate.mixed.domain.plan() != family
            || update.projected.model.domain.plan() != family
            || estimate.coordinates.len() != family.real_parameters
            || update.proposal.len() != family.real_parameters
            || estimate.cells.len() != plan.cells
        {
            return Err(Refusal::ForwardTrace);
        }
        same_loss(
            self.forward_mean_loss(records, renaming)?,
            estimate.old_loss,
        )?;
        for mask in 0..family.patterns {
            let geometry = self.forward_geometry(mask, limits.geometry)?;
            for class in 0..family.objects {
                for sign in 0..2 {
                    for component in 0..2 {
                        let first = offset(family, mask, class, sign, component);
                        let last = first + family.coordinates;
                        let expected = geometry
                            .mixture(&self.parameters[first..last], &update.config.mixture)?;
                        if estimate.mixed.parameters[first..last] != expected {
                            return Err(Refusal::ForwardTrace);
                        }
                    }
                }
            }
        }
        same_loss(
            estimate.mixed.forward_mean_loss(records, renaming)?,
            estimate.mixed_loss,
        )?;
        for mask in 0..family.patterns {
            let geometry = self.forward_geometry(mask, limits.geometry)?;
            for class in 0..family.objects {
                for sign in 0..2 {
                    for component in 0..2 {
                        let first = offset(family, mask, class, sign, component);
                        let last = first + family.coordinates;
                        let cell = &estimate.cells[first / family.coordinates];
                        if (cell.mask, cell.class, cell.sign, cell.component)
                            != (mask, class, sign, component)
                            || cell.directions.len() != geometry.basis().vectors.len()
                        {
                            return Err(Refusal::ForwardTrace);
                        }
                        let mut pairings = Vec::with_capacity(cell.directions.len());
                        for (basis, probe) in geometry.basis().vectors.iter().zip(&cell.directions)
                        {
                            let query = geometry.symmetric_queries(
                                &estimate.mixed.parameters[first..last],
                                basis,
                                &update.config.query_cap,
                            )?;
                            if query.step != probe.step {
                                return Err(Refusal::ForwardTrace);
                            }
                            let plus = estimate.mixed.forward_query_loss(
                                first,
                                &query.plus,
                                records,
                                renaming,
                            )?;
                            let minus = estimate.mixed.forward_query_loss(
                                first,
                                &query.minus,
                                records,
                                renaming,
                            )?;
                            same_loss(plus, probe.plus_loss)?;
                            same_loss(minus, probe.minus_loss)?;
                            let pairing = self.forward_pairing(&query.step, plus, minus)?;
                            if pairing != probe.pairing {
                                return Err(Refusal::ForwardTrace);
                            }
                            pairings.push(pairing);
                        }
                        geometry.verify_pairings(&estimate.coordinates[first..last], &pairings)?;
                    }
                }
            }
        }
        for ((point, direction), proposed) in estimate
            .mixed
            .parameters
            .iter()
            .zip(&estimate.coordinates)
            .zip(&update.proposal)
        {
            let change = &update.config.rate * direction;
            self.limits.projection.check(&change)?;
            let expected = point - change;
            self.limits.projection.check(&expected)?;
            if &expected != proposed {
                return Err(Refusal::ForwardTrace);
            }
        }
        // Reconstruct under the original model's arithmetic limits as well;
        // an externally supplied model with looser limits cannot bypass them.
        let checked = self.with_parameters(update.projected.model.parameters.clone())?;
        checked.verify_projection(&update.proposal, &update.projected.certificates)?;
        same_loss(
            checked.forward_mean_loss(records, renaming)?,
            update.final_loss,
        )
    }
}
