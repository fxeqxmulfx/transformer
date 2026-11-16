//! Actual complete two-sided loss queries and atomic forward-only update.

use super::{
    CellProbes, ComplexModel, DirectionProbe, ForwardConfig, ForwardEstimate, ForwardLimits,
    ForwardUpdate, Record, Renaming,
};
use crate::geometry::ConeGeometry;
use crate::model::offset;
use crate::{Rational, Refusal};
use num_traits::Zero;

impl ComplexModel {
    pub(super) fn forward_cell(
        &self,
        geometry: &ConeGeometry,
        first: usize,
        records: &[Record],
        renaming: Option<&Renaming>,
        cap: &Rational,
    ) -> Result<(Vec<Rational>, Vec<DirectionProbe>), Refusal> {
        let last = first + self.domain.plan().coordinates;
        let point = &self.parameters[first..last];
        let mut directions = Vec::with_capacity(geometry.basis().vectors.len());
        let mut pairings = Vec::with_capacity(geometry.basis().vectors.len());
        for basis in &geometry.basis().vectors {
            let query = geometry.symmetric_queries(point, basis, cap)?;
            let plus_loss = self.forward_query_loss(first, &query.plus, records, renaming)?;
            let minus_loss = self.forward_query_loss(first, &query.minus, records, renaming)?;
            let pairing = self.forward_pairing(&query.step, plus_loss, minus_loss)?;
            pairings.push(pairing.clone());
            directions.push(DirectionProbe {
                step: query.step,
                plus_loss,
                minus_loss,
                pairing,
            });
        }
        let recovered = geometry.recover_pairings(&pairings)?;
        geometry.verify_pairings(&recovered, &pairings)?;
        Ok((recovered, directions))
    }

    pub fn forward_mini_step(
        &self,
        records: &[Record],
        renaming: Option<&Renaming>,
        config: &ForwardConfig,
        limits: ForwardLimits,
    ) -> Result<ForwardUpdate, Refusal> {
        let plan = self.forward_admit(records, renaming, config, limits)?;
        let family = self.domain.plan();
        let old_loss = self.forward_mean_loss(records, renaming)?;
        let mut parameters = Vec::with_capacity(family.real_parameters);
        for mask in 0..family.patterns {
            let geometry = self.forward_geometry(mask, limits.geometry)?;
            for class in 0..family.objects {
                for sign in 0..2 {
                    for component in 0..2 {
                        let first = offset(family, mask, class, sign, component);
                        let point = &self.parameters[first..first + family.coordinates];
                        parameters.extend(geometry.mixture(point, &config.mixture)?);
                    }
                }
            }
        }
        let mixed = self.with_parameters(parameters)?;
        let mixed_loss = mixed.forward_mean_loss(records, renaming)?;
        let mut coordinates = vec![Rational::zero(); family.real_parameters];
        let mut cells = Vec::with_capacity(plan.cells);
        for mask in 0..family.patterns {
            let geometry = self.forward_geometry(mask, limits.geometry)?;
            for class in 0..family.objects {
                for sign in 0..2 {
                    for component in 0..2 {
                        let first = offset(family, mask, class, sign, component);
                        let last = first + family.coordinates;
                        let (recovered, directions) = mixed.forward_cell(
                            &geometry,
                            first,
                            records,
                            renaming,
                            &config.query_cap,
                        )?;
                        coordinates[first..last].clone_from_slice(&recovered);
                        cells.push(CellProbes {
                            mask,
                            class,
                            sign,
                            component,
                            directions,
                        });
                    }
                }
            }
        }
        let mut proposal = Vec::with_capacity(family.real_parameters);
        for (old, direction) in mixed.parameters.iter().zip(&coordinates) {
            let change = &config.rate * direction;
            self.limits.projection.check(&change)?;
            let next = old - change;
            self.limits.projection.check(&next)?;
            proposal.push(next);
        }
        let projected = mixed.project_parameters(&proposal)?;
        let final_loss = projected.model.forward_mean_loss(records, renaming)?;
        Ok(ForwardUpdate {
            config: config.clone(),
            estimate: ForwardEstimate {
                mixed,
                coordinates,
                cells,
                old_loss,
                mixed_loss,
                plan,
            },
            proposal,
            projected,
            final_loss,
        })
    }
}
