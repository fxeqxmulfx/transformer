//! Full-family work/storage admission for one generation OR replay.
//! Geometry workspaces are reused, while all model coordinates and all probe
//! receipts remain present. Bounds include the complete final projection.

use super::ComplexModel;
use crate::Refusal;
use crate::geometry::{GeometryLimits, GeometryPlan};

#[derive(Clone, Copy, Debug)]
pub struct ForwardLimits {
    pub geometry: GeometryLimits,
    pub max_records: usize,
    pub max_geometry_builds: usize,
    pub max_active_candidates: usize,
    pub max_batch_evaluations: usize,
    pub max_forward_passes: usize,
    pub max_forward_terms: usize,
    pub max_constraint_terms: usize,
    pub max_working_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForwardPlan {
    pub records: usize,
    pub passes_per_record: usize,
    pub cells: usize,
    pub max_directions: usize,
    pub geometry_builds: usize,
    pub active_candidates: usize,
    pub batch_evaluations: usize,
    pub forward_passes: usize,
    pub forward_terms: usize,
    pub constraint_terms: usize,
    pub reserved_bytes: usize,
}

impl ForwardPlan {
    pub fn new(
        model: &ComplexModel,
        records: usize,
        paired: bool,
        limits: ForwardLimits,
    ) -> Result<Self, Refusal> {
        if records == 0 {
            return Err(Refusal::EmptyBatch);
        }
        let family = model.domain.plan();
        let passes = if paired { 2 } else { 1 };
        let cells = family.real_parameters / family.coordinates;
        let geometry_builds = family.patterns.checked_mul(2).ok_or(Refusal::Overflow)?;
        let batch_evaluations = family
            .real_parameters
            .checked_mul(2)
            .and_then(|x| x.checked_add(3))
            .ok_or(Refusal::Overflow)?;
        let forward_passes = batch_evaluations
            .checked_mul(records)
            .and_then(|x| x.checked_mul(passes))
            .ok_or(Refusal::Overflow)?;
        // A conservative coordinate-work bound for the complete physical
        // forward at every record/pass; it never substitutes sampled masks.
        let forward_terms = forward_passes
            .checked_mul(family.real_parameters)
            .ok_or(Refusal::Overflow)?;
        let constraint_terms = family
            .real_parameters
            .checked_mul(2)
            .and_then(|x| x.checked_add(2))
            .and_then(|x| x.checked_mul(family.real_parameters))
            .and_then(|x| x.checked_mul(family.prefixes))
            .ok_or(Refusal::Overflow)?;
        let geometry_candidates = geometry_builds
            .checked_mul(family.prefixes)
            .and_then(|x| x.checked_mul(family.patterns))
            .ok_or(Refusal::Overflow)?;
        let projection_candidates = cells
            .checked_mul(family.patterns)
            .ok_or(Refusal::Overflow)?;
        let active_candidates = geometry_candidates
            .checked_add(projection_candidates)
            .ok_or(Refusal::Overflow)?;
        if records > limits.max_records
            || geometry_builds > limits.max_geometry_builds
            || active_candidates > limits.max_active_candidates
            || batch_evaluations > limits.max_batch_evaluations
            || forward_passes > limits.max_forward_passes
            || forward_terms > limits.max_forward_terms
            || constraint_terms > limits.max_constraint_terms
        {
            return Err(Refusal::Budget);
        }
        // This temporary cone was already reserved by ModelLimits. No complete
        // geometry, query trace or extra model is allocated before admission.
        let cone = model.domain.cone(0, model.limits.projection)?;
        let geometry = GeometryPlan::new(&cone, limits.geometry)?;
        let p = family.real_parameters as u128;
        let entry_bytes =
            512 + 8 * u128::from(model.limits.projection.max_coefficient_bits).div_ceil(8);
        // In addition to the model/projection plan: mixed/query model arrays,
        // complete gradient/proposal, 2 rational values per probe, vectors and
        // replay scratch. Include extra recovered-factor caches and metadata.
        let entries = 12 * p + 16 * family.coordinates as u128;
        let extra = entries
            .checked_mul(entry_bytes)
            .and_then(|x| x.checked_add(256 * p + 256 * cells as u128))
            .and_then(|x| {
                x.checked_add((records as u128).checked_mul(64 + 8 * family.context as u128)?)
            })
            .ok_or(Refusal::Overflow)?;
        let bytes = extra
            .checked_add(model.memory.reserved_bytes as u128)
            .and_then(|x| x.checked_add(geometry.reserved_bytes as u128))
            .ok_or(Refusal::Overflow)?;
        let result = Self {
            records,
            passes_per_record: passes,
            cells,
            max_directions: family.real_parameters,
            geometry_builds,
            active_candidates,
            batch_evaluations,
            forward_passes,
            forward_terms,
            constraint_terms,
            reserved_bytes: usize::try_from(bytes).map_err(|_| Refusal::Overflow)?,
        };
        if result.reserved_bytes > limits.max_working_bytes {
            return Err(Refusal::Budget);
        }
        Ok(result)
    }
}
