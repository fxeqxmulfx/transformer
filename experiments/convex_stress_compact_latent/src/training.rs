//! Actual CM65/66 projected CE with CM70 storage. Ported from reduced_latent
//! at b3aa191c: original records, tails, floating operations and all three
//! policies remain. Local admission includes every live training output,
//! dictionary-rebuild scratch and nested projection/head buffers. Original
//! input dictionaries/records and process/runtime reserves are caller-owned.
use crate::{
    head::{Audit, Head},
    memory,
    projection::{self, Target},
    *,
};
pub use legacy::training::{Config, Mode, Step};
use legacy::{domain::Domain, records::Source};
use std::mem::size_of;
#[cfg(test)]
mod tests;

#[derive(Debug)]
pub struct Trained {
    pub config: Config,
    pub parameters: Parameters,
    pub head: Head,
    pub initial_ce: f64,
    pub final_ce: f64,
    pub audit: Audit,
    pub steps: Vec<Step>,
    pub operations: u64,
    pub target_realizability_certified: bool,
    pub exact_real_projection_certified: bool,
}
pub fn loss(head: &Head, records: &[Record], work: &mut Work) -> Result<f64> {
    loss_source(head, Source::Expanded(records), 0..records.len(), work)
}
pub fn loss_source(
    head: &Head,
    records: Source<'_>,
    range: std::ops::Range<usize>,
    work: &mut Work,
) -> Result<f64> {
    if range.is_empty() {
        return Err("empty ordinary CE batch".into());
    }
    records.check_shape(head.shape())?;
    let count = range.len();
    let mut total = 0.;
    records.visit(range, work, |row, work| {
        Domain::record(head.shape(), row)?;
        total += ce(&head.logits(&row.states, work)?, row.label);
        Ok(())
    })?;
    Ok(total / count as f64)
}
pub fn gradient(
    domain: &Domain,
    head: &Head,
    records: &[Record],
    limit: u64,
    work: &mut Work,
) -> Result<(f64, Parameters)> {
    gradient_source(
        domain,
        head,
        Source::Expanded(records),
        0..records.len(),
        limit,
        work,
    )
}
pub fn gradient_source(
    domain: &Domain,
    head: &Head,
    records: Source<'_>,
    range: std::ops::Range<usize>,
    limit: u64,
    work: &mut Work,
) -> Result<(f64, Parameters)> {
    if range.is_empty() || head.shape() != domain.shape {
        return Err("invalid observed CE gradient batch".into());
    }
    records.admit(
        domain.shape,
        domain.unknown.len(),
        domain.head_columns_bound,
    )?;
    let output_room = memory::remaining(
        limit,
        4096 + 2 * memory::vector_bytes::<f64>(domain.shape.classes)
            + memory::vector_bytes::<usize>(domain.shape.programs),
    )?;
    work.charge(
        8 * domain.parameters() as u128 + 16 * range.len() as u128 * domain.shape.programs as u128,
    )?;
    let (mut total, mut selected) = (0., 0.);
    let scale = 1. / range.len() as f64;
    records.visit(range, work, |row, work| {
        Domain::record(domain.shape, row)?;
        for (p, &s) in row.states.iter().enumerate() {
            if domain.labels[p * domain.shape.states + s] != Some(row.label) {
                return Err("gradient record is not an original observed anchor".into());
            }
        }
        let logits = head.logits(&row.states, work)?;
        total += ce(&logits, row.label);
        let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = logits.iter().map(|x| (x - top).exp()).sum();
        selected += ((logits[row.label] - top).exp() / sum - 1.) * scale;
        Ok(())
    })?;
    let (mut masses, room) =
        crate::parameters::allocate_masses(domain.shape.programs, output_room)?;
    masses.resize(domain.shape.programs, selected);
    let mut free = rows::Builder::new(domain.shape.classes, domain.unknown.len(), room)?;
    for _ in &domain.unknown {
        free.push_repeated(0.)?;
    }
    let derivative = Parameters::from_parts(domain, masses, free.finish()?, output_room)?;
    Ok((total * scale, derivative))
}
pub fn train(
    domain: &Domain,
    records: &[Record],
    config: Config,
    limit: u64,
    work: &mut Work,
) -> Result<Trained> {
    train_source(domain, Source::Expanded(records), config, limit, work)
}
pub fn train_source(
    domain: &Domain,
    records: Source<'_>,
    config: Config,
    limit: u64,
    work: &mut Work,
) -> Result<Trained> {
    config.validate()?;
    let (_, _, fixed) = workspace(domain, records, &config)?;
    let initial = Parameters::initial(
        domain,
        config.initial_mass,
        memory::remaining(limit, fixed)?,
        work,
    )?;
    train_from_source(domain, records, config, initial, limit, work)
}
pub fn train_from(
    domain: &Domain,
    records: &[Record],
    config: Config,
    parameters: Parameters,
    limit: u64,
    work: &mut Work,
) -> Result<Trained> {
    train_from_source(
        domain,
        Source::Expanded(records),
        config,
        parameters,
        limit,
        work,
    )
}
pub fn train_from_source(
    domain: &Domain,
    records: Source<'_>,
    config: Config,
    mut parameters: Parameters,
    limit: u64,
    work: &mut Work,
) -> Result<Trained> {
    config.validate()?;
    let (batch, step_count, fixed) = workspace(domain, records, &config)?;
    let room = |owned: u128| memory::remaining(limit, fixed + owned);
    room(parameters.allocation_bytes() as u128)?;
    let start = work.used;
    if *domain
        != Domain::from_source(
            domain.shape,
            records,
            domain.minimum_mass,
            domain.maximum_mass,
            work,
        )?
    {
        return Err("reduced domain has constraints absent from original training records".into());
    }
    parameters.check(domain, 1e-10, work)?;
    let mut head = Head::build(
        domain,
        &parameters,
        room(parameters.allocation_bytes() as u128)?,
        work,
    )?;
    let initial_ce = loss_source(&head, records, 0..records.len(), work)?;
    let mut steps = memory::vector(step_count, limit)?;
    for epoch in 0..config.epochs {
        for first in (0..records.len()).step_by(batch) {
            let end = (first + batch).min(records.len());
            let current = parameters.allocation_bytes() as u128 + head.allocation_bytes() as u128;
            let (before_ce, derivative) =
                gradient_source(domain, &head, records, first..end, room(current)?, work)?;
            let with_derivative = current + derivative.allocation_bytes() as u128;
            let proposal = Target::step(
                domain,
                &parameters,
                &derivative,
                config.learning_rate,
                room(with_derivative)?,
                work,
            )?;
            let with_proposal = with_derivative + proposal.allocation_bytes() as u128;
            let result = projection::project(domain, &proposal, room(with_proposal)?, work)?;
            let next = Head::build(
                domain,
                &result.parameters,
                room(
                    with_proposal
                        + result.parameters.allocation_bytes() as u128
                        + size_of::<projection::Projection>() as u128,
                )?,
                work,
            )?;
            let after_ce = loss_source(&next, records, first..end, work)?;
            if after_ce > before_ce + 1e-8 {
                return Err("observed CE increased after weighted projection".into());
            }
            let change = parameters
                .distance(&result.parameters, domain, work)?
                .sqrt();
            steps.push(Step {
                epoch,
                first_record: first,
                end_record_exclusive: end,
                before_ce,
                after_ce,
                before_mass: parameters.masses().iter().sum(),
                after_mass: result.parameters.masses().iter().sum(),
                parameter_change_l2: change,
                maximum_feasibility_residual: result.maximum_feasibility_residual,
                maximum_kkt_residual: result.maximum_kkt_residual,
                uniform_unknown_programs: result.uniform_unknown_programs,
            });
            parameters = result.parameters;
            head = next;
        }
    }
    let final_ce = loss_source(&head, records, 0..records.len(), work)?;
    let audit = head.audit(domain, &parameters, work)?;
    Ok(Trained {
        config,
        parameters,
        head,
        initial_ce,
        final_ce,
        audit,
        steps,
        operations: work.used - start,
        target_realizability_certified: false,
        exact_real_projection_certified: false,
    })
}

fn workspace(
    domain: &Domain,
    records: Source<'_>,
    config: &Config,
) -> Result<(usize, usize, u128)> {
    records.admit(
        domain.shape,
        domain.unknown.len(),
        domain.head_columns_bound,
    )?;
    if records.is_empty() {
        return Err("empty original training records".into());
    }
    let batch = if config.mode == Mode::Mini {
        config.batch_size
    } else {
        records.len()
    };
    let step_count = config
        .epochs
        .checked_mul(records.len().div_ceil(batch))
        .ok_or("complete training step count overflow")?;
    // Reserve a complete dictionary rebuild, one expanded packed record,
    // all step metadata and class scratch before initialization as well.
    let fixed = memory::vector_bytes::<Step>(step_count)
        + 40 * domain.shape.rows() as u128
        + 16 * domain.shape.programs as u128
        + 2 * memory::vector_bytes::<f64>(domain.shape.classes)
        + 8192;
    Ok((batch, step_count, fixed))
}
