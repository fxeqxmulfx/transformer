//! Actual projected ordinary CE in CM65/66 coordinates. On original anchor
//! records, the derivative of each shared mass is the sum of selected full
//! decoder derivatives; unknown rows have zero observed derivative. All logits
//! still come from complete rounded complex products. Floating trajectories
//! are measured against, not claimed bit-identical to, the dense implementation.
use crate::{
    domain::Domain,
    head::{Audit, Head},
    projection::{self, Target},
    *,
};
pub use dense::training::{Config, Mode, Step};

#[derive(Clone, Debug, PartialEq)]
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
    if records.is_empty() {
        return Err("empty ordinary CE batch".into());
    }
    let mut total = 0.;
    for row in records {
        Domain::record(head.shape, row)?;
        total += ce(&head.logits(&row.states, work)?, row.label);
    }
    Ok(total / records.len() as f64)
}
pub fn gradient(
    domain: &Domain,
    head: &Head,
    records: &[Record],
    work: &mut Work,
) -> Result<(f64, Parameters)> {
    if records.is_empty() || head.shape != domain.shape {
        return Err("invalid observed CE gradient batch".into());
    }
    domain.admit(records.len())?;
    work.charge(
        8 * domain.parameters() as u128
            + 16 * records.len() as u128 * domain.shape.programs as u128,
    )?;
    let (mut total, mut selected) = (0., 0.);
    let scale = 1. / records.len() as f64;
    for row in records {
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
    }
    let mut values = vec![0.; domain.parameters()];
    values[..domain.shape.programs].fill(selected);
    Ok((total * scale, Parameters { values }))
}
pub fn train(
    domain: &Domain,
    records: &[Record],
    config: Config,
    work: &mut Work,
) -> Result<Trained> {
    config.validate()?;
    let initial = domain.initial(config.initial_mass, work)?;
    train_from(domain, records, config, initial, work)
}
pub fn train_from(
    domain: &Domain,
    records: &[Record],
    config: Config,
    mut parameters: Parameters,
    work: &mut Work,
) -> Result<Trained> {
    config.validate()?;
    domain.admit(records.len())?;
    let start = work.used;
    if *domain
        != Domain::new(
            domain.shape,
            records,
            domain.minimum_mass,
            domain.maximum_mass,
            work,
        )?
    {
        return Err("reduced domain has constraints absent from original training records".into());
    }
    domain.check(&parameters, 1e-10, work)?;
    let mut head = Head::build(domain, &parameters, work)?;
    let initial_ce = loss(&head, records, work)?;
    let batch = if config.mode == Mode::Mini {
        config.batch_size
    } else {
        records.len()
    };
    let mut steps = Vec::with_capacity(config.epochs * records.len().div_ceil(batch));
    for epoch in 0..config.epochs {
        for first in (0..records.len()).step_by(batch) {
            let end = (first + batch).min(records.len());
            let rows = &records[first..end];
            let (before_ce, derivative) = gradient(domain, &head, rows, work)?;
            let proposal =
                Target::step(domain, &parameters, &derivative, config.learning_rate, work)?;
            let result = projection::project(domain, &proposal, work)?;
            let next = Head::build(domain, &result.parameters, work)?;
            let after_ce = loss(&next, rows, work)?;
            if after_ce > before_ce + 1e-8 {
                return Err("observed CE increased after weighted projection".into());
            }
            let change = domain
                .distance(&parameters, &result.parameters, work)?
                .sqrt();
            steps.push(Step {
                epoch,
                first_record: first,
                end_record_exclusive: end,
                before_ce,
                after_ce,
                before_mass: parameters.values[..domain.shape.programs].iter().sum(),
                after_mass: result.parameters.values[..domain.shape.programs]
                    .iter()
                    .sum(),
                parameter_change_l2: change,
                maximum_feasibility_residual: result.maximum_feasibility_residual,
                maximum_kkt_residual: result.maximum_kkt_residual,
                uniform_unknown_programs: result.uniform_unknown_programs,
            });
            parameters = result.parameters;
            head = next;
        }
    }
    let final_ce = loss(&head, records, work)?;
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
