//! Actual full joint projected CE updates. The analytic derivative is that
//! of U (U* gamma), whose Gram identity is proved in CM64; its probabilities
//! are computed from complete saved complex products. No factors are trained
//! independently and subsequently multiplied. This is not a convergence proof.
use crate::{
    domain::Domain,
    head::{Head, HeadAudit},
    projection, *,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    Forward,
    Full,
    Mini,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub mode: Mode,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub initial_mass: f64,
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.epochs > 128
            || (self.mode == Mode::Forward) != (self.epochs == 0)
            || !(1..=65536).contains(&self.batch_size)
            || !self.learning_rate.is_finite()
            || !(1e-6..=1e6).contains(&self.learning_rate)
        {
            return Err("invalid bounded joint training policy".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub epoch: usize,
    pub first_record: usize,
    pub end_record_exclusive: usize,
    pub before_ce: f64,
    pub after_ce: f64,
    pub before_mass: f64,
    pub after_mass: f64,
    pub parameter_change_l2: f64,
    pub maximum_feasibility_residual: f64,
    pub maximum_kkt_residual: f64,
    pub uniform_unknown_programs: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trained {
    pub config: Config,
    pub parameters: Parameters,
    pub head: Head,
    pub initial_ce: f64,
    pub final_ce: f64,
    pub audit: HeadAudit,
    pub steps: Vec<Step>,
    pub operations: u64,
    pub target_realizability_certified: bool,
    pub exact_real_projection_certified: bool,
}

pub fn loss(head: &Head, records: &[Record], work: &mut Work) -> Result<f64> {
    if records.is_empty() {
        return Err("empty CE record batch".into());
    }
    let mut total = 0.;
    for row in records {
        Domain::record(head.shape, row)?;
        total += ce(&head.logits(&row.states, work)?, row.label);
    }
    Ok(total / records.len() as f64)
}
pub fn gradient(head: &Head, records: &[Record], work: &mut Work) -> Result<(f64, Parameters)> {
    if records.is_empty() {
        return Err("empty CE gradient batch".into());
    }
    let sh = head.shape;
    sh.admit(records.len())?;
    work.charge(
        sh.parameters() as u128
            + 12 * records.len() as u128 * sh.programs as u128 * sh.classes as u128,
    )?;
    let mut values = vec![0.; sh.parameters()];
    let mut total = 0.;
    let scale = 1. / records.len() as f64;
    for row in records {
        Domain::record(sh, row)?;
        let logits = head.logits(&row.states, work)?;
        total += ce(&logits, row.label);
        let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let probabilities: Vec<_> = logits.iter().map(|z| (z - top).exp()).collect();
        let normalizer = probabilities.iter().sum::<f64>();
        for (p, &s) in row.states.iter().enumerate() {
            for (c, &probability) in probabilities.iter().enumerate() {
                values[sh.at(p, s, c)] +=
                    (probability / normalizer - if c == row.label { 1. } else { 0. }) * scale;
            }
        }
    }
    Ok((total * scale, Parameters { values }))
}

pub fn train(
    domain: &Domain,
    records: &[Record],
    config: Config,
    work: &mut Work,
) -> Result<Trained> {
    config.validate()?;
    let sh = domain.shape;
    sh.admit(records.len())?;
    work.charge(8 * (sh.parameters() as u128 + records.len() as u128 * sh.programs as u128))?;
    // Reject additional decoder constraints absent from the supplied training
    // records. The external adapter must pass only original training rows.
    if *domain != Domain::new(sh, records, domain.minimum_mass, domain.maximum_mass)? {
        return Err("joint domain contains constraints absent from training records".into());
    }
    let start = work.used;
    let mut parameters = domain.initial(config.initial_mass)?;
    let mut head = Head::build(sh, &parameters, work)?;
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
            let (before_ce, derivative) = gradient(&head, rows, work)?;
            work.charge(4 * sh.parameters() as u128)?;
            let proposal = Parameters {
                values: parameters
                    .values
                    .iter()
                    .zip(&derivative.values)
                    .map(|(q, g)| q - config.learning_rate * g)
                    .collect(),
            };
            let result = projection::project(domain, &proposal, work)?;
            let next = Head::build(sh, &result.parameters, work)?;
            let after_ce = loss(&next, rows, work)?;
            if after_ce > before_ce + 1e-8 {
                return Err("observed joint CE increased after projection".into());
            }
            let change = parameters
                .values
                .iter()
                .zip(&result.parameters.values)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            steps.push(Step {
                epoch,
                first_record: first,
                end_record_exclusive: end,
                before_ce,
                after_ce,
                before_mass: parameters.values[..sh.programs].iter().sum(),
                after_mass: result.parameters.values[..sh.programs].iter().sum(),
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
    let audit = head.audit(&parameters, work)?;
    let bound = 2. * domain.maximum_mass * sh.states as f64;
    if audit
        .column_l1
        .iter()
        .any(|&v| !v.is_finite() || v > bound + 1e-7)
    {
        return Err("complete head exceeds the CM64 column bound".into());
    }
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
