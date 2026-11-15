//! Same joint CM57 projected CE update as the ungrouped trainer at819d4547,
//! using CM60 exact batch weights. Forward performs projection but zero
//! gradient updates. CM61 at822bb5e8 permits an explicit anchor selection;
//! every original observed label and multiplicity still enters ordinary CE.
use crate::{
    corpus::{Batch, Corpus},
    domain::{Anchors, Domain, HeadSpec},
    objective, *,
};
use joint::{
    constraints::Projection,
    head::Head,
    training::{Mode, Policy},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Forecast {
    pub scalar_work: u64,
    pub buffer_bytes: u64,
    pub gradient_steps: usize,
    pub maximum_weighted_visits: usize,
}

pub fn forecast(corpus: &Corpus, policy: &Policy) -> Result<Forecast> {
    corpus.validate()?;
    let limits = corpus.limits();
    let (n, g, p, v) = (
        limits.records,
        corpus.groups().len(),
        limits.programs,
        limits.vocabulary,
    );
    // Retain the reviewed policy validation, including genuine zero-step forward.
    // A single distinct group is currently an explicit domain-constructor refusal.
    policy.forecast(g, p, v)?;
    let batches = if policy.mode == Mode::Mini {
        n.div_ceil(policy.batch)
    } else {
        1
    };
    let steps = batches * policy.epochs;
    let visits = policy.epochs
        * if policy.mode == Mode::Mini {
            n.min(batches * g.min(policy.batch))
        } else {
            g
        };
    let (nn, gg, pp, vv) = (n as u128, g as u128, p as u128, v as u128);
    let scalar = 512 * nn * pp * (policy.epochs as u128 + 5)
        + (4 * gg + 8 * visits as u128) * pp * (256 * vv + 512)
        + (steps as u128 + 1)
            * (2048 * gg * pp + policy.projection_cycles as u128 * (16384 * pp + 64 * gg * pp));
    let bytes = u128::from(limits.buffer_bytes()?)
        + u128::from(joint::data::forecast(g, p, v)?)
        + 32 * (steps as u128 * (256 + 128 * pp) + 16 * visits as u128);
    if scalar > 600_000_000_000_000 || bytes > joint::MAX_BYTES {
        return Err("complete weighted training work or buffer forecast refused".into());
    }
    Ok(Forecast {
        scalar_work: scalar as u64,
        buffer_bytes: bytes as u64,
        gradient_steps: steps,
        maximum_weighted_visits: visits,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub epoch: usize,
    /// Original IDs are retained once in the bound corpus, addressed by this range.
    pub batch: Batch,
    pub before_ce: f64,
    pub after_ce: f64,
    pub gradient: Vec<Parameters>,
    pub projection: Projection,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Work {
    pub gradient_batches: usize,
    pub gradient_records: usize,
    pub weighted_gradient_groups: usize,
    pub effective_updates: usize,
    pub projection_cycles: usize,
    pub complete_logit_products: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    pub specification: HeadSpec,
    pub corpus_sha256: String,
    pub original_records: usize,
    pub distinct_groups: usize,
    pub policy: Policy,
    pub forecast: Forecast,
    pub domain: Domain,
    pub initial_projection: Projection,
    pub initial_ce: f64,
    pub final_ce: f64,
    pub steps: Vec<Step>,
    pub work: Work,
    pub head: Head,
}

pub fn train(
    corpus: &Corpus,
    coordinates: Vec<f64>,
    budgets: Vec<f64>,
    radius: f64,
    minimum_mass: f64,
    policy: Policy,
) -> Result<Outcome> {
    fit(
        corpus,
        &HeadSpec {
            coordinates,
            budgets,
            radius,
            minimum_mass,
            anchors: Anchors::All,
        },
        policy,
    )
}

/// Fix the anchor selection before the joint CE stage. It changes only the
/// domain; all original observed records remain in every complete epoch.
pub fn fit(corpus: &Corpus, specification: &HeadSpec, policy: Policy) -> Result<Outcome> {
    let forecast = forecast(corpus, &policy)?;
    if specification.coordinates.len() != corpus.limits().vocabulary
        || specification.budgets.len() != corpus.limits().programs
    {
        return Err("training head dimensions differ from the original corpus".into());
    }
    // CM60 constraints_iff justifies removing repeated identical anchors.
    // Label is part of the grouping key: conflicts cannot be hidden here.
    let domain = Domain::new(corpus, specification)?;
    let mut initial = vec![[0.; 6]; specification.budgets.len()];
    let initial_mass = policy.initial_total_mass / initial.len() as f64;
    for theta in &mut initial {
        theta[5] = initial_mass;
    }
    let initial_projection = domain.project(
        &initial,
        policy.projection_cycles,
        policy.projection_tolerance,
    )?;
    let mut head = Head::new(
        specification.coordinates.clone(),
        initial_projection.theta.clone(),
    )?;
    let n = corpus.limits().records;
    let whole = corpus.batch(0..n)?;
    let initial_ce = objective::score(corpus, &head, &whole)?;
    let mut steps = Vec::with_capacity(forecast.gradient_steps);
    let mut work = Work {
        projection_cycles: initial_projection.cycles,
        ..Work::default()
    };
    let size = if policy.mode == Mode::Mini {
        policy.batch
    } else {
        n
    };
    for epoch in 0..policy.epochs {
        for start in (0..n).step_by(size) {
            let batch = corpus.batch(start..(start + size).min(n))?;
            let (before_ce, gradient) = objective::gradient(corpus, &head, &batch)?;
            let mut proposal = head.theta.clone();
            for (theta, grad) in proposal.iter_mut().zip(&gradient) {
                for j in 0..6 {
                    theta[j] -= policy.learning_rate * grad[j];
                }
            }
            let projection = domain.project(
                &proposal,
                policy.projection_cycles,
                policy.projection_tolerance,
            )?;
            work.effective_updates += usize::from(projection.theta != head.theta);
            work.projection_cycles += projection.cycles;
            head = Head::new(head.coordinates, projection.theta.clone())?;
            let after_ce = objective::score(corpus, &head, &batch)?;
            work.gradient_batches += 1;
            work.gradient_records += batch.records();
            work.weighted_gradient_groups += batch.counts.len();
            steps.push(Step {
                epoch,
                batch,
                before_ce,
                after_ce,
                gradient,
                projection,
            });
        }
    }
    let final_ce = objective::score(corpus, &head, &whole)?;
    work.complete_logit_products = 16
        * head.theta.len() as u64
        * head.coordinates.len() as u64
        * (2 * whole.counts.len() as u64 + 2 * work.weighted_gradient_groups as u64);
    if work.gradient_batches != forecast.gradient_steps
        || work.gradient_records != n * policy.epochs
        || work.weighted_gradient_groups > forecast.maximum_weighted_visits
    {
        return Err("weighted work differs from original record accounting".into());
    }
    Ok(Outcome {
        specification: specification.clone(),
        corpus_sha256: corpus.fingerprint()?,
        original_records: n,
        distinct_groups: corpus.groups().len(),
        policy,
        forecast,
        domain,
        initial_projection,
        initial_ce,
        final_ce,
        steps,
        work,
        head,
    })
}

pub fn replay(saved: &Outcome, corpus: &Corpus) -> Result<()> {
    saved.head.validate()?;
    if corpus.fingerprint()? != saved.corpus_sha256 {
        return Err("original grouped corpus binding changed".into());
    }
    let rebuilt = fit(corpus, &saved.specification, saved.policy.clone())?;
    if *saved != rebuilt {
        return Err("weighted training differs from deterministic replay".into());
    }
    Ok(())
}
