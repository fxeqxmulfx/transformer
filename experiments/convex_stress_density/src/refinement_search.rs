//! CM23: add certified local proposals to a fully retained interval comparison.
//! Earlier-first CE ties, explicit budgets, no original-task or hidden-label inputs.
use crate::{Model, Result, comparison, interval_search, refinement, refinement_audit};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub rounds: usize,
    pub seeds: usize,
    pub assignments: usize,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            rounds: 24,
            seeds: 16,
            assignments: 256,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Source {
    Prior(usize),
    Exact(usize),
    Interval(usize),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Choice {
    Earlier(interval_search::RetainedChoice),
    Refined(usize),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    pub source: Source,
    pub reason: String,
    pub outcome: Option<refinement::Outcome>,
    pub audit: Option<Value>,
    pub assignments_charged: usize,
    pub train_ce: Option<f64>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub format: u32,
    pub parent: interval_search::Comparison,
    pub policy: Policy,
    pub attempts: Vec<Attempt>,
    pub selected: Option<Choice>,
    pub seeds_run: usize,
    pub assignments_charged: usize,
    pub endpoint_audits: usize,
    pub latent_frequency_audit_allowance: usize,
    pub buffer_bound_bytes: u64,
}
fn source_model(parent: &interval_search::Comparison, source: Source) -> Option<&Model> {
    match source {
        Source::Prior(i) => Some(&parent.earlier.prior[i].model),
        Source::Exact(i) => parent.proposal.exact.attempts[i]
            .train_ce
            .and(parent.proposal.exact.attempts[i].model.as_ref()),
        Source::Interval(i) => parent.proposal.attempts[i]
            .train_ce
            .and(parent.proposal.attempts[i].model.as_ref()),
    }
}
fn earlier_ce(parent: &interval_search::Comparison) -> f64 {
    use interval_search::RetainedChoice as R;
    match parent.selected {
        None => f64::INFINITY,
        Some(R::Earlier(comparison::Choice::Prior(i))) => parent.earlier.prior[i].train_ce,
        Some(R::Earlier(comparison::Choice::Density(i))) => {
            parent.proposal.exact.attempts[i].train_ce.unwrap()
        }
        Some(R::Interval(i)) => parent.proposal.attempts[i].train_ce.unwrap(),
    }
}
pub fn run(
    original: &interval_search::Input,
    prior: &[Model],
    parent: interval_search::Comparison,
    policy: Policy,
) -> Result<Report> {
    if !(1..=128).contains(&policy.rounds)
        || !(1..=32).contains(&policy.seeds)
        || !(2..=8192).contains(&policy.assignments)
    {
        return Err("invalid refinement search policy".into());
    }
    interval_search::replay_comparison(original, prior, &parent)?;
    let mut sources = vec![];
    sources.extend((0..parent.earlier.prior.len()).map(Source::Prior));
    sources.extend((0..parent.proposal.exact.attempts.len()).map(Source::Exact));
    sources.extend((0..parent.proposal.attempts.len()).map(Source::Interval));
    let mut best = earlier_ce(&parent);
    let selected = parent.selected.map(Choice::Earlier);
    // Bounded traces keep O(rounds*n) coordinates and integer duals per seed.
    // Parent reports, repeated replay and all numerical work need the process cap too.
    let bound = 4 * parent.proposal.buffer_bound_bytes
        + 256 * 1024 * 1024
        + policy.seeds as u64 * (policy.rounds as u64 * 65536 + 4 * 1024 * 1024);
    let mut out = Report {
        format: 1,
        parent,
        policy,
        attempts: vec![],
        selected,
        seeds_run: 0,
        assignments_charged: 0,
        endpoint_audits: 0,
        latent_frequency_audit_allowance: 0,
        buffer_bound_bytes: bound,
    };
    for source in sources {
        let mut attempt = Attempt {
            source,
            reason: "source_without_audited_model".into(),
            outcome: None,
            audit: None,
            assignments_charged: 0,
            train_ce: None,
        };
        if let Some(seed) = source_model(&out.parent, source) {
            let remaining = policy.assignments - out.assignments_charged;
            if out.seeds_run >= policy.seeds {
                attempt.reason = "seed_budget_exhausted".into();
            } else if remaining < 2 {
                attempt.reason = "assignment_budget_exhausted".into();
            } else {
                let rounds = policy.rounds.min(remaining / 2);
                let input = refinement::Input {
                    seed: seed.clone(),
                    observations: original.categorical.observations.clone(),
                    rounds,
                };
                out.seeds_run += 1;
                match refinement::run(input.clone()) {
                    Err(error) => {
                        attempt.assignments_charged = 2 * rounds;
                        attempt.reason = format!("refinement_error: {error}");
                    }
                    Ok(value) => {
                        attempt.assignments_charged = value.cost.assignments;
                        attempt.reason = value.reason.clone();
                        if value.best.is_some() {
                            out.endpoint_audits += 1;
                            out.latent_frequency_audit_allowance += input.seed.q;
                        }
                        match refinement_audit::check(&input, &value) {
                            Err(error) => {
                                attempt.reason = format!("refinement_raw_audit_failed: {error}")
                            }
                            Ok(audit) => {
                                if let Some(ce) = audit["endpoint"]["train_ce"].as_f64() {
                                    attempt.train_ce = Some(ce);
                                    if ce + 1e-12 < best {
                                        best = ce;
                                        out.selected = Some(Choice::Refined(out.attempts.len()));
                                    }
                                }
                                attempt.audit = Some(audit);
                            }
                        }
                        attempt.outcome = Some(value);
                    }
                }
                out.assignments_charged += attempt.assignments_charged;
            }
        }
        out.attempts.push(attempt);
    }
    Ok(out)
}
pub fn replay(
    original: &interval_search::Input,
    prior: &[Model],
    parent: &interval_search::Comparison,
    saved: &Report,
) -> Result<()> {
    if saved.format != 1
        || &saved.parent != parent
        || run(original, prior, parent.clone(), saved.policy)? != *saved
    {
        return Err("refinement search original input, parent or full replay mismatch".into());
    }
    Ok(())
}
