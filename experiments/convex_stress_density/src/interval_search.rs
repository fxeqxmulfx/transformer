//! Bounded training-only interval proposals, retaining the entire exact search.
//! Same label sketch on both axes, categorical decoding and ordinary CE; mixed
//! kernel columns are empirical candidates, never certified hidden modes.
use crate::{C, Model, Result, comparison, interval, interval_audit, random::Random, search};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    pub variants: Vec<interval::Options>,
    pub total_iterations: usize,
    pub model_audits: usize,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            variants: [0.03, 0.1, 0.3]
                .iter()
                .map(|&epsilon| interval::Options {
                    model_tolerance: epsilon,
                    iterations: 4096,
                    ..interval::Options::default()
                })
                .collect(),
            total_iterations: 73_728,
            model_audits: 8,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Input {
    pub categorical: search::Input,
    pub policy: Policy,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cost {
    pub repeated_sketches: usize,
    pub repeated_sketch_observations: usize,
    pub centering_cells: usize,
    pub iterations: usize,
    pub iterations_charged: usize,
    pub phase_products: usize,
    pub decoder_observations: usize,
    pub model_audits: usize,
    pub latent_frequency_audit_allowance: usize,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    pub exact_attempt: usize,
    pub variant: usize,
    pub reason: String,
    pub extraction: Option<interval_audit::Artifact>,
    pub model: Option<Model>,
    pub model_audit: Option<Value>,
    pub train_ce: Option<f64>,
    pub iterations_charged: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Choice {
    Exact(usize),
    Interval(usize),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    pub format: u32,
    pub input: Input,
    pub buffer_bound_bytes: u64,
    pub exact: search::Proposal,
    pub attempts: Vec<Attempt>,
    pub cost: Cost,
    pub selected: Option<Choice>,
}

pub fn propose(input: Input) -> Result<Proposal> {
    let cat = &input.categorical;
    let samples = search::validate(cat)?;
    let (n, vocab) = (cat.n, cat.vocab);
    let policy = &input.policy;
    if policy.variants.is_empty()
        || policy.variants.len() > 8
        || !(1..=131_072).contains(&policy.total_iterations)
        || !(1..=32).contains(&policy.model_audits)
        || policy.variants.iter().enumerate().any(|(i, o)| {
            policy.variants[..i]
                .iter()
                .any(|p| p.model_tolerance == o.model_tolerance)
        })
    {
        return Err("invalid bounded interval search policy".into());
    }
    for &options in &policy.variants {
        interval::validate(&interval::Input {
            n,
            rank: 1,
            features: vec![C::default(); n],
            rows: [0, 1],
            options,
        })?;
    }
    let exact = search::propose(cat.clone())?;
    let selected = exact.selected.map(Choice::Exact);
    let mut best = exact
        .selected
        .map_or(f64::INFINITY, |i| exact.attempts[i].train_ce.unwrap());
    let bound = exact.buffer_bound_bytes
        + 256 * 1024 * 1024
        + 512 * policy.total_iterations as u64
        + (exact.attempts.len() * policy.variants.len()) as u64 * (32 * n * n + 256 * n) as u64;
    let mut classes = vec![];
    for s in &samples {
        if !classes.contains(&s.label) {
            classes.push(s.label);
        }
    }
    let mut rng = Random::new(cat.options.seed);
    let mut out = Proposal {
        format: 1,
        buffer_bound_bytes: bound,
        input,
        exact,
        attempts: vec![],
        cost: Cost::default(),
        selected,
    };
    // The exact search's feature slots preserve trial/rank order. Rebuilding
    // each original sketch follows its identical observed-label RNG order.
    for trial in 0..out.input.categorical.options.code_trials {
        let s = search::sketch(n, vocab, &samples, &classes, &mut rng);
        out.cost.repeated_sketches += 1;
        out.cost.repeated_sketch_observations += 2 * samples.len();
        out.cost.centering_cells += 2 * n * n;
        for (index, old) in out
            .exact
            .attempts
            .iter()
            .enumerate()
            .filter(|(_, a)| a.trial == trial)
        {
            for (variant, &options) in out.input.policy.variants.iter().enumerate() {
                let mut attempt = Attempt {
                    exact_attempt: index,
                    variant,
                    reason: "missing_exact_subspace".into(),
                    extraction: None,
                    model: None,
                    model_audit: None,
                    train_ce: None,
                    iterations_charged: 0,
                };
                let Some(b) = &old.features else {
                    out.attempts.push(attempt);
                    continue;
                };
                let remaining = out.input.policy.total_iterations - out.cost.iterations_charged;
                if remaining == 0 {
                    attempt.reason = "interval_iteration_budget_exhausted".into();
                    out.attempts.push(attempt);
                    continue;
                }
                let options = interval::Options {
                    iterations: options.iterations.min(remaining),
                    ..options
                };
                let request = interval::Input {
                    n,
                    rank: old.rank,
                    features: b.clone(),
                    rows: [0, 1],
                    options,
                };
                let result = match interval::solve(request) {
                    Ok(value) => value,
                    Err(error) => {
                        attempt.reason = format!("interval_error: {error}");
                        attempt.iterations_charged = options.iterations;
                        out.cost.iterations_charged += options.iterations;
                        out.attempts.push(attempt);
                        continue;
                    }
                };
                attempt.iterations_charged = result.trace.len();
                out.cost.iterations += result.trace.len();
                out.cost.iterations_charged += result.trace.len();
                let raw = match interval_audit::endpoint(&result.input, &result) {
                    Ok(value) => value,
                    Err(error) => {
                        attempt.reason = format!("interval_raw_audit_failed: {error}");
                        attempt.extraction = Some(interval_audit::Artifact {
                            outcome: result,
                            raw_audit: Value::Null,
                        });
                        out.attempts.push(attempt);
                        continue;
                    }
                };
                attempt.reason = result.reason.clone();
                let left = result.kernel_column.clone();
                attempt.extraction = Some(interval_audit::Artifact {
                    outcome: result,
                    raw_audit: raw,
                });
                if let Some(left) = left {
                    let right: Vec<_> = (0..n)
                        .map(|j| {
                            (0..n)
                                .map(|i| s[i + n * j].conj() * left[i])
                                .sum::<C>()
                                .conj()
                        })
                        .collect();
                    out.cost.phase_products += n * n;
                    if let (Some(rows), Some(cols)) = (search::clock(&left), search::clock(&right))
                    {
                        out.cost.decoder_observations += samples.len();
                        if let Some(d) = search::decoder(n, vocab, &samples, &classes, &rows, &cols)
                        {
                            if out.cost.model_audits < out.input.policy.model_audits {
                                let model = Model::construct(vocab, rows, cols, d)?;
                                out.cost.model_audits += 1;
                                out.cost.latent_frequency_audit_allowance += n;
                                match model.audit(&samples) {
                                    Ok(audit) => {
                                        let ce = audit["train_ce"]
                                            .as_f64()
                                            .ok_or("missing interval model CE")?;
                                        attempt.train_ce = Some(ce);
                                        attempt.model_audit = Some(audit);
                                        attempt.reason = "audited_interval_phase_candidate".into();
                                        if ce + 1e-12 < best {
                                            best = ce;
                                            out.selected =
                                                Some(Choice::Interval(out.attempts.len()));
                                        }
                                    }
                                    Err(error) => {
                                        attempt.reason =
                                            format!("interval_latent_audit_failed: {error}")
                                    }
                                }
                                attempt.model = Some(model);
                            } else {
                                attempt.reason = "interval_model_audit_budget_exhausted".into();
                            }
                        } else {
                            attempt.reason = "uncovered_interval_decoder_state".into();
                        }
                    } else {
                        attempt.reason = "interval_phase_clock_not_injective".into();
                    }
                }
                out.attempts.push(attempt);
            }
        }
    }
    Ok(out)
}
pub fn replay(original: &Input, saved: &Proposal) -> Result<()> {
    if saved.format != 1 || &saved.input != original || propose(original.clone())? != *saved {
        return Err("interval proposal input or full replay mismatch".into());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetainedChoice {
    Earlier(comparison::Choice),
    Interval(usize),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    pub earlier: comparison::Comparison,
    pub proposal: Proposal,
    pub selected: Option<RetainedChoice>,
}
pub fn compare(proposal: Proposal, previous: Vec<Model>) -> Result<Comparison> {
    replay(&proposal.input, &proposal)?;
    let earlier = comparison::compare(proposal.exact.clone(), previous)?;
    let mut selected = earlier.selected.map(RetainedChoice::Earlier);
    let mut best = match earlier.selected {
        None => f64::INFINITY,
        Some(comparison::Choice::Prior(i)) => earlier.prior[i].train_ce,
        Some(comparison::Choice::Density(i)) => earlier.proposal.attempts[i].train_ce.unwrap(),
    };
    for (i, a) in proposal.attempts.iter().enumerate() {
        if let Some(ce) = a.train_ce
            && ce + 1e-12 < best
        {
            best = ce;
            selected = Some(RetainedChoice::Interval(i));
        }
    }
    Ok(Comparison {
        earlier,
        proposal,
        selected,
    })
}
pub fn replay_comparison(original: &Input, previous: &[Model], saved: &Comparison) -> Result<()> {
    if &saved.proposal.input != original
        || compare(saved.proposal.clone(), previous.to_vec())? != *saved
    {
        return Err("interval retained comparison replay mismatch".into());
    }
    Ok(())
}
