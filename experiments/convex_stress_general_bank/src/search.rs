//! Try observed anchors by descending incident degree, then ascending total
//! anchor-fiber size and lexicographic cell. All criteria use training data.
//! The best complete certificate maximizes certified cells, breaking ties by
//! first discovery. An incomplete attempt supplies no answers or constraints.
use crate::*;
use propagation::{Certificate, Config, Costs, Observation, Outcome, Problem, Refusal, Verified};
use std::{cmp::Reverse, collections::BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub anchors: usize,
    pub sweeps: usize,
    /// Shared budget of producer decoder comparisons across every attempt.
    pub producer_checks: u64,
    pub branches: usize,
    pub memory_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Complete {
    pub coverage: usize,
    pub live_branches: usize,
    pub inconsistent: bool,
    pub certificate_sha256: String,
    pub costs: Costs,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttemptResult {
    Complete(Complete),
    Refused(Refusal),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attempt {
    pub config: Config,
    pub result: AttemptResult,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stop {
    FullCoverage,
    Inconsistent,
    Anchors,
    WorkBudget,
    ResourceRefusal,
}

/// Saved evidence is untrusted until replayed. Only `discover`/`replay` construct
/// a checked value, through the independent propagation verifier.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub limits: Limits,
    pub attempts: Vec<Attempt>,
    pub selected_attempt: Option<usize>,
    pub selected: Option<Certificate>,
    pub producer_checks: u64,
    pub checker_checks: u64,
    pub stop: Stop,
}

pub struct Discovery {
    pub record: Record,
    pub verified: Option<Verified>,
}

fn checks(costs: Costs) -> Result<u64> {
    costs
        .support_checks
        .checked_add(costs.output_checks)
        .ok_or_else(|| "cost overflow".into())
}

fn config(anchor: Observation, limits: Limits, remaining: u64) -> Config {
    Config {
        anchor,
        steps: limits.sweeps,
        max_checks: remaining,
        max_branches: limits.branches,
        memory_limit: limits.memory_bytes,
    }
}

pub fn discover(problem: &Problem, limits: Limits) -> Result<Discovery> {
    if limits.anchors == 0
        || limits.anchors > 4096
        || limits.producer_checks > 10_000_000_000
        || limits.memory_bytes > RUN_MEMORY_BYTES
        || limits.branches > 16_384
    {
        return Err("invalid anchor search limits".into());
    }
    let first = *problem
        .observations
        .iter()
        .min()
        .ok_or("empty anchor search")?;
    // Validate all dimensions and labels before degree arrays or canonical input
    // copies. The propagation forecast includes copies used by independent replay.
    let preflight = propagation::preflight(problem, config(first, limits, limits.producer_checks))?;
    if preflight.memory_bound > limits.memory_bytes {
        return Err("anchor search preflight exceeds its memory limit".into());
    }
    let unique: BTreeSet<_> = problem.observations.iter().copied().collect();
    let mut rows = vec![0_usize; problem.side];
    let mut cols = vec![0_usize; problem.side];
    let mut fibers = vec![0_usize; problem.vocab];
    for &t in &unique {
        rows[t.row] += 1;
        cols[t.column] += 1;
    }
    for d in &problem.decoders {
        for &y in d {
            fibers[y] += 1;
        }
    }
    let mut anchors: Vec<_> = unique.into_iter().collect();
    anchors.sort_by_key(|t| (Reverse(rows[t.row] + cols[t.column]), fibers[t.label], *t));
    anchors.truncate(limits.anchors);
    let mut result = Discovery {
        record: Record {
            limits,
            attempts: vec![],
            selected_attempt: None,
            selected: None,
            producer_checks: 0,
            checker_checks: 0,
            stop: Stop::Anchors,
        },
        verified: None,
    };
    let mut best_coverage = 0;
    for anchor in anchors {
        let remaining = limits.producer_checks - result.record.producer_checks;
        let c = config(anchor, limits, remaining);
        // Candidate anchors can have larger fibers than the first observation.
        // Check joint storage for the old best and the new full certificate too.
        let flight = propagation::preflight(problem, c)?;
        let previous = result
            .record
            .selected
            .as_ref()
            .map_or(0, |cert| cert.memory_bound);
        let required = flight
            .memory_bound
            .checked_mul(2)
            .and_then(|x| x.checked_add(previous))
            .and_then(|x| x.checked_add(64 * 1024 * 1024))
            .ok_or("memory bound overflow")?;
        if required > limits.memory_bytes {
            result.record.attempts.push(Attempt {
                config: c,
                result: AttemptResult::Refused(Refusal {
                    reason: propagation::RefusalReason::Memory {
                        required,
                        limit: limits.memory_bytes,
                    },
                    costs: Costs::default(),
                    completed_branches: 0,
                    total_branches: flight.total_branches,
                }),
            });
            result.record.stop = Stop::ResourceRefusal;
            break;
        }
        let outcome = propagation::propagate(problem, c)?;
        match outcome {
            Outcome::Refused(refusal) => {
                result.record.producer_checks += checks(refusal.costs)?;
                result.record.stop = if refusal.reason == propagation::RefusalReason::WorkBudget {
                    Stop::WorkBudget
                } else {
                    Stop::ResourceRefusal
                };
                result.record.attempts.push(Attempt {
                    config: c,
                    result: AttemptResult::Refused(refusal),
                });
                break;
            }
            Outcome::Complete(cert) => {
                let verified = propagation::verify(problem, c, &cert)?;
                let work = checks(cert.costs)?;
                result.record.producer_checks += work;
                result.record.checker_checks = result
                    .record
                    .checker_checks
                    .checked_add(work)
                    .ok_or("checker cost overflow")?;
                let coverage = cert.answers.iter().filter(|a| a.len() == 1).count();
                let inconsistent = verified.inconsistent();
                let summary = Complete {
                    coverage,
                    inconsistent,
                    live_branches: cert.branches.iter().filter(|b| b.live).count(),
                    costs: cert.costs,
                    certificate_sha256: convex_stress_relabel::hash(&serde_json::to_vec(&cert)?),
                };
                result.record.attempts.push(Attempt {
                    config: c,
                    result: AttemptResult::Complete(summary),
                });
                if inconsistent || result.verified.is_none() || coverage > best_coverage {
                    result.record.selected_attempt = Some(result.record.attempts.len() - 1);
                    result.record.selected = Some(*cert);
                    result.verified = Some(verified);
                    best_coverage = coverage;
                }
                if inconsistent || coverage == problem.side * problem.side {
                    result.record.stop = if inconsistent {
                        Stop::Inconsistent
                    } else {
                        Stop::FullCoverage
                    };
                    break;
                }
                if result.record.producer_checks == limits.producer_checks {
                    result.record.stop = Stop::WorkBudget;
                    break;
                }
            }
        }
    }
    Ok(result)
}

/// Recompute every attempt and the deterministic choice, not only its winner.
pub fn replay(problem: &Problem, record: &Record) -> Result<Discovery> {
    let actual = discover(problem, record.limits)?;
    if actual.record != *record {
        return Err("anchor discovery evidence changed".into());
    }
    Ok(actual)
}
