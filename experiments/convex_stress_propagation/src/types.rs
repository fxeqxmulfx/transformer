use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub row: usize,
    pub column: usize,
    pub label: usize,
}

/// Only categorical training cells, dimensions and the declared fixed bank.
/// Class names carry no numerical meaning; latent states are residues modulo q.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Problem {
    pub side: usize,
    pub states: usize,
    pub vocab: usize,
    pub decoders: Vec<Vec<usize>>,
    pub observations: Vec<Observation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub anchor: Observation,
    pub steps: usize,
    pub max_checks: u64,
    pub max_branches: usize,
    pub memory_limit: u64,
}

impl Config {
    pub fn new(anchor: Observation, steps: usize) -> Self {
        Self {
            anchor,
            steps,
            max_checks: 10_000_000,
            max_branches: 16_384,
            memory_limit: crate::RUN_MEMORY_BYTES,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Costs {
    pub support_checks: u64,
    pub output_checks: u64,
}

impl Costs {
    pub(crate) fn tick(&mut self, output: bool, maximum: u64) -> Option<()> {
        if self.support_checks.checked_add(self.output_checks)? >= maximum {
            return None;
        }
        if output {
            self.output_checks += 1;
        } else {
            self.support_checks += 1;
        }
        Some(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Domains {
    pub rows: Vec<Vec<usize>>,
    pub columns: Vec<Vec<usize>>,
}

impl Domains {
    pub(crate) fn live(&self) -> bool {
        self.rows.iter().chain(&self.columns).all(|v| !v.is_empty())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Branch {
    pub decoder: usize,
    pub anchor_state: usize,
    pub domains: Domains,
    pub removed_per_sweep: Vec<usize>,
    pub live: bool,
}

/// Untrusted saved evidence. Only `verify` can produce a `Verified` value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Certificate {
    pub version: u32,
    pub problem: Problem,
    pub config: Config,
    pub memory_bound: u64,
    pub branches: Vec<Branch>,
    /// Row-major cells; each answer set is sorted and contains no duplicates.
    pub answers: Vec<Vec<usize>>,
    pub costs: Costs,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefusalReason {
    Memory { required: u64, limit: u64 },
    BranchBudget { required: usize, limit: usize },
    WorkBudget,
}

/// A budget refusal contains no purported answer sets, including partial ones.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Refusal {
    pub reason: RefusalReason,
    pub costs: Costs,
    pub completed_branches: usize,
    pub total_branches: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Complete(Box<Certificate>),
    Refused(Refusal),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Preflight {
    pub memory_bound: u64,
    pub total_branches: usize,
}
