//! CM29 / Lean af4a6d14: exact observed row consequences for proposed labels.
//! The shared prefix plus each query's suffix is an actual RowOp program.
//! Correctness is conditional on an unknown injective code for the declared
//! observed class set and additive row/column realizability over ZMod q.
use crate::{Result, row_program};
use row_program::Op;
use serde::{Deserialize, Serialize};
mod audit;
mod plan;
mod run;
#[cfg(test)]
mod tests;
pub use audit::check;
pub use run::run;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub equations: usize,
    pub queries: usize,
    pub reduction_products: u64,
    pub certificate_products: u64,
    pub certificate_operations: usize,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            equations: 4096,
            queries: 257 * 257,
            reduction_products: 500_000_000,
            certificate_products: 500_000_000,
            certificate_operations: 2_000_000,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub n: usize,
    pub vocabulary: usize,
    pub modulus: u16,
    /// Cell = row + n*column. Identifiers have no arithmetic meaning.
    pub observations: Vec<[usize; 2]>,
    /// Candidate output table only. No unknown target or true code is supplied.
    pub predictions: Vec<usize>,
    pub prefix_policy: row_program::Policy,
    pub policy: Policy,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Term {
    pub row: usize,
    pub factor: u16,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    UnknownClass,
    Nondivisible {
        column: usize,
        value: u16,
        divisor: u16,
    },
    Unrepresented,
    ReductionCap,
    CertificateOperationCap,
    CertificateProductCap,
    Passed,
}
impl Decision {
    fn capped(&self) -> bool {
        matches!(
            self,
            Self::ReductionCap | Self::CertificateOperationCap | Self::CertificateProductCap
        )
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Query {
    pub cell: usize,
    pub label: usize,
    pub class: Option<usize>,
    pub terms: Vec<Term>,
    pub residual: Vec<u16>,
    pub receiver: Option<usize>,
    pub operations: Vec<Op>,
    pub decision: Decision,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub prefix: row_program::Work,
    pub query_rows: usize,
    pub pivot_tests: usize,
    pub reduction_products: u64,
    pub certificate_products: u64,
    pub certificate_operations: usize,
    pub coefficient_checks: u64,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Report {
    pub input: Request,
    pub buffer_preflight_bytes: u64,
    pub classes: Vec<usize>,
    pub prefix: row_program::Outcome,
    pub queries: Vec<Query>,
    pub certified: bool,
    pub reason: String,
    pub work: Work,
}
pub fn replay(input: &Request, out: &Report) -> Result<()> {
    if run(input.clone())? != *out {
        return Err("decoded original-data/prefix/query/certificate/work replay differs".into());
    }
    Ok(())
}
