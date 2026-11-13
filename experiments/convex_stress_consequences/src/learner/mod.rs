//! CM27 observed code discovery, CM25 raw complex moments/ordinary CE and
//! CM29 decoded certificates. Class names are mapped in observed order.
//! Discrete proposal search is separate from convex prediction-space CE;
//! an integer certificate identifies answers only in its declared family.
use crate::{Result, component, composite, decoded, row_program};
use convex_stress_modular::{Model, Sample, System, coding::Partition};
use serde::{Deserialize, Serialize};
use serde_json::Value;
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
    pub models: usize,
    pub component: component::Policy,
    pub composite: composite::Policy,
    pub prefix: row_program::Policy,
    pub decoded: decoded::Policy,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            equations: 4096,
            models: 2,
            component: component::Policy::default(),
            composite: composite::Policy::default(),
            prefix: row_program::Policy::default(),
            decoded: decoded::Policy::default(),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub n: usize,
    pub vocabulary: usize,
    pub observations: Vec<[usize; 2]>,
    pub policy: Policy,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    ObservationConflict { observation: usize },
    Duplicate { candidate: usize },
    ModelCap,
    RawRefusal { error: String },
    Admitted { candidate: usize },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attempt {
    pub branch: usize,
    pub decision: Decision,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub branch: usize,
    pub model: Model,
    pub raw_audit: Value,
    pub certificate: decoded::Report,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub composite: Option<composite::Work>,
    pub observation_checks: usize,
    pub model_constructions: usize,
    pub decoder_coordinates: usize,
    pub prediction_cells: usize,
    pub duplicate_tests: usize,
    pub raw_audits: usize,
    pub completed_raw_frequencies: usize,
    /// A numerical refusal has unknown partial extent, bounded by q. It is
    /// retained as None rather than incorrectly charged as zero frequencies.
    pub raw_frequency_extents: Vec<Option<usize>>,
    pub certificates: Vec<decoded::Work>,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub input: Request,
    pub buffer_preflight_bytes: u64,
    pub partition: Partition,
    pub construction: Option<composite::Report>,
    pub attempts: Vec<Attempt>,
    pub candidates: Vec<Candidate>,
    pub selected: Option<usize>,
    pub reason: String,
    pub work: Work,
}
pub fn replay(input: &Request, out: &Report) -> Result<()> {
    if run(input.clone())? != *out {
        return Err(
            "learner original-data/codes/models/raw-audit/certificate/work replay differs".into(),
        );
    }
    Ok(())
}
