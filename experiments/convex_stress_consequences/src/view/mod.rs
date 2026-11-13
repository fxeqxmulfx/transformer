//! CM29 / Lean 6eb80433: decoded row consequences through a fixed categorical
//! view, with a separate output vocabulary and an explicitly assumed strip.
//! Each view has a separate dense report so a cohort can stream them to disk.
use crate::{Result, decoded, learner};
use convex_stress_tagged::{Model, Sample, views};
use serde::{Deserialize, Serialize};
use serde_json::Value;
mod audit;
mod binding;
mod plan;
mod run;
#[cfg(test)]
mod tests;
pub use audit::check;
pub use plan::catalogue;
pub use run::run;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Input {
    pub categorical: learner::Request,
    pub minimum_strip_support: usize,
    pub max_views: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub input: Input,
    pub view: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Disposition {
    Ready { buffer_preflight_bytes: u64 },
    ViewCap,
    EmptyInterior,
    PreflightRefusal { error: String },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Slot {
    pub view: views::View,
    pub disposition: Disposition,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Origin {
    ObservedConsequence { interior_cell: usize, query: usize },
    UncertifiedInterior { interior_cell: usize },
    AssumedConstantStrip,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scope {
    pub theorem: String,
    pub target_family: String,
    pub modulus: usize,
    pub class_to_output: Vec<usize>,
    pub row_map: Vec<Option<usize>>,
    pub column_map: Vec<Option<usize>>,
    pub constant: Option<usize>,
    pub origins: Vec<Origin>,
    pub certified_interior_cells: usize,
    pub assumed_strip_cells: usize,
    pub uncertified_cells: usize,
    pub full_table_certified_within_family: bool,
    pub unconditional_identification: bool,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub interior: usize,
    pub duplicate_of: Option<usize>,
    pub model: Model,
    pub raw_audit: Value,
    pub scope: Scope,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    Admitted { candidate: usize },
    RawRefusal { error: String },
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub interior: Option<learner::Work>,
    pub lifts: usize,
    pub full_prediction_cells: usize,
    pub raw_audits: usize,
    pub raw_frequency_extents: Vec<Option<usize>>,
    pub completed_raw_frequencies: usize,
    pub scope_cells: usize,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub input: Request,
    pub slot: Slot,
    pub interior: Option<learner::Report>,
    pub decisions: Vec<Decision>,
    pub candidates: Vec<Candidate>,
    pub selected: Option<usize>,
    pub work: Work,
}
pub fn replay(input: &Request, out: &Report) -> Result<()> {
    if run(input.clone())? != *out {
        return Err("fixed view original-data/model/certificate/work replay differs".into());
    }
    Ok(())
}
