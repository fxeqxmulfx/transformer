//! Exact two-mode endpoint inventory, original IDs and uncollapsed raw scores.
use crate::io::Artifact;
use joint::{
    head::Audit,
    training::{Config, Mode, Step},
};
use model::{
    features::{Alphabet, Program},
    input::Observation,
    partition::Partition,
};
use probes::scoring::Score;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub schema: u32,
    pub producer: String,
    pub data_inventory: Artifact,
    pub required_modes: [Mode; 2],
    pub process_bytes: u64,
    pub total_ram_bytes: u64,
    pub file_max: u64,
    pub run_max: u64,
    pub per_run_forecast: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Family {
    pub alphabet: Alphabet,
    pub programs: Vec<Program>,
    pub partition: Partition,
    pub observations: Vec<Observation>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub config: Config,
    pub checkpoint: Artifact,
    pub payload_sha256: String,
    pub complete_head_sha256: String,
    pub initial_ce: f64,
    pub final_ce: f64,
    pub audit: Audit,
    pub steps: Vec<Step>,
    pub operations: u64,
    pub logical_parameters: usize,
    pub unknown_rows: usize,
    pub target_realizability_certified: bool,
    pub exact_real_projection_certified: bool,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub name: String,
    pub refusal: Option<String>,
    pub family: Option<Family>,
    pub endpoints: Vec<Endpoint>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Seal {
    pub schema: u32,
    pub entries: Vec<String>,
    pub artifacts: BTreeMap<String, Artifact>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitScore {
    pub split: String,
    pub trace: Artifact,
    pub calls: u64,
    pub score: Score,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModeScore {
    pub mode: Mode,
    pub splits: Vec<SplitScore>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scored {
    pub name: String,
    pub seal: Artifact,
    pub refusal: Option<String>,
    pub modes: Vec<ModeScore>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Replayed {
    pub name: String,
    pub score: Artifact,
    pub raw_calls: u64,
}
