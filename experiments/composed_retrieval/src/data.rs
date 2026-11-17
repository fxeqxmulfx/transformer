//! The harness owns the case ID, source fingerprints, selection and budgets.
//! Only original training examples and object dimensions reach the optimizer.
use crate::{Result, model::WIDTH};
use baseline::{data::Task, hash};
use learned_retrieval::cohort::SUITE;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

pub const BASE_PROTOCOL: &str = include_str!("../protocol.json");
pub const DENSE_PROTOCOL: &str = include_str!("../dense/protocol.json");
const LOGODDS_PROTOCOL: &str = include_str!("../logodds/protocol.json");
pub const TRAINING_PROTOCOL: &str = if cfg!(feature = "logodds-depth") {
    LOGODDS_PROTOCOL
} else if cfg!(feature = "dense-forward") {
    DENSE_PROTOCOL
} else {
    BASE_PROTOCOL
};
pub const ARITHMETIC_PROTOCOL: &str = include_str!("../arithmetic/protocol.json");
pub const JOINT_PROTOCOL: &str = include_str!("../logodds/arithmetic/protocol.json");
pub const PROTOCOL: &str = if cfg!(feature = "output-offsets") {
    include_str!("../output_offsets/protocol.json")
} else if cfg!(all(
    feature = "arithmetic-cohort",
    feature = "logodds-depth"
)) {
    JOINT_PROTOCOL
} else if cfg!(feature = "arithmetic-cohort") {
    ARITHMETIC_PROTOCOL
} else {
    TRAINING_PROTOCOL
};
#[derive(Clone, Deserialize)]
pub struct CaseSpec {
    pub vocabulary: usize,
    pub context: usize,
    pub batch: usize,
    pub train: usize,
    pub selection: String,
    pub metric: String,
    pub target: f64,
    pub metadata_sha256: Option<String>,
    pub source_fingerprints: Option<Value>,
}
#[derive(Clone, Deserialize)]
pub struct Protocol {
    pub name: String,
    pub suite_sha256: String,
    pub catalog_sha256: String,
    pub cases: BTreeMap<String, CaseSpec>,
    pub data_seed: u32,
    pub seeds: Vec<u32>,
    pub modes: Vec<String>,
    pub updates: usize,
    pub width: usize,
    pub capacity: f64,
    pub output_offset_bound: Option<f64>,
    pub initial_key_scale: f64,
    pub initial_depth: f64,
    pub initial_log_odds: Option<f64>,
    pub initial_product: f64,
    pub angle_lr: f64,
    pub routing_lr: f64,
    pub recency_lr: f64,
    pub gate_lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub query_step: f64,
    /// A count selects a seeded cycle in each table. None visits every model coordinate.
    pub coordinates_per_table: Option<usize>,
}
pub fn protocol() -> Result<Protocol> {
    let mut p: Protocol = serde_json::from_str(BASE_PROTOCOL)?;
    if cfg!(feature = "dense-forward") {
        let dense: Value = serde_json::from_str(DENSE_PROTOCOL)?;
        if dense["parent_protocol_sha256"] != hash(BASE_PROTOCOL.as_bytes())
            || dense["forward_coordinates"] != "all"
        {
            return Err("dense comparison or its frozen parent changed".into());
        }
        p.name = dense["name"]
            .as_str()
            .ok_or("missing comparison name")?
            .into();
        p.coordinates_per_table = None;
    }
    if cfg!(feature = "logodds-depth") {
        let d: Value = serde_json::from_str(LOGODDS_PROTOCOL)?;
        if d["parent_protocol_sha256"] != hash(DENSE_PROTOCOL.as_bytes()) || d["initial_odds"] != 9
        {
            return Err("changed log-odds comparison or parent".into());
        }
        p.name = d["name"].as_str().ok_or("comparison name")?.into();
        let z = 9_f64.ln();
        p.initial_log_odds = Some(z);
        p.initial_depth = crate::model::sigmoid(z);
    }
    #[cfg(feature = "arithmetic-cohort")]
    crate::arithmetic::configure(&mut p)?;
    if p.width != WIDTH
        || hash(SUITE.as_bytes()) != p.suite_sha256
        || hash(learned_retrieval_transfer::CATALOG.as_bytes()) != p.catalog_sha256
    {
        return Err("frozen protocol, width or original suite changed".into());
    }
    Ok(p)
}
pub fn original(p: &Protocol, id: &str) -> Result<(Task, Value)> {
    let s = p.cases.get(id).ok_or("case outside frozen cohort")?;
    #[cfg(feature = "arithmetic-cohort")]
    if let Some(pins) = &s.source_fingerprints {
        let task = crate::arithmetic::original(id)?;
        if task.vocab != s.vocabulary
            || task.splits["train"].len() != s.train
            || task.selection != s.selection
        {
            return Err("arithmetic source dimensions changed".into());
        }
        return Ok((task, pins.clone()));
    }
    let (task, pins) = if let Some(expected) = &s.metadata_sha256 {
        let (task, metadata) = learned_retrieval_transfer::original_task(id)?;
        if hash(&serde_json::to_vec(&metadata)?) != *expected {
            return Err("sequence data fingerprint mismatch".into());
        }
        (task, metadata)
    } else {
        let suite: Value = serde_json::from_str(SUITE)?;
        let row = suite["tasks"]
            .as_array()
            .ok_or("missing suite")?
            .iter()
            .find(|r| r["name"] == id)
            .ok_or("missing original")?;
        if row["case"]["batch"] != s.batch
            || row["case"]["metric"] != s.metric
            || row["case"]["target"] != s.target
            || row["case"]["data_seed"] != p.data_seed
        {
            return Err("original metric or training configuration mismatch".into());
        }
        (
            baseline::data::build(row)?,
            row["data_fingerprints"].clone(),
        )
    };
    if task.vocab != s.vocabulary
        || task.selection != s.selection
        || task.splits["train"].len() != s.train
    {
        return Err("original data shape changed".into());
    }
    Ok((task, pins))
}
