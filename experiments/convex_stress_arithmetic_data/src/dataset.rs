//! Preserve original row order, masks and disjoint half-table split. A model
//! receives Training alone, which carries neither an oracle nor heldout data.
use crate::{Result, oracle, reference};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Training {
    pub vocabulary: usize,
    /// Original order; one supervised final token per pair. No row deduplication.
    pub rows: Vec<[u32; 3]>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dataset {
    pub training: Training,
    pub validation: Vec<[u32; 3]>,
}
impl Dataset {
    pub fn validate(&self) -> Result<()> {
        let p = self.training.vocabulary;
        if ![17, 97, 193].contains(&p)
            || self.training.rows.is_empty()
            || self.validation.is_empty()
        {
            return Err("invalid source vocabulary or empty split".into());
        }
        let mut pairs = BTreeSet::new();
        for row in self.training.rows.iter().chain(&self.validation) {
            if row.iter().any(|&x| x as usize >= p) || !pairs.insert([row[0], row[1]]) {
                return Err("label range or disjoint-pair constraint failed".into());
            }
        }
        Ok(())
    }
}
pub fn build(row: &Value) -> Result<Option<Dataset>> {
    let case = &row["case"];
    let old = &row["history"];
    if case["kind"] != "arithmetic"
        || case["data_seed"] != 1
        || old["seed"] != 1
        || case["name"] != old["case"]
        || case["data"] != old["data"]
        || case["metric"] != "sequence_accuracy"
        || case["target"] != 0.99
    {
        return Err("original case, seed or criterion changed".into());
    }
    let Some(mut rows) = oracle::table(&case["data"])? else {
        return Ok(None);
    };
    reference::SplitMix64(1).shuffle(&mut rows);
    let mid = rows.len() / 2;
    let d = Dataset {
        training: Training {
            vocabulary: case["data"]["prime"].as_u64().ok_or("prime")? as usize,
            rows: rows[..mid].to_vec(),
        },
        validation: rows[mid..].to_vec(),
    };
    d.validate()?;
    for (name, split) in [("train", &d.training.rows), ("validation", &d.validation)] {
        if old["data_fingerprints"][name] != reference::fingerprint(split)
            || old[name]["examples"].as_u64() != Some(split.len() as u64)
            || old[name]["targets"].as_u64() != Some(split.len() as u64)
        {
            return Err(format!(
                "original data fingerprint or denominator differs: {}/{name}",
                case["name"]
            )
            .into());
        }
    }
    Ok(Some(d))
}
