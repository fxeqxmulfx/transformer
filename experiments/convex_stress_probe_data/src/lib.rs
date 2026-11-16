//! Data-only native port of the complete six GPT mechanism probes at DiffGPT
//! 5a37c515. Answer formulas and numeric teachers belong only to this producer.
//! No learned model, parameter update or accuracy measurement occurs here.
pub use baseline::{Result, data, hash};
pub mod boundary;
pub mod dataset;
#[path = "../../convex_stress_generative_data/src/scoring.rs"]
mod generation;
pub mod inventory;
mod native;
pub mod scoring;
pub mod storage;
#[cfg(test)]
mod tests;
pub mod wire;
pub const CASES: &str = include_str!("../references/cases.json");
pub const ORIGINAL35: &str = include_str!("../../convex_stress/suite.json");

pub fn cases() -> Result<Vec<serde_json::Value>> {
    if hash(CASES.as_bytes()) != "cded43ad6d2cdf734f35dde88f28709bf98fc1a79ba20e42b03bb16e8502f05e"
        || hash(ORIGINAL35.as_bytes())
            != "6d2735e987026959c3cde995fbe866281e4f0b2308cc71be4b9d2edd13840436"
    {
        return Err("required original source extract changed".into());
    }
    let rows: Vec<serde_json::Value> = serde_json::from_str(CASES)?;
    let expected = [
        "routing-mixture",
        "branch-amplitudes",
        "conditional-amplitudes",
        "linear-transport",
        "order-discrimination",
        "key-value-binding",
    ];
    if rows.len() != expected.len() {
        return Err("complete six-probe inventory required".into());
    }
    for (row, name) in rows.iter().zip(expected) {
        if row["case"]["name"] != format!("gpt-probe-{name}")
            || row["case"]["data"]["task"] != name
            || row["case"] != row["history"]["protocol"]
            || row["case"]["data_seed"] != 1
        {
            return Err("original probe identity, order or protocol differs".into());
        }
    }
    Ok(rows)
}
