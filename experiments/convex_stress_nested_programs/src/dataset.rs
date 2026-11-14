//! Six original datasets: unchanged four-case controls and two new cases.
//! Names remain confined to adapters; the model receives only observed rows.
use crate::*;
pub use sequences::dataset::Dataset;
pub fn cases() -> Result<Vec<String>> {
    Ok(serde_json::from_value(
        serde_json::from_str::<serde_json::Value>(PROTOCOL)?["cases"].clone(),
    )?)
}
pub fn build(case: &str) -> Result<Dataset> {
    if typed::CASES.contains(&case) {
        typed::dataset::build(case)
    } else {
        sequences::dataset::build(case)
    }
}
