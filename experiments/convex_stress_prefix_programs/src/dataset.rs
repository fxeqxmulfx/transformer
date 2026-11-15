//! Dataset identity belongs only to this adapter, never a model predictor.
use crate::*;
pub use parent::dataset::{Dataset, input};
pub fn cases() -> Result<Vec<String>> {
    Ok(serde_json::from_value(
        serde_json::from_str::<serde_json::Value>(PROTOCOL)?["cases"].clone(),
    )?)
}
pub fn build(case: &str) -> Result<Dataset> {
    if generative::CASES.contains(&case) {
        generative::dataset::build(case)
    } else {
        parent::dataset::build(case)
    }
}
pub fn metadata(data: &Dataset) -> serde_json::Value {
    if generative::CASES.contains(&data.name.as_str()) {
        generative::dataset::metadata(data)
    } else {
        parent::dataset::metadata(data)
    }
}
