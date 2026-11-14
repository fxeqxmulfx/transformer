//! CM44 at a8f0020a: actual complete complex heads for frozen phase decoders.
//! The declared maps are learned upstream. This component selects confidence
//! on original training records, and makes no new raw-label identification.
pub use convex_stress_shared_roles::records;
pub use convex_stress_tagged as tagged;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub use tagged::{C, Result, metrics};
mod binding;
mod gradient;
mod model;
mod selection;
pub use gradient::Gradient;
pub use model::forecast;
pub use selection::{Candidate, Selection};
#[cfg(test)]
mod tests;

pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const BANK: [f64; 6] = [1., 2., 4., 8., 16., 24.];
pub const PREDICTION_LIMIT: usize = 4096;
pub const WORK_CAP: u64 = 32_000_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Full,
    Mini,
    Forward,
}

/// Describes the saved source endpoint, not a new optimizer execution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub mode: Mode,
    pub source_reference: String,
    pub source_gradient_training_run: bool,
    pub source_optimizer_updates: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub source_sha256: String,
    pub records_sha256: String,
    pub endpoint_sha256: String,
    pub protocol_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bounds {
    pub source_buffer_bytes: u64,
    pub buffer_bytes: u64,
    pub construction_scalar_work: u64,
    pub complex_features: usize,
    pub real_coefficients: usize,
    pub cached_logits: usize,
    pub new_optimizer_updates: usize,
    pub new_gradient_batches: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Model {
    source: tagged::Model,
    records: Vec<records::Record>,
    endpoint: Endpoint,
    binding: Binding,
    selection: Selection,
    bounds: Bounds,
    code: Vec<C>,
    coefficients: Vec<f64>,
    table: Vec<f64>,
    radius: f64,
}

pub fn digest<T: Serialize>(value: &T) -> Result<String> {
    Ok(Sha256::digest(serde_json::to_vec(value)?)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
