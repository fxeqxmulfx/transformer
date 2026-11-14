//! CM45 at2f0dc9ba: convex mixtures of fixed context maps with a frozen head.
//! A declared source-exclusion diagnostic learns the mixture on original raw
//! records. Flatness supplies bounds, not locality or unseen raw targets.
pub use convex_stress_context_scale as parent;
pub use parent::{C, Result, digest, kernel, records};
use serde::{Deserialize, Serialize};
mod bank;
mod binding;
mod diagnostic;
mod model;
mod selection;
pub use bank::Context;
pub use binding::forecast;
pub use selection::{Candidate, Selection, Step, Work};
#[cfg(test)]
mod tests;

pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const FULL_STEPS: usize = 64;
pub const MINI_EPOCHS: usize = 8;
pub const LINE_SEARCH: [f64; 6] = [0., 0.0625, 0.125, 0.25, 0.5, 1.];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Full,
    Mini,
    Forward,
}

/// Source training metadata is retained separately from new mixture work.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub mode: Mode,
    pub source_reference: String,
    pub source_gradient_training_run: bool,
    pub source_optimizer_updates: Option<usize>,
    pub mixture_batch: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub parent_sha256: String,
    pub endpoint_sha256: String,
    pub protocol_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bounds {
    pub parent_buffer_bytes: u64,
    pub buffer_bytes: u64,
    pub inherited_scalar_work: u64,
    pub construction_scalar_work: u64,
    pub optimization_scalar_work: u64,
    pub total_scalar_work: u64,
    pub bank_size: usize,
    pub max_active_classes: usize,
    pub max_states: usize,
    pub max_gradient_batches: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Weight {
    state: usize,
    mass: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Model {
    parent: parent::Model,
    endpoint: Endpoint,
    binding: Binding,
    bank: Vec<Context>,
    selection: Selection,
    rows: Vec<Option<Vec<Weight>>>,
    bounds: Bounds,
}

impl Model {
    pub fn parent(&self) -> &parent::Model {
        &self.parent
    }
    pub fn source(&self) -> &kernel::Model {
        self.parent.source()
    }
    pub fn records(&self) -> &[records::Record] {
        self.parent.records()
    }
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }
    pub fn binding(&self) -> &Binding {
        &self.binding
    }
    pub fn bank(&self) -> &[Context] {
        &self.bank
    }
    pub fn selection(&self) -> &Selection {
        &self.selection
    }
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    /// CM45 batch_fderiv_eq: all observed actual rows are unchanged.
    pub fn batch_gradient(&self, positions: &[usize]) -> Result<parent::context::Gradient> {
        self.parent.batch_gradient(positions)
    }
}
