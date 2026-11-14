//! Generic CM46 at c60816fa instantiated with five fixed class-count gates.
//! Ported from a8deaa63, sharing local coefficients across source blocks.
//! The source spectrum bounds features; it does not select an unseen raw label.
pub use convex_stress_context_mixture as parent;
pub use parent::parent as scale;
pub use parent::{C, Context, Endpoint, Mode, Result, digest, kernel, records};
use serde::{Deserialize, Serialize};
mod bank;
mod binding;
mod diagnostic;
mod gates;
mod model;
mod selection;
pub use binding::forecast;
pub use gates::Profile;
pub use selection::{Candidate, Selection, Step, Work};
#[cfg(test)]
mod tests;

pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const RADIUS: f64 = 0.5;
pub const GATE_COUNT: usize = 5;
pub const FULL_STEPS: usize = 64;
pub const MINI_EPOCHS: usize = 8;
pub const LINE_SEARCH: [f64; 6] = [0., 0.0625, 0.125, 0.25, 0.5, 1.];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub parent_sha256: String,
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
    pub groups: usize,
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
    binding: Binding,
    profiles: Vec<Profile>,
    gates: Vec<usize>,
    anchor: Vec<f64>,
    radius: f64,
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
        self.parent.endpoint()
    }
    pub fn binding(&self) -> &Binding {
        &self.binding
    }
    pub fn bank(&self) -> &[Context] {
        self.parent.bank()
    }
    pub fn gates(&self) -> &[usize] {
        &self.gates
    }
    pub fn profiles(&self) -> &[Profile] {
        &self.profiles
    }
    pub fn anchor(&self) -> &[f64] {
        &self.anchor
    }
    pub fn radius(&self) -> f64 {
        self.radius
    }
    pub fn selection(&self) -> &Selection {
        &self.selection
    }
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    /// CM46 batch_fderiv_eq: preserve every original observed complete-head derivative.
    pub fn batch_gradient(&self, positions: &[usize]) -> Result<scale::context::Gradient> {
        self.parent.batch_gradient(positions)
    }
}
