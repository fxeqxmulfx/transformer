//! CM47 at85aa4a2e, ported from70fb09bb with an inspectable16T work budget.
//! CM43/46 bound the bank; the reference may still have incorrect predictions.
pub use convex_stress_gated_context as parent;
pub use parent::scale;
pub use parent::{C, Context, Endpoint, Mode, Result, digest, kernel, records};
use serde::{Deserialize, Serialize};
mod bank;
mod binding;
mod diagnostic;
mod gates;
mod guard;
mod model;
mod selection;
pub use binding::{estimate, forecast};
pub use gates::Profile;
pub use guard::{Certificate, argmax, margins_hold};
pub use selection::{Candidate, Selection, Step, Work};
#[cfg(test)]
mod tests;

pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const RADIUS: f64 = 0.5;
pub const RETENTION: f64 = 0.5;
pub const REFERENCE_SCALE: f64 = 0.5;
pub const GATE_COUNT: usize = 5;
pub const FULL_STEPS: usize = 64;
pub const MINI_EPOCHS: usize = 8;
pub const BACKTRACKS: usize = 16;
pub const LINE_SEARCH: [f64; 6] = [0., 0.0625, 0.125, 0.25, 0.5, 1.];
pub const WORK_CAP: u64 = 16_000_000_000_000;

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
    pub guarded_cells: usize,
    pub guard_classes: usize,
    pub max_active_classes: usize,
    pub max_states: usize,
    pub max_gradient_batches: usize,
    pub max_guard_query_visits: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Weight {
    state: usize,
    mass: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureRow {
    reference_mass: f64,
    residual: Vec<C>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Model {
    parent: parent::Model,
    binding: Binding,
    profiles: Vec<Profile>,
    gates: Vec<usize>,
    radius: f64,
    retention: f64,
    selection: Selection,
    rows: Vec<Option<FeatureRow>>,
    certificate: Certificate,
    bounds: Bounds,
}

pub(crate) fn anchor(j: usize) -> Vec<f64> {
    let mut result = vec![0.; j];
    result[0] = 1.;
    result
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
    /// These are the91 original maps; reference and scaled reference precede them.
    pub fn bank(&self) -> &[Context] {
        self.parent.bank()
    }
    pub fn gates(&self) -> &[usize] {
        &self.gates
    }
    pub fn profiles(&self) -> &[Profile] {
        &self.profiles
    }
    pub fn radius(&self) -> f64 {
        self.radius
    }
    pub fn retention(&self) -> f64 {
        self.retention
    }
    pub fn selection(&self) -> &Selection {
        &self.selection
    }
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }
    pub fn certificate(&self) -> &Certificate {
        &self.certificate
    }

    /// CM46 batch_fderiv_eq: original observed complete-head derivatives stay exact.
    pub fn batch_gradient(&self, positions: &[usize]) -> Result<scale::context::Gradient> {
        self.parent.batch_gradient(positions)
    }
}
