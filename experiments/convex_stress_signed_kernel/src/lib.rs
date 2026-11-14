//! CM51 at803958cc: bounded signed maps of the actual complex source features.
//! CM45 controls a training-only convex mixture with a frozen canonical head.
//! Kernel structure and query reference residuals are not consequences of flatness.
pub use convex_stress_crossing_exclusion as parent;
pub use parent::regions;
pub use parent::{C, Endpoint, Mode, Result, digest, kernel, records};
pub use regions::parent as gated;
use serde::{Deserialize, Serialize};
mod baseline;
mod binding;
mod diagnostic;
#[cfg(test)]
mod diagnostic_tests;
pub mod kernels;
mod model;
mod selection;
pub mod solver;
#[cfg(test)]
mod solver_tests;
#[cfg(test)]
mod tests;
pub use binding::forecast;
pub use kernels::{FitReport, Spec};
pub use selection::{Candidate, OptimizationWork, Selection, Step};

pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const BUFFER_LIMIT: u64 = 56 * 1024 * 1024 * 1024;
pub const WORK_LIMIT: u64 = 256_000_000_000;
pub const FULL_STEPS: usize = 64;
pub const MINI_EPOCHS: usize = 8;
pub const LINE_SEARCH: [f64; 6] = [0., 0.0625, 0.125, 0.25, 0.5, 1.];

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Work {
    pub kernel_matrices: usize,
    pub system_solves: usize,
    pub inverse_residual_products: u64,
    pub excluded_state_rows: usize,
    pub prediction_rows: usize,
    pub normalizations: usize,
    pub gradient_batches: usize,
    pub optimizer_updates: usize,
}

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
    pub prediction_scalar_work_per_query: u64,
    pub cells: usize,
    pub eligible_cells: usize,
    pub active_blocks: usize,
    pub bank_size: usize,
    pub max_states: usize,
    pub max_gradient_batches: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub block: usize,
    pub parent_mass: f64,
    pub weights: Vec<f64>,
    pub normalization_scale: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Statistics {
    pub retained_parent_cells: usize,
    pub eligible_cells: usize,
    pub changed_feature_rows: usize,
    pub negative_weight_rows: usize,
    pub refused_kernel_fits: usize,
    pub retained_baseline_rows: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Model {
    parent: parent::Model,
    binding: Binding,
    bank: Vec<Spec>,
    fits: Vec<FitReport>,
    selection: Selection,
    diagnostic_sha256: String,
    rows: Vec<Option<Row>>,
    statistics: Statistics,
    work: Work,
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
    pub fn bank(&self) -> &[Spec] {
        &self.bank
    }
    pub fn fits(&self) -> &[FitReport] {
        &self.fits
    }
    pub fn selection(&self) -> &Selection {
        &self.selection
    }
    pub fn rows(&self) -> &[Option<Row>] {
        &self.rows
    }
    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }
    pub fn work(&self) -> &Work {
        &self.work
    }
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }
    /// CM51 retains Dirac source rows, hence every original batch derivative.
    pub fn batch_gradient(&self, positions: &[usize]) -> Result<gated::scale::context::Gradient> {
        self.parent.batch_gradient(positions)
    }
}
