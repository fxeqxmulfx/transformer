//! CM50 at2b17dc86: exact two-sided hull conflicts exclude candidate raw labels.
//! Identification is conditional on convex target regions and membership in the
//! declared observed-label pool. Neither premise is inferred from flatness.
//! All features and predictions use the unchanged actual complete complex head.
pub use convex_stress_raw_exclusion as parent;
pub use parent::parent as regions;
pub use parent::{C, Endpoint, Mode, Result, digest, kernel, records};
pub use parent::{Certificate, GeometryWork, Hull, Point, RationalWeight};
use serde::{Deserialize, Serialize};
mod binding;
mod certificates;
pub mod exclusion;
#[cfg(test)]
mod geometry_tests;
mod model;
#[cfg(test)]
mod tests;
pub use binding::forecast;
pub use exclusion::{Decision, Exclusion};

pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const BUFFER_LIMIT: u64 = 56 * 1024 * 1024 * 1024;
pub const WORK_LIMIT: u64 = 256_000_000_000;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Work {
    pub geometry: GeometryWork,
    pub candidate_tests: usize,
    pub class_pair_tests: usize,
    pub edge_pair_tests: usize,
    pub exclusion_certificates: usize,
    pub rows_verified: usize,
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
    pub total_scalar_work: u64,
    pub prediction_scalar_work_per_query: u64,
    pub cells: usize,
    pub max_states: usize,
    pub candidate_query_point_square_sum: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub block: usize,
    pub label: usize,
    pub pool: Vec<usize>,
    pub exclusions: Vec<Exclusion>,
    pub denominator: u64,
    pub weights: Vec<RationalWeight>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Statistics {
    pub retained_parent_cells: usize,
    pub eligible_cells: usize,
    pub identified_cells: usize,
    pub ambiguous_cells: usize,
    pub inconsistent_cells: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Model {
    parent: parent::Model,
    binding: Binding,
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

    /// All original observed rows and their complete-head derivatives are exact.
    pub fn batch_gradient(
        &self,
        positions: &[usize],
    ) -> Result<regions::parent::scale::context::Gradient> {
        self.parent.batch_gradient(positions)
    }
}
