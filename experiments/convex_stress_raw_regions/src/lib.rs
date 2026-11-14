//! CM48 at 71d96c93: observed input hulls supply fixed complex context rows.
//! Convex raw-class regions are an extra realizability assumption, not a
//! consequence of spectral flatness. All prediction scores use the actual
//! unchanged complete head. Overlap and missing coverage keep the CM46 parent.
pub use convex_stress_gated_context as parent;
pub use parent::{C, Endpoint, Mode, Result, digest, kernel, records};
use serde::{Deserialize, Serialize};
mod binding;
pub mod geometry;
#[cfg(test)]
mod geometry_tests;
mod model;
#[cfg(test)]
mod tests;
pub use binding::forecast;
pub use geometry::{Certificate, Hull, Point, RationalWeight};

pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const BUFFER_LIMIT: u64 = 56 * 1024 * 1024 * 1024;
pub const WORK_LIMIT: u64 = 128_000_000_000;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryWork {
    pub sorted_points: usize,
    pub orientations: usize,
    pub membership_tests: usize,
    pub class_pairs: usize,
    pub certificates_verified: usize,
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
    pub source_points: usize,
    pub max_states: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    Empty,
    SingleLabel,
    Separated,
    OverlapRefused,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockGeometry {
    pub disposition: Disposition,
    pub hulls: Vec<Hull>,
    pub overlap: Option<[usize; 2]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub block: usize,
    pub certificate: Certificate,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Statistics {
    pub observed_cells: usize,
    pub unsupported_cells: usize,
    pub inactive_cells: usize,
    pub overlap_refused_cells: usize,
    pub covered_cells: usize,
    pub uncovered_cells: usize,
    pub separated_blocks: usize,
    pub overlap_refused_blocks: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Model {
    parent: parent::Model,
    binding: Binding,
    geometry: Vec<BlockGeometry>,
    rows: Vec<Option<Row>>,
    statistics: Statistics,
    work: GeometryWork,
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
    pub fn geometry(&self) -> &[BlockGeometry] {
        &self.geometry
    }
    pub fn rows(&self) -> &[Option<Row>] {
        &self.rows
    }
    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }
    pub fn work(&self) -> &GeometryWork {
        &self.work
    }
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    /// Exact observed rows retain all original derivatives and record weights.
    pub fn batch_gradient(&self, positions: &[usize]) -> Result<parent::scale::context::Gradient> {
        self.parent.batch_gradient(positions)
    }
}
