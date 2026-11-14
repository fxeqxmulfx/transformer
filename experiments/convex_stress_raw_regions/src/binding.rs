//! Shape guards precede new hull/certificate allocations. The original parent
//! keeps its complete costs. These are conservative executable forecasts, not
//! Lean theorems about the allocator or floating-point execution time.
use crate::*;

pub(crate) fn estimate(
    source: &kernel::Model,
    parent_buffer: u64,
    inherited: u64,
    records: usize,
) -> Result<Bounds> {
    source.bounds()?;
    kernel::limits::dimensions(source.n, source.vocabulary)?;
    let cells = (source.n as u128).pow(2);
    let points: u128 = source.blocks.iter().map(|b| b.observed.len() as u128).sum();
    let squares: u128 = source
        .blocks
        .iter()
        .map(|b| (b.observed.len() as u128).pow(2))
        .sum();
    let states = source
        .blocks
        .iter()
        .map(|b| b.states())
        .max()
        .ok_or("empty source")? as u128;
    let buffer = u128::from(parent_buffer)
        + 512 * (cells + points + source.blocks.len() as u128 + records as u128);
    // All pairwise hull edge/containment checks fit a quadratic block bound.
    // Query triangle fans use at most the block's total observed vertex count.
    // This includes independent rational-coordinate verification for every row.
    let construction = 512 * squares
        + 256 * cells * (states + 8)
        + 256 * points * (1 + states.ilog2() as u128)
        + 64 * records as u128;
    let total = u128::from(inherited) + construction;
    if buffer > u128::from(BUFFER_LIMIT) || total > u128::from(WORK_LIMIT) {
        return Err(format!(
            "raw-region forecast refused: {buffer} buffer bytes, {total} scalar work"
        )
        .into());
    }
    Ok(Bounds {
        parent_buffer_bytes: parent_buffer,
        buffer_bytes: u64::try_from(buffer)?,
        inherited_scalar_work: inherited,
        construction_scalar_work: u64::try_from(construction)?,
        total_scalar_work: u64::try_from(total)?,
        prediction_scalar_work_per_query: u64::try_from(
            8 * parent::scale::context::PREDICTION_LIMIT as u128
                + 32 * states * (states + source.vocabulary as u128 + 8),
        )?,
        cells: usize::try_from(cells)?,
        source_points: usize::try_from(points)?,
        max_states: usize::try_from(states)?,
    })
}

pub fn forecast(parent: &parent::Model) -> Result<Bounds> {
    estimate(
        parent.source(),
        parent.bounds().buffer_bytes,
        parent.bounds().total_scalar_work,
        parent.records().len(),
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    parent: serde_json::Value,
    binding: Binding,
    geometry: Vec<BlockGeometry>,
    rows: Vec<Option<Row>>,
    statistics: Statistics,
    work: GeometryWork,
    bounds: Bounds,
}

impl Model {
    pub fn new(parent: parent::Model, expected_parent_sha256: &str) -> Result<Self> {
        let bounds = forecast(&parent)?;
        let parent_sha256 = digest(&parent)?;
        if parent_sha256 != expected_parent_sha256 {
            return Err("raw-region parent differs from external binding".into());
        }
        let mut work = GeometryWork::default();
        let geometry = model::build_geometry(parent.source(), &mut work)?;
        let (rows, statistics) = model::build_rows(parent.source(), &geometry, &mut work)?;
        for record in parent.records() {
            if rows[record.cell].is_some() {
                return Err("raw-region map changed an original observed row".into());
            }
        }
        let binding = Binding {
            parent_sha256,
            protocol_sha256: digest(&serde_json::from_str::<serde_json::Value>(PROTOCOL)?)?,
        };
        Ok(Self {
            parent,
            binding,
            geometry,
            rows,
            statistics,
            work,
            bounds,
        })
    }

    pub fn load(value: serde_json::Value, expected: &Binding) -> Result<Self> {
        let saved: Saved = serde_json::from_value(value)?;
        if &saved.binding != expected {
            return Err("saved raw-region binding differs".into());
        }
        let parent_binding = serde_json::from_value(saved.parent["binding"].clone())?;
        let parent = parent::Model::load(saved.parent, &parent_binding)?;
        let actual = Self::new(parent, &expected.parent_sha256)?;
        if actual.binding != *expected
            || actual.geometry != saved.geometry
            || actual.rows != saved.rows
            || actual.statistics != saved.statistics
            || actual.work != saved.work
            || actual.bounds != saved.bounds
        {
            return Err("saved raw-region geometry, certificates, counts or costs differ".into());
        }
        Ok(actual)
    }
}
