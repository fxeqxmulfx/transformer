//! CM50 implementation guard: charge hull construction, both containment
//! directions, edge pairs, rational verification and retained parent costs.
use crate::*;

pub fn forecast(parent: &parent::Model) -> Result<Bounds> {
    estimate(
        parent,
        parent.bounds().buffer_bytes,
        parent.bounds().total_scalar_work,
    )
}

pub(crate) fn estimate(
    parent: &parent::Model,
    parent_buffer: u64,
    inherited: u64,
) -> Result<Bounds> {
    let source = parent.source();
    source.bounds()?;
    kernel::limits::dimensions(source.n, source.vocabulary)?;
    let cells = source.n.pow(2) as u128;
    let states = source
        .blocks
        .iter()
        .map(|b| b.states())
        .max()
        .ok_or("empty source")? as u128;
    let mut squares = 0u128;
    let mut row_points = 0u128;
    // Sum actual eligible block sizes instead of charging the largest block to
    // every query. These are still conservative upper bounds, not measurements.
    for cell in 0..source.n.pow(2) {
        if model::eligible(parent, cell) {
            let size = source.blocks[source.location(cell).0].observed.len() as u128 + 1;
            squares += size * size;
            row_points += size;
        }
    }
    let buffer = u128::from(parent_buffer) + 4096 * (cells + row_points + states);
    // Across all candidates the class hull sizes sum to at most points+classes.
    // Every pair of hull edges and every vertex/triangle pair is quadratic
    // across all classes. Original separation is also checked. The factor
    // includes checked integer arithmetic, sorting, repeated verification,
    // prediction rows, retained certificates and serialization reserves.
    let construction = 2048 * squares + 512 * cells * states;
    let total = u128::from(inherited) + construction;
    if buffer > u128::from(BUFFER_LIMIT) || total > u128::from(WORK_LIMIT) {
        return Err(format!(
            "exclusion forecast refused: {buffer} buffer bytes, {total} scalar work"
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
            8 * regions::parent::scale::context::PREDICTION_LIMIT as u128
                + 32 * states * (states + source.vocabulary as u128 + 8),
        )?,
        cells: usize::try_from(cells)?,
        max_states: usize::try_from(states)?,
        candidate_query_point_square_sum: u64::try_from(squares)?,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    parent: serde_json::Value,
    binding: Binding,
    rows: Vec<Option<Row>>,
    statistics: Statistics,
    work: Work,
    bounds: Bounds,
}

impl Model {
    pub fn new(parent: parent::Model, expected_parent_sha256: &str) -> Result<Self> {
        let bounds = forecast(&parent)?;
        let parent_sha256 = digest(&parent)?;
        if parent_sha256 != expected_parent_sha256 {
            return Err("exclusion parent differs from external binding".into());
        }
        let mut work = Work::default();
        let (rows, statistics) = model::build_rows(&parent, &mut work)?;
        for record in parent.records() {
            if rows[record.cell].is_some() {
                return Err("exclusion changed an observed row".into());
            }
        }
        let binding = Binding {
            parent_sha256,
            protocol_sha256: digest(&serde_json::from_str::<serde_json::Value>(PROTOCOL)?)?,
        };
        Ok(Self {
            parent,
            binding,
            rows,
            statistics,
            work,
            bounds,
        })
    }
    pub fn load(value: serde_json::Value, expected: &Binding) -> Result<Self> {
        let saved: Saved = serde_json::from_value(value)?;
        if &saved.binding != expected {
            return Err("saved exclusion binding differs".into());
        }
        let binding = serde_json::from_value(saved.parent["binding"].clone())?;
        let parent = parent::Model::load(saved.parent, &binding)?;
        let actual = Self::new(parent, &expected.parent_sha256)?;
        if actual.binding != *expected
            || actual.rows != saved.rows
            || actual.statistics != saved.statistics
            || actual.work != saved.work
            || actual.bounds != saved.bounds
        {
            return Err("saved exclusion pool, witnesses, rows or costs differ".into());
        }
        Ok(actual)
    }
}
