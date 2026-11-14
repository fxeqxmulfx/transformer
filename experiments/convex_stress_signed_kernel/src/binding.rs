//! Forecast all kernel matrices, inverse residuals, source exclusion, repeated
//! record optimization and complete prediction products before construction.
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
    let active = model::active_blocks(parent);
    let bank = kernels::bank().len() as u128;
    let records = parent.records().len() as u128;
    let states = source
        .blocks
        .iter()
        .map(|b| b.states())
        .max()
        .ok_or("empty signed source")? as u128;
    let classes = source
        .blocks
        .iter()
        .map(|b| {
            let mut labels = b.observed_labels.clone();
            labels.push(b.default_label);
            labels.sort_unstable();
            labels.dedup();
            labels.len()
        })
        .max()
        .ok_or("empty signed classes")? as u128;
    let (mut squares, mut cubes, mut query_squares, mut eligible) = (0u128, 0u128, 0u128, 0usize);
    for (b, &on) in source.blocks.iter().zip(&active) {
        let n = b.observed.len();
        if on && n <= solver::MAX_OBSERVATIONS {
            squares += (n as u128).pow(2);
            cubes += (n as u128).pow(3);
        }
    }
    for cell in 0..source.n.pow(2) {
        if model::eligible(parent, cell) {
            eligible += 1;
            let n = source.blocks[source.location(cell).0].observed.len();
            if n <= solver::MAX_OBSERVATIONS {
                query_squares += (n as u128 + 1).pow(2);
            }
        }
    }
    let endpoint = parent.endpoint();
    if records == 0 || endpoint.mixture_batch == 0 {
        return Err("invalid diagnostic record count or batch".into());
    }
    let epochs = match endpoint.mode {
        Mode::Full => FULL_STEPS,
        Mode::Mini => MINI_EPOCHS,
        Mode::Forward => 0,
    };
    let batches = if endpoint.mode == Mode::Mini {
        parent.records().len().div_ceil(endpoint.mixture_batch)
    } else {
        1
    };
    let max_gradient_batches = epochs * batches;
    if max_gradient_batches > 4096 {
        return Err("signed diagnostic gradient-batch cap".into());
    }
    let cells = source.n.pow(2) as u128;
    let construction = 64 * bank * cubes
        + 16 * bank * query_squares
        + 64 * records * (bank + baseline::gated(parent).bank().len() as u128) * states
        + 512 * cells * states;
    let optimization = 128 * records * bank * classes * (epochs as u128 + 1);
    let total = u128::from(inherited) + construction + optimization;
    // Includes live inverse/temporary systems, cache templates and repeated
    // record cache, all rows, serialization and the complete retained parent.
    let buffer = u128::from(parent_buffer)
        + 128 * bank * squares
        + 128 * records * bank * classes
        + 512 * cells
        + 128 * eligible as u128 * states;
    if buffer > u128::from(BUFFER_LIMIT) || total > u128::from(WORK_LIMIT) {
        return Err(format!(
            "signed kernel forecast refused: {buffer} buffer bytes, {total} scalar work"
        )
        .into());
    }
    Ok(Bounds {
        parent_buffer_bytes: parent_buffer,
        buffer_bytes: u64::try_from(buffer)?,
        inherited_scalar_work: inherited,
        construction_scalar_work: u64::try_from(construction)?,
        optimization_scalar_work: u64::try_from(optimization)?,
        total_scalar_work: u64::try_from(total)?,
        prediction_scalar_work_per_query: u64::try_from(
            u128::from(parent.bounds().prediction_scalar_work_per_query)
                + 64 * states * (states + source.vocabulary as u128 + 8),
        )?,
        cells: usize::try_from(cells)?,
        eligible_cells: eligible,
        active_blocks: active.iter().filter(|&&a| a).count(),
        bank_size: usize::try_from(bank)?,
        max_states: usize::try_from(states)?,
        max_gradient_batches,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    parent: serde_json::Value,
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
    pub fn new(parent: parent::Model, expected_parent_sha256: &str) -> Result<Self> {
        let bounds = forecast(&parent)?;
        let parent_sha256 = digest(&parent)?;
        if parent_sha256 != expected_parent_sha256 {
            return Err("signed parent differs from external binding".into());
        }
        let bank = kernels::bank();
        let active = model::active_blocks(&parent);
        let mut work = Work::default();
        let (fitted, fits) = kernels::fit_bank(&parent, &bank, &active, &mut work)?;
        let cache = diagnostic::Cache::new(&parent, &fitted, &mut work)?;
        let diagnostic_sha256 = digest(&cache)?;
        let selection = selection::select(&parent, &cache)?;
        work.gradient_batches = selection.work.gradient_batches;
        work.optimizer_updates = selection.work.optimizer_updates;
        let (rows, mut statistics) = model::rows(&parent, &fitted, &selection, &mut work)?;
        statistics.refused_kernel_fits = fits.iter().filter(|r| r.refusal.is_some()).count();
        if parent.records().iter().any(|r| rows[r.cell].is_some()) {
            return Err("signed map changed an observed row".into());
        }
        let binding = Binding {
            parent_sha256,
            protocol_sha256: digest(&serde_json::from_str::<serde_json::Value>(PROTOCOL)?)?,
        };
        Ok(Self {
            parent,
            binding,
            bank,
            fits,
            selection,
            diagnostic_sha256,
            rows,
            statistics,
            work,
            bounds,
        })
    }

    pub fn load(value: serde_json::Value, expected: &Binding) -> Result<Self> {
        let saved: Saved = serde_json::from_value(value)?;
        if &saved.binding != expected {
            return Err("saved signed binding differs".into());
        }
        let binding = serde_json::from_value(saved.parent["binding"].clone())?;
        let parent = parent::Model::load(saved.parent, &binding)?;
        let actual = Self::new(parent, &expected.parent_sha256)?;
        if actual.binding != *expected
            || actual.bank != saved.bank
            || actual.fits != saved.fits
            || actual.selection != saved.selection
            || actual.diagnostic_sha256 != saved.diagnostic_sha256
            || actual.rows != saved.rows
            || actual.statistics != saved.statistics
            || actual.work != saved.work
            || actual.bounds != saved.bounds
        {
            return Err("saved signed kernels, selection, rows or costs differ".into());
        }
        Ok(actual)
    }
}
