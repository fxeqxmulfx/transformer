//! Bind original records, source coefficients, fixed weights and their costs.
//! Native bounds are allocation/work forecasts, not floating-point Lean proofs.
use crate::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    source: kernel::Model,
    records: Vec<records::Record>,
    binding: Binding,
    selection: Selection,
    rows: Vec<Option<Vec<usize>>>,
    bounds: Bounds,
}

impl Model {
    pub fn new(
        source: kernel::Model,
        records: Vec<records::Record>,
        expected_source_sha256: &str,
    ) -> Result<Self> {
        let source_bounds = source.bounds()?;
        let cells = source.n * source.n;
        let max_states = source
            .blocks
            .iter()
            .map(|b| b.states())
            .max()
            .ok_or("empty source")?;
        let extra_buffer_bytes =
            128 * cells as u64 * 10 + 256 * records.len() as u64 + 64 * source.vocabulary as u64;
        let buffer_bytes = source_bounds
            .buffer_bytes
            .checked_add(extra_buffer_bytes)
            .ok_or("context memory overflow")?;
        let construction_scalar_work = 64
            * (cells + BANK.len() * records.len()) as u64
            * (max_states * (1 + max_states.ilog2() as usize) + source.vocabulary) as u64;
        if buffer_bytes > 56 * 1024 * 1024 * 1024
            || construction_scalar_work > kernel::limits::WORK_CAP
        {
            return Err("context interpolation exceeds memory/work forecast".into());
        }
        source.validate()?;
        let source_sha256 = digest(&source)?;
        if source_sha256 != expected_source_sha256 {
            return Err("source head differs from external binding".into());
        }
        let mut observations: Vec<_> = source
            .blocks
            .iter()
            .flat_map(|b| {
                b.observed
                    .iter()
                    .zip(&b.observed_labels)
                    .map(|(&x, &y)| [x, y])
            })
            .collect();
        observations.sort_unstable();
        records::validate(
            &kernel::discovery::Input {
                n: source.n,
                vocabulary: source.vocabulary,
                observations,
                policy: kernel::discovery::Policy::default(),
            },
            &records,
        )?;
        records::encode(&source, &records)?;
        let selection = selection::select(&source, &records);
        let k = selection.selected_neighbors;
        let mut rows = vec![None; cells];
        let mut entries = 0;
        let mut changed = 0;
        if k > 0 {
            for (cell, row) in rows.iter_mut().enumerate() {
                let (block, state) = source.location(cell);
                let b = &source.blocks[block];
                if source.known(cell) && state == b.observed.len() && ambiguous(b) {
                    let states = nearest(&source, block, cell, k);
                    if states.is_empty() {
                        return Err("ambiguous matched cell has no observed source".into());
                    }
                    entries += states.len();
                    changed += 1;
                    *row = Some(states);
                }
            }
        }
        let binding = Binding {
            source_sha256,
            records_sha256: digest(&records)?,
            protocol_sha256: digest(&serde_json::from_str::<serde_json::Value>(PROTOCOL)?)?,
        };
        Ok(Self {
            source,
            records,
            binding,
            selection,
            rows,
            bounds: Bounds {
                source: source_bounds,
                extra_buffer_bytes,
                buffer_bytes,
                construction_scalar_work,
                interpolation_entries: entries,
                changed_cells: changed,
            },
        })
    }

    pub fn load(value: serde_json::Value, expected: &Binding) -> Result<Self> {
        let saved: Saved = serde_json::from_value(value)?;
        if &saved.binding != expected {
            return Err("saved context binding differs from external binding".into());
        }
        let actual = Self::new(saved.source, saved.records, &expected.source_sha256)?;
        if &actual.binding != expected
            || actual.rows != saved.rows
            || actual.bounds != saved.bounds
            || actual.selection != saved.selection
        {
            return Err("saved context weights, selection, records or costs changed".into());
        }
        Ok(actual)
    }
}
