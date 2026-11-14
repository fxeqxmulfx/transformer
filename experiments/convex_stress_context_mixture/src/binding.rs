//! External parent/endpoint binding and conservative allocation/work forecasts.
//! The integer forecasts are native guards, not Lean floating-point proofs.
use crate::*;
use std::collections::BTreeSet;

pub fn forecast(parent: &parent::Model, endpoint: &Endpoint) -> Result<Bounds> {
    let source = parent.source();
    let n = parent.records().len();
    if n == 0
        || !(1..=65536).contains(&endpoint.mixture_batch)
        || endpoint.source_reference.is_empty()
        || endpoint.source_reference.len() > 1024
        || (!endpoint.source_gradient_training_run
            && endpoint.source_optimizer_updates.is_some_and(|n| n != 0))
        || (endpoint.mode == Mode::Forward
            && (endpoint.source_gradient_training_run
                || endpoint.source_optimizer_updates != Some(0)))
    {
        return Err("invalid inherited endpoint or mixture minibatch".into());
    }
    let max_states = source
        .blocks
        .iter()
        .map(|b| b.states())
        .max()
        .ok_or("empty source")?;
    let max_active_classes = source
        .blocks
        .iter()
        .map(|b| {
            b.observed_labels
                .iter()
                .copied()
                .chain(std::iter::once(b.default_label))
                .collect::<BTreeSet<_>>()
                .len()
        })
        .max()
        .ok_or("empty source")?;
    let bank_size = bank::build(parent).len();
    let epochs = match endpoint.mode {
        Mode::Full => FULL_STEPS,
        Mode::Mini => MINI_EPOCHS,
        Mode::Forward => 0,
    };
    let batches = if endpoint.mode == Mode::Mini {
        n.div_ceil(endpoint.mixture_batch)
    } else {
        1
    };
    let max_gradient_batches = epochs * batches;
    if max_gradient_batches > 4096 {
        return Err("mixture gradient batch cap exceeded".into());
    }
    let (s, a, j, r) = (
        max_states as u128,
        max_active_classes as u128,
        bank_size as u128,
        n as u128,
    );
    let cells = (source.n as u128).pow(2);
    let extra = 128 * (cells * s + r * j * a + max_gradient_batches as u128 * j + 3 * s);
    let buffer = parent.bounds().buffer_bytes as u128 + extra;
    let inherited = parent.bounds().context.source.audit_work as u128
        + parent.bounds().context.construction_scalar_work as u128
        + parent.bounds().calibration_scalar_work as u128;
    let construction = 64
        * (r + parent.bounds().context.changed_cells as u128)
        * (3 * s * (1 + max_states.ilog2() as u128) + j * 17 + a);
    let visits = epochs as u128 * r;
    let optimization = 32 * (r * j * a + visits * (4 * j * a + 6 * a));
    let total = inherited + construction + optimization;
    if buffer > 56 * 1024 * 1024 * 1024 || total > kernel::limits::WORK_CAP as u128 {
        return Err("context mixture exceeds56GiB memory or32B work forecast".into());
    }
    Ok(Bounds {
        parent_buffer_bytes: parent.bounds().buffer_bytes,
        buffer_bytes: buffer as u64,
        inherited_scalar_work: inherited as u64,
        construction_scalar_work: construction as u64,
        optimization_scalar_work: optimization as u64,
        total_scalar_work: total as u64,
        bank_size,
        max_active_classes,
        max_states,
        max_gradient_batches,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    parent: serde_json::Value,
    endpoint: Endpoint,
    binding: Binding,
    bank: Vec<Context>,
    selection: Selection,
    rows: Vec<Option<Vec<Weight>>>,
    bounds: Bounds,
}

impl Model {
    pub fn new(
        parent: parent::Model,
        endpoint: Endpoint,
        expected_parent_sha256: &str,
    ) -> Result<Self> {
        let bounds = forecast(&parent, &endpoint)?;
        let parent_sha256 = digest(&parent)?;
        if parent_sha256 != expected_parent_sha256 {
            return Err("parent differs from external binding".into());
        }
        for r in parent.records() {
            if parent.context().neighbors(r.cell)?.is_some() {
                return Err("observed parent row is not preserved".into());
            }
        }
        let bank = bank::build(&parent);
        let selection = selection::select(&parent, &bank, &endpoint)?;
        let rows = model::rows(&parent, &bank, &selection.weights)?;
        let binding = Binding {
            parent_sha256,
            endpoint_sha256: digest(&endpoint)?,
            protocol_sha256: digest(&serde_json::from_str::<serde_json::Value>(PROTOCOL)?)?,
        };
        Ok(Self {
            parent,
            endpoint,
            binding,
            bank,
            selection,
            rows,
            bounds,
        })
    }

    pub fn load(value: serde_json::Value, expected: &Binding) -> Result<Self> {
        let saved: Saved = serde_json::from_value(value)?;
        if &saved.binding != expected {
            return Err("saved mixture binding differs".into());
        }
        let parent_binding = serde_json::from_value(saved.parent["binding"].clone())?;
        let parent = parent::Model::load(saved.parent, &parent_binding)?;
        let actual = Self::new(parent, saved.endpoint, &expected.parent_sha256)?;
        if &actual.binding != expected
            || actual.bank != saved.bank
            || actual.selection != saved.selection
            || actual.rows != saved.rows
            || actual.bounds != saved.bounds
        {
            return Err("saved mixture selection, rows, bank or costs changed".into());
        }
        Ok(actual)
    }
}
