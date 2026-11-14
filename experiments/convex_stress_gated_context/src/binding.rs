//! CM46 native binding and preallocation guards. These conservative integer
//! forecasts are executable guards, not Lean floating-point cost theorems.
use crate::*;

pub fn forecast(parent: &parent::Model) -> Result<Bounds> {
    let source = parent.source();
    let n = parent.records().len();
    let ep = parent.endpoint();
    let (j, g, s, a, r) = (
        parent.bank().len() as u128,
        source.blocks.len() as u128,
        parent.bounds().max_states as u128,
        parent.bounds().max_active_classes as u128,
        n as u128,
    );
    let epochs = match ep.mode {
        Mode::Full => FULL_STEPS,
        Mode::Mini => MINI_EPOCHS,
        Mode::Forward => 0,
    };
    let batches = if ep.mode == Mode::Mini {
        n.div_ceil(ep.mixture_batch)
    } else {
        1
    };
    let max_gradient_batches = epochs * batches;
    if max_gradient_batches > 4096 {
        return Err("local gradient batch cap exceeded".into());
    }
    let steps = max_gradient_batches as u128;
    let cells = (source.n as u128).pow(2);
    let buffer = parent.bounds().buffer_bytes as u128
        + 128 * (cells * s + r * j * a + 4 * g * j + steps * g + 3 * s);
    let inherited = parent.bounds().total_scalar_work as u128;
    let construction = 64
        * (r + parent.parent().bounds().context.changed_cells as u128)
        * (3 * s * (1 + parent.bounds().max_states.ilog2() as u128) + j * 17 + a)
        + 64 * (cells + g * j);
    let optimization = 32 * (r * j * a + epochs as u128 * r * (4 * j * a + 6 * a) + steps * g * j);
    let total = inherited + construction + optimization;
    if buffer > 56 * 1024 * 1024 * 1024 || total > kernel::limits::WORK_CAP as u128 {
        return Err("gated context exceeds56GiB memory or32B work forecast".into());
    }
    Ok(Bounds {
        parent_buffer_bytes: parent.bounds().buffer_bytes,
        buffer_bytes: buffer as u64,
        inherited_scalar_work: inherited as u64,
        construction_scalar_work: construction as u64,
        optimization_scalar_work: optimization as u64,
        total_scalar_work: total as u64,
        groups: g as usize,
        bank_size: j as usize,
        max_active_classes: a as usize,
        max_states: s as usize,
        max_gradient_batches,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    parent: serde_json::Value,
    binding: Binding,
    gates: Vec<usize>,
    anchor: Vec<f64>,
    radius: f64,
    selection: Selection,
    rows: Vec<Option<Vec<Weight>>>,
    bounds: Bounds,
}

impl Model {
    pub fn new(parent: parent::Model, expected_parent_sha256: &str) -> Result<Self> {
        let bounds = forecast(&parent)?;
        let parent_sha256 = digest(&parent)?;
        if parent_sha256 != expected_parent_sha256 {
            return Err("CM45 parent differs from external binding".into());
        }
        let gates = (0..parent.source().n * parent.source().n)
            .map(|cell| parent.source().location(cell).0)
            .collect();
        let anchor = parent.selection().weights.clone();
        let selection = selection::select(&parent)?;
        let rows = model::rows(&parent, &selection.weights)?;
        for r in parent.records() {
            if rows[r.cell].is_some() {
                return Err("gated model changed an observed source row".into());
            }
        }
        let binding = Binding {
            parent_sha256,
            protocol_sha256: digest(&serde_json::from_str::<serde_json::Value>(PROTOCOL)?)?,
        };
        Ok(Self {
            parent,
            binding,
            gates,
            anchor,
            radius: RADIUS,
            selection,
            rows,
            bounds,
        })
    }

    pub fn load(value: serde_json::Value, expected: &Binding) -> Result<Self> {
        let saved: Saved = serde_json::from_value(value)?;
        if &saved.binding != expected {
            return Err("saved gated binding differs".into());
        }
        let parent_binding = serde_json::from_value(saved.parent["binding"].clone())?;
        let parent = parent::Model::load(saved.parent, &parent_binding)?;
        let actual = Self::new(parent, &expected.parent_sha256)?;
        if &actual.binding != expected
            || actual.gates != saved.gates
            || actual.anchor != saved.anchor
            || actual.radius != saved.radius
            || actual.selection != saved.selection
            || actual.rows != saved.rows
            || actual.bounds != saved.bounds
        {
            return Err("saved gated architecture,selection,rows or costs changed".into());
        }
        Ok(actual)
    }
}
