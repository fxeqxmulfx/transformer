//! Preserve the complete reviewed context representation and its external
//! digest. Reconstruct selection and costs when loading, rejecting mutations.
use crate::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    context: serde_json::Value,
    binding: Binding,
    selection: Selection,
    bounds: Bounds,
}

impl Model {
    pub fn new(context: context::Model, expected_context_sha256: &str) -> Result<Self> {
        let context_sha256 = digest(&context)?;
        if context_sha256 != expected_context_sha256 {
            return Err("context model differs from external binding".into());
        }
        let source = context.source();
        let max_states = source
            .blocks
            .iter()
            .map(|b| b.states())
            .max()
            .ok_or("empty context")?;
        let extra_buffer_bytes = 512 * max_states as u64
            + 256 * context.records().len() as u64
            + 128 * source.vocabulary as u64;
        let buffer_bytes = context
            .bounds()
            .buffer_bytes
            .checked_add(extra_buffer_bytes)
            .ok_or("context scale memory overflow")?;
        let calibration_scalar_work = 64
            * context.records().len() as u64
            * (max_states * (1 + max_states.ilog2() as usize) + BANK.len() * source.vocabulary)
                as u64;
        if buffer_bytes > 56 * 1024 * 1024 * 1024
            || calibration_scalar_work > kernel::limits::WORK_CAP
        {
            return Err("context scale exceeds memory/work forecast".into());
        }
        for r in context.records() {
            if context.neighbors(r.cell)?.is_some() {
                return Err("original observed context row must be unchanged".into());
            }
        }
        let selection = selection::select(&context)?;
        let bounds = Bounds {
            context: context.bounds().clone(),
            extra_buffer_bytes,
            buffer_bytes,
            calibration_scalar_work,
            scaled_cells: if selection.selected_scale == 1. {
                0
            } else {
                context.bounds().changed_cells
            },
        };
        let binding = Binding {
            context_sha256,
            protocol_sha256: digest(&serde_json::from_str::<serde_json::Value>(PROTOCOL)?)?,
        };
        Ok(Self {
            context,
            binding,
            selection,
            bounds,
        })
    }

    pub fn load(value: serde_json::Value, expected: &Binding) -> Result<Self> {
        let saved: Saved = serde_json::from_value(value)?;
        if &saved.binding != expected {
            return Err("saved scale binding differs".into());
        }
        let context_binding = serde_json::from_value(saved.context["binding"].clone())?;
        let context = context::Model::load(saved.context, &context_binding)?;
        let actual = Self::new(context, &expected.context_sha256)?;
        if &actual.binding != expected
            || actual.selection != saved.selection
            || actual.bounds != saved.bounds
        {
            return Err("saved scale selection, context or costs changed".into());
        }
        Ok(actual)
    }
}
