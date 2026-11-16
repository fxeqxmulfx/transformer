//! Caller-owned capacity ledger around the compact model. Inputs and local
//! numeric buffers have their own checks. Metadata includes actual Vec/String
//! capacities and conservative node bounds for private BTreeMap storage.
use super::*;
use std::mem::size_of;
pub const PARSER_SCRATCH: u128 = 2 * 1024u128.pow(3);
pub const ENDPOINT_METADATA: u128 = 32 * 1024 * 1024;
const INVENTORY_MAX: u128 = 128 * 1024 * 1024;
fn tree<K, V>(n: usize) -> u128 {
    size_of::<BTreeMap<K, V>>() as u128
        + n as u128 * (12 * (size_of::<K>() + size_of::<V>()) + 256) as u128
}
fn value(v: &Value) -> u128 {
    match v {
        Value::String(s) => s.capacity() as u128,
        Value::Array(a) => {
            a.capacity() as u128 * size_of::<Value>() as u128 + a.iter().map(value).sum::<u128>()
        }
        Value::Object(o) => {
            tree::<String, Value>(o.len())
                + o.iter()
                    .map(|(k, v)| k.capacity() as u128 + value(v))
                    .sum::<u128>()
        }
        _ => 0,
    }
}
fn optional(s: &Option<String>) -> u128 {
    s.as_ref().map_or(0, |s| s.capacity() as u128)
}
pub fn inventory(inv: &Inventory) -> u128 {
    size_of::<Inventory>() as u128
        + inv.producer_commit.capacity() as u128
        + value(&inv.protocol)
        + inv.cases.capacity() as u128 * size_of::<Case>() as u128
        + inv
            .cases
            .iter()
            .map(|c| {
                c.name.capacity() as u128
                    + optional(&c.input_refusal)
                    + optional(&c.source_refusal)
                    + c.audit.as_ref().map_or(0, |a| {
                        a.dictionary_sha256.capacity() as u128
                            + a.program_major_states_sha256.capacity() as u128
                            + optional(&a.admission_refusal)
                    })
            })
            .sum::<u128>()
        + tree::<String, disk::Artifact>(inv.inputs.len())
        + inv
            .inputs
            .iter()
            .map(|(k, v)| (k.capacity() + v.sha256.capacity()) as u128)
            .sum::<u128>()
}
pub fn caller(inv: &Inventory, guard: &resources::Guard) -> Result<u128> {
    let actual = inventory(inv)
        + size_of::<resources::Guard>() as u128
        + (guard.workspace.capacity() + guard.storage.capacity() + guard.run.capacity()) as u128;
    if actual > INVENTORY_MAX {
        return Err("runner inventory exceeds bounded caller capacity".into());
    }
    // Generic metadata reads happen before large model allocation. This also
    // reserves file/JSON/grammar scratch for subsequent bounded operations.
    Ok(actual + PARSER_SCRATCH + ENDPOINT_METADATA)
}
pub fn input_forecast(shape: joint::Shape, records: usize, unknown: usize) -> Result<u64> {
    let bytes = joint::domain::dictionary_bytes(shape, shape.rows(), unknown)?
        + joint::records::packed_bytes(shape, records)?;
    joint::admission::Budget::new(
        RAM_LIMIT_BYTES,
        INVENTORY_MAX + PARSER_SCRATCH + ENDPOINT_METADATA,
    )?
    .remaining(bytes)?;
    Ok(u64::try_from(bytes)?)
}
pub fn dataset(data: &Dataset) -> u128 {
    size_of::<Dataset>() as u128
        + (data.training.rows.capacity() + data.validation.capacity()) as u128
            * size_of::<[u32; 3]>() as u128
}
fn state(s: &model::State) -> u128 {
    s.tensor.sha256.capacity() as u128
        + s.audit.column_l1.capacity() as u128 * size_of::<f64>() as u128
        + s.steps.capacity() as u128 * size_of::<joint::training::Step>() as u128
}
pub fn stage(s: &stage::Stage) -> Result<()> {
    let bytes = size_of::<stage::Stage>() as u128
        + s.case.capacity() as u128
        + tree::<String, stage::Endpoint>(s.endpoints.len())
        + s.endpoints
            .iter()
            .map(|(k, e)| {
                k.capacity() as u128 + optional(&e.refusal) + e.model.as_ref().map_or(0, state)
            })
            .sum::<u128>();
    if bytes > ENDPOINT_METADATA / 4 {
        return Err("training metadata exceeds complete reserved capacity".into());
    }
    Ok(())
}
pub fn score(s: &scoring::Score) -> Result<()> {
    let bytes = size_of::<scoring::Score>() as u128
        + s.case.capacity() as u128
        + s.training_seal.sha256.capacity() as u128
        + tree::<String, scoring::Endpoint>(s.endpoints.len())
        + s.endpoints
            .iter()
            .map(|(k, e)| {
                k.capacity() as u128
                    + optional(&e.refusal)
                    + tree::<String, scoring::Split>(e.splits.len())
                    + e.splits
                        .iter()
                        .map(|(k, s)| (k.capacity() + s.raw.sha256.capacity()) as u128)
                        .sum::<u128>()
            })
            .sum::<u128>();
    if bytes > ENDPOINT_METADATA / 4 {
        return Err("scoring metadata exceeds complete reserved capacity".into());
    }
    Ok(())
}
pub fn remaining_model(
    p: &preparation::Prepared,
    parameters: &joint::Parameters,
    head: &joint::head::Head,
) -> Result<u64> {
    crate::memory::remaining(
        p.budget.model_room(&p.domain, &p.records)?,
        parameters.allocation_bytes() as u128 + head.allocation_bytes() as u128,
    )
}
