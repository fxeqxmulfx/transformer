//! Original record identities and multiplicities are separate from distinct
//! algebraic equations. CM39 batch identities at 2ca6d2cf and CM40 at ec5f839d
//! require the actual record denominator, including repetitions and tails.
use crate::*;
use kernel::{discovery::Input, model::Encoded};
use std::collections::{BTreeMap, BTreeSet};
pub const MAX_RECORDS: usize = 257 * 257;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Record {
    pub id: usize,
    pub cell: usize,
    pub label: usize,
}
pub fn validate(input: &Input, records: &[Record]) -> Result<()> {
    limits::dimensions(input.n, input.vocabulary)?;
    let mut equations = BTreeMap::new();
    for &[cell, label] in &input.observations {
        if cell >= input.n * input.n
            || label >= input.vocabulary
            || equations.insert(cell, label).is_some()
        {
            return Err("invalid distinct equation dictionary for original records".into());
        }
    }
    let (mut ids, mut covered) = (BTreeSet::new(), BTreeSet::new());
    if records.is_empty() || records.len() > MAX_RECORDS {
        return Err("original record count exceeds its bounded domain".into());
    }
    for r in records {
        if !ids.insert(r.id) || equations.get(&r.cell) != Some(&r.label) {
            return Err(
                "duplicate record identity or record absent from original equations".into(),
            );
        }
        covered.insert(r.cell);
    }
    if covered.len() != equations.len() {
        return Err("an original observed equation has no training record".into());
    }
    Ok(())
}
pub fn encode(model: &Model, records: &[Record]) -> Result<Vec<Encoded>> {
    if records.is_empty() || records.len() > MAX_RECORDS {
        return Err("invalid bounded original record count".into());
    }
    let mut ids = BTreeSet::new();
    records
        .iter()
        .map(|r| {
            if r.cell >= model.n * model.n || r.label >= model.vocabulary || !ids.insert(r.id) {
                return Err("invalid original record identity, cell or label".into());
            }
            let (block, state) = model.location(r.cell);
            if model.latent(r.cell).is_none()
                || model.blocks[block].observed_labels.get(state) != Some(&r.label)
            {
                return Err("original record is absent from the fixed observed dictionary".into());
            }
            Ok(Encoded {
                block,
                state,
                label: r.label,
            })
        })
        .collect()
}
