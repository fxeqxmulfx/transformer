//! Prospective disk upper bound from complete original shapes, before fitting.
//! Assume all176 programs survive and every state/class row is unknown. This
//! bounds saved coordinates, not simultaneous in-memory model admission.
use crate::*;
use probes::{dataset::Dataset, inventory::Inventory};
use std::path::Path;
pub(crate) fn case(data: &Dataset) -> Result<u64> {
    let programs = model::features::PROGRAMS as u64;
    let states = model::features::STATES as u64;
    let model =
        88 + 4 * programs * states + 8 * (programs + programs * states * data.vocabulary as u64);
    if model > resources::FILE_MAX {
        return Err("prospective complete checkpoint exceeds file budget".into());
    }
    let mut raw = 0u64;
    for (name, split) in &data.splits {
        let length = trace::length(data.vocabulary, trace::maximum_calls(split, name)?)?;
        if length > resources::FILE_MAX {
            return Err("prospective complete raw trace exceeds file budget".into());
        }
        raw = raw.checked_add(length).ok_or("disk forecast overflow")?;
    }
    Ok(MODES.len() as u64 * (model + raw))
}
pub(crate) fn all(root: &Path, inventory: &Inventory) -> Result<u64> {
    // Input inventory/header/seal plus every full stage/score/replay JSON cap.
    let mut total = (3 + 3 * inventory.cases.len() as u64) * io::JSON_MAX;
    for entry in &inventory.cases {
        if let Some(expected) = &entry.artifact {
            let path = root.join(format!("{}.json", inputs::id(entry)?));
            let data: Dataset = io::read(&path)?;
            probe(data.validate(&entry.source))?;
            total = total
                .checked_add(expected.bytes as u64 + case(&data)?)
                .ok_or("cohort disk forecast overflow")?;
        }
    }
    if total > resources::RUN_MAX {
        return Err("complete original/repeat output forecast exceeds fixed reservation".into());
    }
    Ok(total)
}
