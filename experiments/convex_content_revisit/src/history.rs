//! Exact historical-state replay from the original model and accepted atom edits.
//! Readout and capacity changes do not change a component's state execution.
use crate::{Result, cache::Cache};
use bilinear::train::Record;
use rotary::{atom::Atom, model::Model};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Entry {
    pub version: usize,
    pub component: Option<usize>,
    pub atom: Option<Atom>,
    pub content_sha256: String,
}
pub fn verify(
    base: &Model,
    records: &[Record<'_>],
    cache: &Cache,
    history: &[Entry],
) -> Result<usize> {
    cache.validate(base)?;
    if records.len() != cache.identities.len() {
        return Err("history record count changed".into());
    }
    let mut grouped = vec![Vec::new(); history.len() + 1];
    for (id, (&version, r)) in cache.origins.iter().zip(records).enumerate() {
        if retained::index::identity(r) != cache.identities[id] {
            return Err("history record identity changed".into());
        }
        grouped
            .get_mut(version)
            .ok_or("history version missing")?
            .push(id);
    }
    let mut model = base.clone();
    let mut checked = 0;
    for (v, ids) in grouped.iter().enumerate() {
        if v > 0 {
            let e = &history[v - 1];
            if e.version != v {
                return Err("history version order changed".into());
            }
            match (e.component, &e.atom) {
                (Some(j), Some(a)) if j > 0 && j < model.atoms.len() => {
                    a.validate(model.objects, model.family)?;
                    model.atoms[j] = a.clone();
                }
                (None, None) => {}
                _ => return Err("invalid content history edit".into()),
            }
            if readouts::source::content_hash(&model)? != e.content_sha256 {
                return Err("content history hash changed".into());
            }
        }
        for &id in ids {
            for (j, a) in model.atoms.iter().enumerate() {
                let state = a.state(records[id].prefix)?;
                for (actual, stored) in state.iter().zip(&cache.states[id][j]) {
                    if actual.re.to_bits() != stored.re.to_bits()
                        || actual.im.to_bits() != stored.im.to_bits()
                    {
                        return Err("stored state disagrees with recorded content origin".into());
                    }
                }
            }
            checked += 1;
        }
    }
    if checked != records.len() {
        return Err("incomplete historical state replay".into());
    }
    Ok(checked)
}
