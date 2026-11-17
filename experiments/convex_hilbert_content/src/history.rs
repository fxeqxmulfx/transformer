//! Replay the actual finite estimates at their recorded pre-update parameters.
use crate::{Result, block::Block, cache::Cache, estimate};
use bilinear::{
    experiment as original,
    train::{Mode, Record},
};
use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize)]
pub struct Entry {
    pub version: usize,
    pub block: Block,
    pub block_sha256: String,
}
impl Entry {
    pub fn new(version: usize, block: &Block) -> Result<Self> {
        Ok(Self {
            version,
            block: block.clone(),
            block_sha256: original::hash(&serde_json::to_vec(block)?),
        })
    }
}
pub fn validate(history: &[Entry], initial: &Block, endpoint: &Block) -> Result<()> {
    if history.is_empty() {
        return Err("missing complete parameter history".into());
    }
    for (i, e) in history.iter().enumerate() {
        e.block.validate()?;
        if e.version != i
            || e.block.objects() != initial.objects()
            || e.block.geometry != initial.geometry
            || original::hash(&serde_json::to_vec(&e.block)?) != e.block_sha256
        {
            return Err("historical parameter hash or version mismatch".into());
        }
    }
    if history[0].block_sha256 != Entry::new(0, initial)?.block_sha256
        || history.last().ok_or("history")?.block_sha256 != Entry::new(0, endpoint)?.block_sha256
    {
        return Err("history start or endpoint mismatch".into());
    }
    Ok(())
}
pub fn replay_record(
    cache: &Cache,
    history: &[Entry],
    id: usize,
    r: &Record<'_>,
    base: &[f64],
    query: (f64, Mode, f64),
) -> Result<usize> {
    if cache.identities.get(id) != Some(&retained::index::identity(r)) {
        return Err("historical occurrence changed".into());
    }
    let version = cache.origins[id].ok_or("unobserved history record")?;
    if version + 1 >= history.len() {
        return Err("estimate origin is not a pre-update version".into());
    }
    let e = &history[version];
    let result = estimate::batch(
        &e.block,
        &[base.to_vec()],
        std::slice::from_ref(r),
        query.0,
        query.1,
        query.2,
    )?;
    if result.values[0].len() != cache.values[id].len()
        || result.values[0]
            .iter()
            .zip(&cache.values[id])
            .any(|(x, y)| x.to_bits() != y.to_bits())
    {
        return Err("cached estimate disagrees with its actual recorded origin".into());
    }
    Ok(result.ce_evaluations)
}
