//! Indexed original supervised occurrences; identical records retain distinct IDs.
use crate::Result;
use baseline::data::Example;
use bilinear::{experiment as original, train::Record};

pub struct Index {
    offsets: Vec<usize>,
}
impl Index {
    pub fn new(examples: &[Example]) -> Result<Self> {
        let mut offsets = vec![0usize];
        for e in examples {
            if e.tokens.len() != e.targets.len() {
                return Err("original target mask shape changed".into());
            }
            offsets.push(
                offsets
                    .last()
                    .ok_or("offset")?
                    .checked_add(e.targets.iter().filter(|&&t| t != -100).count())
                    .ok_or("record count overflow")?,
            );
        }
        if offsets.last() == Some(&0) {
            return Err("no supervised records".into());
        }
        Ok(Self { offsets })
    }
    pub fn batch<'a>(
        &self,
        examples: &'a [Example],
        indices: &[usize],
    ) -> Result<(Vec<usize>, Vec<Record<'a>>)> {
        if examples.len() + 1 != self.offsets.len() {
            return Err("example count changed".into());
        }
        let records = original::records(examples, indices)?;
        let mut ids = Vec::with_capacity(records.len());
        for &i in indices {
            let end = *self.offsets.get(i + 1).ok_or("example index")?;
            ids.extend(self.offsets[i]..end);
        }
        if ids.len() != records.len() {
            return Err("supervised mask changed".into());
        }
        Ok((ids, records))
    }
}
pub fn identity(r: &Record<'_>) -> String {
    let mut bytes = Vec::with_capacity(16 + 8 * r.prefix.len());
    bytes.extend_from_slice(&(r.prefix.len() as u64).to_le_bytes());
    for token in r.prefix {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    bytes.extend_from_slice(&(r.target as u64).to_le_bytes());
    original::hash(&bytes)
}
