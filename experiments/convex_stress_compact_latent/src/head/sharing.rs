//! CM71 ordered equality, Lean 91281572: SharedColumns.stored_shared_eq.
//! The union of every coefficient exception and every known-only active class
//! is conservative across ALL blocks. Only its complement shares computation.
//! No coefficient, norm term, output class or arbitrary completion is removed.
use super::*;
use crate::memory;
use std::mem::size_of;

#[derive(Debug)]
pub(super) struct Sharing {
    representatives: Vec<u16>,
    duplicates: Vec<u16>,
    representative: u16,
}
impl Sharing {
    pub fn build(head: &Head, limit: u64) -> Result<Option<Self>> {
        let classes = head.shape.classes;
        let mut marked = memory::vector::<u8>(
            classes,
            memory::remaining(limit, size_of::<Self>() as u128)?,
        )?;
        marked.resize(classes, 0);
        for e in &head.exceptions {
            marked[e.class as usize] = 1;
        }
        for &c in &head.active {
            marked[c as usize] = 1;
        }
        let common = marked.iter().filter(|&&flag| flag == 0).count();
        if common < 2 {
            return Ok(None);
        }
        let representative = marked.iter().position(|&flag| flag == 0).unwrap() as u16;
        let fixed = size_of::<Self>() as u128 + memory::vector_bytes::<u8>(marked.capacity());
        let mut representatives =
            memory::vector::<u16>(classes - common + 1, memory::remaining(limit, fixed)?)?;
        let mut duplicates = memory::vector::<u16>(
            common - 1,
            memory::remaining(
                limit,
                fixed + memory::vector_bytes::<u16>(representatives.capacity()),
            )?,
        )?;
        for (c, &flag) in marked.iter().enumerate() {
            if flag != 0 || c == representative as usize {
                representatives.push(c as u16);
            } else {
                duplicates.push(c as u16);
            }
        }
        Ok(Some(Self {
            representatives,
            duplicates,
            representative,
        }))
    }
    pub fn representatives(&self) -> &[u16] {
        &self.representatives
    }
    pub fn heap_bytes(&self) -> usize {
        (self.representatives.capacity() + self.duplicates.capacity()) * size_of::<u16>()
    }
    pub fn expand<T: Clone>(&self, values: &mut [T]) {
        // Borrow the original representative. BigInt expansion creates only
        // each original output column's allocation, with no extra integer copy.
        let representative = self.representative as usize;
        for &c in &self.duplicates {
            let c = c as usize;
            if c > representative {
                let (left, right) = values.split_at_mut(c);
                right[0].clone_from(&left[representative]);
            } else {
                let (left, right) = values.split_at_mut(representative);
                left[c].clone_from(&right[0]);
            }
        }
    }
    pub fn validate(&self, head: &Head) -> Result<()> {
        let classes = head.shape.classes;
        if self.representative as usize >= classes
            || self.duplicates.is_empty()
            || self.representatives.len() + self.duplicates.len() != classes
            || self.representatives.windows(2).any(|w| w[0] >= w[1])
            || self.duplicates.windows(2).any(|w| w[0] >= w[1])
            || self
                .representatives
                .last()
                .is_some_and(|&c| c as usize >= classes)
            || self
                .duplicates
                .last()
                .is_some_and(|&c| c as usize >= classes)
            || self
                .representatives
                .binary_search(&self.representative)
                .is_err()
        {
            return Err("invalid complete column computation plan".into());
        }
        for c in 0..classes as u16 {
            if self.representatives.binary_search(&c).is_ok()
                == self.duplicates.binary_search(&c).is_ok()
            {
                return Err("computation plan omitted or duplicated an output class".into());
            }
        }
        for c in head
            .exceptions
            .iter()
            .map(|e| e.class)
            .chain(head.active.iter().copied())
        {
            if c == self.representative || self.representatives.binary_search(&c).is_err() {
                return Err(
                    "computation plan merged different coefficients or active supports".into(),
                );
            }
        }
        Ok(())
    }
}

impl Head {
    /// Complete representatives for a dense coefficient-reading pass. That
    /// pass still performs its own explicit zero operations in inactive blocks.
    pub(super) fn computation_classes(&self) -> impl Iterator<Item = usize> + '_ {
        let n = self
            .sharing
            .as_ref()
            .map_or(self.shape.classes, |s| s.representatives.len());
        (0..n).map(|i| {
            self.sharing
                .as_ref()
                .map_or(i, |s| s.representatives[i] as usize)
        })
    }
    /// Physical columns evaluated once per Fourier term. Logical work still
    /// charges every original active class; this diagnostic changes no budget.
    pub fn evaluated_columns(&self, states: &[usize]) -> Result<usize> {
        self.query(states)?;
        Ok(states
            .iter()
            .enumerate()
            .map(|(p, &s)| {
                let b = &self.blocks[(p * self.shape.states + s) / 4];
                self.computation_columns(b).len()
            })
            .sum())
    }
}

#[cfg(test)]
mod tests;
