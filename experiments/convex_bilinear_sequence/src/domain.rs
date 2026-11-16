//! CM86 Reader/Sequence: raw object identities, positions and adjacent counts.
//! No target, task name, arithmetic or answer program enters these addresses.
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Shape {
    pub objects: usize,
    pub context: usize,
    pub queries: usize,
    pub columns: usize,
}

impl Shape {
    pub fn new(objects: usize, context: usize) -> Result<Self> {
        if objects == 0 || context == usize::MAX {
            return Err("empty vocabulary or overflowing prefix cap".into());
        }
        let queries = objects.checked_add(1).ok_or("query overflow")?;
        let positions = objects.checked_mul(context).ok_or("position overflow")?;
        let pairs = if context >= 2 {
            objects.checked_mul(objects).ok_or("pair overflow")?
        } else {
            0
        };
        let columns = 1usize
            .checked_add(positions)
            .and_then(|n| n.checked_add(pairs))
            .ok_or("column overflow")?;
        Ok(Self {
            objects,
            context,
            queries,
            columns,
        })
    }

    pub fn real_coordinates(self) -> u128 {
        2 * self.objects as u128 * self.queries as u128 * self.columns as u128
    }

    /// Conservative simultaneous-state/workspace reservation. Logical absent
    /// columns are exact zeros; any new nonzero column consumes this budget.
    pub fn reservation(self, stored: usize) -> Result<u64> {
        if stored > self.columns {
            return Err("stored support exceeds complete domain".into());
        }
        let v = self.objects as u128;
        let q = self.queries as u128;
        let k = stored as u128;
        u64::try_from(16 * (8 * v * q * k + 4 * v * q * q + 8 * q * k + 8 * q * q))
            .map_err(Into::into)
    }

    pub fn addresses(self, prefix: &[i64]) -> Result<Addresses> {
        if prefix.len() > self.context + 1 {
            return Err("prefix exceeds declared capacity; no truncation".into());
        }
        if prefix.iter().any(|&t| t < 0 || t as usize >= self.objects) {
            return Err("object outside original vocabulary".into());
        }
        let mut queries = vec![0];
        if let Some(&last) = prefix.last() {
            queries.push(1 + last as usize);
        }
        let past = &prefix[..prefix.len().saturating_sub(1)];
        let mut columns = BTreeMap::from([(0, 1usize)]);
        for (i, &t) in past.iter().enumerate() {
            *columns
                .entry(1 + i * self.objects + t as usize)
                .or_default() += 1;
        }
        for pair in past.windows(2) {
            let d = 1
                + self.context * self.objects
                + pair[0] as usize * self.objects
                + pair[1] as usize;
            *columns.entry(d).or_default() += 1;
        }
        Ok(Addresses {
            queries,
            columns: columns.into_iter().collect(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Addresses {
    pub queries: Vec<usize>,
    pub columns: Vec<(usize, usize)>,
}
