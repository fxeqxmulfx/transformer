//! CM89 positive diagonal scaling from original supervised input prefixes.
//! Counts use all indexed records and squared multiplicities, never labels.
use crate::{Result, domain::Shape};
use std::collections::BTreeMap;

pub struct FrequencyScale {
    pub shape: Shape,
    pub records: u64,
    squares: BTreeMap<(usize, usize), u64>,
}

impl FrequencyScale {
    pub fn from_prefixes<'a>(
        shape: Shape,
        prefixes: impl Iterator<Item = &'a [i64]>,
    ) -> Result<Self> {
        let mut result = Self {
            shape,
            records: 0,
            squares: BTreeMap::new(),
        };
        for prefix in prefixes {
            result.records = result
                .records
                .checked_add(1)
                .ok_or("scaling record overflow")?;
            let input = shape.addresses(prefix)?;
            for &(d, count) in &input.columns {
                let count = u64::try_from(count)?;
                let square = count
                    .checked_mul(count)
                    .ok_or("squared response overflow")?;
                for &q in &input.queries {
                    let value = result.squares.entry((q, d)).or_default();
                    *value = value.checked_add(square).ok_or("frequency sum overflow")?;
                }
            }
        }
        if result.records == 0
            || result.records >= (1u64 << 53)
            || result.squares.values().any(|&n| n >= (1u64 << 53))
        {
            return Err("empty or inexact integer scaling counts".into());
        }
        Ok(result)
    }

    pub fn value(&self, query: usize, column: usize) -> Result<f64> {
        if query >= self.shape.queries || column >= self.shape.columns {
            return Err("scaling coordinate outside complete domain".into());
        }
        let square = self.squares.get(&(query, column)).copied().unwrap_or(0);
        let scale = (self.records + 1) as f64 / (square + 1) as f64;
        if !scale.is_finite() || scale <= 0. {
            return Err("nonpositive diagonal scale".into());
        }
        Ok(scale)
    }
}
