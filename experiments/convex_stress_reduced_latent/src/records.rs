//! CM68 after Lean84a455b7: lossless observed words with original record IDs.
//! Every state and label is range checked before conversion. At most one row
//! is expanded during a visit. No decoder parameter or head is compressed.
use crate::{domain::Domain, *};
use std::ops::Range;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packed {
    shape: Shape,
    /// Row-major storage preserves sequential CE access.
    states: Vec<u16>,
    labels: Vec<u16>,
}
impl Packed {
    /// Column-order construction matches the frozen program-major state audit.
    /// The callback is invoked once for every original record/program pair.
    pub fn from_columns<F>(
        shape: Shape,
        labels: &[usize],
        mut state: F,
        work: &mut Work,
    ) -> Result<Self>
    where
        F: FnMut(usize, usize, &mut Work) -> Result<usize>,
    {
        shape.admit_packed(labels.len(), 0, 0)?;
        if labels.is_empty() || labels.iter().any(|&y| y >= shape.classes) {
            return Err("empty packed observations or raw label outside vocabulary".into());
        }
        work.charge(8 * labels.len() as u128 * shape.programs as u128 + 64 * labels.len() as u128)?;
        let labels: Vec<u16> = labels
            .iter()
            .map(|&y| u16::try_from(y))
            .collect::<std::result::Result<_, _>>()?;
        let mut states = vec![0u16; labels.len() * shape.programs];
        for p in 0..shape.programs {
            for row in 0..labels.len() {
                let value = state(row, p, work)?;
                if value >= shape.states {
                    return Err("packed state is outside the declared latent domain".into());
                }
                states[row * shape.programs + p] = u16::try_from(value)?;
            }
        }
        Ok(Self {
            shape,
            states,
            labels,
        })
    }
    /// Conversion controls may retain both representations at once; charge both.
    pub fn from_records(shape: Shape, records: &[Record], work: &mut Work) -> Result<Self> {
        shape.admit_storage(
            records.len(),
            0,
            0,
            records.len() as u128 * (10 * shape.programs as u128 + 64) + 8 * shape.programs as u128,
        )?;
        for row in records {
            Domain::record(shape, row)?;
        }
        let labels: Vec<_> = records.iter().map(|r| r.label).collect();
        Self::from_columns(shape, &labels, |row, p, _| Ok(records[row].states[p]), work)
    }
    pub fn len(&self) -> usize {
        self.labels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }
    pub fn state(&self, row: usize, program: usize) -> Result<usize> {
        if row >= self.len() || program >= self.shape.programs {
            return Err("packed record or program index outside original dimensions".into());
        }
        Ok(usize::from(
            self.states[row * self.shape.programs + program],
        ))
    }
    pub fn label(&self, row: usize) -> Result<usize> {
        self.labels
            .get(row)
            .map(|&x| usize::from(x))
            .ok_or_else(|| "packed record index outside original dimensions".into())
    }
    pub fn stored_bytes(&self) -> u64 {
        2 * (self.states.len() as u64 + self.labels.len() as u64)
    }
    pub fn scratch_bytes(&self) -> u64 {
        8 * self.shape.programs as u64
    }
}

#[derive(Clone, Copy)]
pub enum Source<'a> {
    Expanded(&'a [Record]),
    Packed(&'a Packed),
}
impl Source<'_> {
    pub fn len(self) -> usize {
        match self {
            Self::Expanded(r) => r.len(),
            Self::Packed(r) => r.len(),
        }
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    pub fn check_shape(self, shape: Shape) -> Result<()> {
        shape.validate()?;
        if let Self::Packed(p) = self
            && p.shape != shape
        {
            return Err("packed observation header differs from complete model shape".into());
        }
        Ok(())
    }
    pub fn admit(self, shape: Shape, unknown: usize, columns: usize) -> Result<u64> {
        self.check_shape(shape)?;
        match self {
            Self::Expanded(_) => shape.admit(self.len(), unknown, columns),
            Self::Packed(_) => shape.admit_packed(self.len(), unknown, columns),
        }
    }
    /// Iterate exactly the requested original index interval, with one reusable
    /// decoded row. The callback cannot retain a reference after the visit.
    pub fn visit<F>(self, range: Range<usize>, work: &mut Work, mut apply: F) -> Result<()>
    where
        F: FnMut(&Record, &mut Work) -> Result<()>,
    {
        if range.start > range.end || range.end > self.len() {
            return Err("observation range exceeds original record IDs".into());
        }
        match self {
            Self::Expanded(rows) => {
                for row in &rows[range] {
                    apply(row, work)?;
                }
            }
            Self::Packed(rows) => {
                work.charge(
                    8 * range.len() as u128 * rows.shape.programs as u128 + 2 * range.len() as u128,
                )?;
                let mut row = Record {
                    states: vec![0; rows.shape.programs],
                    label: 0,
                };
                for index in range {
                    let start = index * rows.shape.programs;
                    for (target, &word) in row
                        .states
                        .iter_mut()
                        .zip(&rows.states[start..start + rows.shape.programs])
                    {
                        *target = usize::from(word);
                    }
                    row.label = usize::from(rows.labels[index]);
                    apply(&row, work)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
