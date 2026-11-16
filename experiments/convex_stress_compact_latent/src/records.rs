//! CM68 observed words with the original record/program indices, ported from
//! reduced_latent at88fb15dc. Only admission changes: count actual capacities
//! independently of dense model storage. Caller-owned inputs are separate.
use crate::{domain::Domain, memory, *};
use std::{mem::size_of, ops::Range};

#[derive(Debug)]
pub struct Packed {
    shape: Shape,
    states: Vec<u16>,
    labels: Vec<u16>,
}
pub fn packed_bytes(shape: Shape, records: usize) -> Result<u128> {
    shape.validate()?;
    if records == 0 || records > 65536 {
        return Err("invalid original packed record count".into());
    }
    let words = records
        .checked_mul(shape.programs)
        .ok_or("packed state count overflow")?;
    Ok(size_of::<Packed>() as u128 + 2 * (words as u128 + records as u128))
}
impl Packed {
    /// Invoke the state callback once for every original pair in the original
    /// program-major order. The callback's own storage belongs to the caller.
    pub fn from_columns<F>(
        shape: Shape,
        labels: &[usize],
        mut state: F,
        limit: u64,
        work: &mut Work,
    ) -> Result<Self>
    where
        F: FnMut(usize, usize, &mut Work) -> Result<usize>,
    {
        memory::remaining(limit, packed_bytes(shape, labels.len())?)?;
        if labels.iter().any(|&y| y >= shape.classes) {
            return Err("raw label outside packed vocabulary".into());
        }
        work.charge(8 * labels.len() as u128 * shape.programs as u128 + 64 * labels.len() as u128)?;
        let mut words = memory::vector(labels.len(), limit)?;
        words.extend(labels.iter().map(|&y| y as u16));
        let mut states = memory::vector(labels.len() * shape.programs, limit)?;
        states.resize(labels.len() * shape.programs, 0);
        for p in 0..shape.programs {
            for row in 0..labels.len() {
                let value = state(row, p, work)?;
                if value >= shape.states {
                    return Err("packed state is outside the declared latent domain".into());
                }
                states[row * shape.programs + p] = u16::try_from(value)?;
            }
        }
        let out = Self {
            shape,
            states,
            labels: words,
        };
        if out.allocation_bytes() != packed_bytes(shape, out.len())? {
            return Err("actual packed capacities differ from admission".into());
        }
        Ok(out)
    }
    /// Reference conversion. Input records remain separately owned; the local
    /// cap includes temporary labels together with the complete packed output.
    pub fn from_records(
        shape: Shape,
        records: &[Record],
        limit: u64,
        work: &mut Work,
    ) -> Result<Self> {
        packed_bytes(shape, records.len())?;
        let room = memory::remaining(limit, memory::vector_bytes::<usize>(records.len()))?;
        memory::remaining(room, packed_bytes(shape, records.len())?)?;
        for row in records {
            Domain::record(shape, row)?;
        }
        let mut labels = memory::vector(records.len(), limit)?;
        labels.extend(records.iter().map(|r| r.label));
        Self::from_columns(
            shape,
            &labels,
            |row, p, _| Ok(records[row].states[p]),
            room,
            work,
        )
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
    pub fn allocation_bytes(&self) -> u128 {
        size_of::<Self>() as u128
            + 2 * (self.states.capacity() as u128 + self.labels.capacity() as u128)
    }
    pub fn scratch_bytes(&self) -> u128 {
        size_of::<Record>() as u128 + self.shape.programs as u128 * size_of::<usize>() as u128
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
        if self.len() > 65536 {
            return Err("original record count exceeds admission".into());
        }
        if let Self::Packed(p) = self
            && (p.shape != shape || p.states.len() != p.len() * shape.programs)
        {
            return Err("packed observation header differs from complete model shape".into());
        }
        Ok(())
    }
    pub fn scratch_bytes(self) -> u128 {
        match self {
            Self::Expanded(_) => 0,
            Self::Packed(p) => p.scratch_bytes(),
        }
    }
    /// Keep every original ID and multiplicity, including a singleton tail.
    /// The caller reserves scratch_bytes() before invoking this operation.
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
                self.check_shape(rows.shape)?;
                work.charge(
                    8 * range.len() as u128 * rows.shape.programs as u128 + 2 * range.len() as u128,
                )?;
                let mut states = memory::vector(rows.shape.programs, 96_000_000_000)?;
                states.resize(rows.shape.programs, 0);
                let mut row = Record { states, label: 0 };
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
