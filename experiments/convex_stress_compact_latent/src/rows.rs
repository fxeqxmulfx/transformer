//! CM70 exact value storage. Unlike the general Lean exception codec, this
//! parameter-row specialization uses a full dense fallback whenever any word
//! differs. Original row IDs, order and logical class counts never change.
use crate::Result;
use std::mem::size_of;

const MAX_BYTES: u64 = 96_000_000_000;
const SHARED: usize = usize::MAX;

#[derive(Clone, Debug)]
struct Row {
    value: f64,
    offset: usize,
}

#[derive(Debug)]
pub struct Rows {
    classes: usize,
    rows: Vec<Row>,
    dense: Vec<f64>,
}

/// Owns an admitted row table. The limit covers its capacities, builder
/// metadata and one class-sized scratch row. The caller must separately
/// charge other simultaneously live model tensors and source buffers.
pub struct Builder {
    value: Rows,
    expected: usize,
    limit: u64,
}

fn footprint(rows: usize, dense: usize, classes: usize) -> Result<u64> {
    let bytes = size_of::<Builder>() as u128
        + size_of::<Vec<f64>>() as u128
        + rows as u128 * size_of::<Row>() as u128
        + dense as u128 * size_of::<f64>() as u128
        + classes as u128 * size_of::<f64>() as u128;
    Ok(u64::try_from(bytes)?)
}

impl Builder {
    pub fn new(classes: usize, rows: usize, limit: u64) -> Result<Self> {
        if !(1..=1024).contains(&classes) || limit == 0 || limit > MAX_BYTES {
            return Err("invalid row shape or storage limit".into());
        }
        rows.checked_mul(classes)
            .ok_or("logical row size overflow")?;
        if footprint(rows, 0, classes)? > limit {
            return Err("complete row metadata exceeds admitted storage".into());
        }
        let mut value = Rows {
            classes,
            rows: Vec::new(),
            dense: Vec::new(),
        };
        value.rows.try_reserve_exact(rows)?;
        let out = Self {
            value,
            expected: rows,
            limit,
        };
        out.check()?;
        Ok(out)
    }
    fn check(&self) -> Result<()> {
        if footprint(
            self.value.rows.capacity(),
            self.value.dense.capacity(),
            self.value.classes,
        )? > self.limit
        {
            return Err("actual row capacities exceed admitted storage".into());
        }
        Ok(())
    }
    fn next(&self) -> Result<()> {
        if self.value.rows.len() >= self.expected {
            return Err("more rows than the complete declared table".into());
        }
        Ok(())
    }
    pub fn push_repeated(&mut self, value: f64) -> Result<()> {
        self.next()?;
        self.value.rows.push(Row {
            value,
            offset: SHARED,
        });
        Ok(())
    }
    pub fn push(&mut self, values: &[f64]) -> Result<()> {
        self.next()?;
        if values.len() != self.value.classes {
            return Err("incomplete original class row".into());
        }
        if values.iter().all(|x| x.to_bits() == values[0].to_bits()) {
            return self.push_repeated(values[0]);
        }
        let offset = self.value.dense.len();
        let required = offset
            .checked_add(values.len())
            .ok_or("dense row size overflow")?;
        let old_capacity = self.value.dense.capacity();
        if required > old_capacity {
            let peak = |capacity: usize| -> Result<u64> {
                footprint(
                    self.value.rows.capacity(),
                    old_capacity
                        .checked_add(capacity)
                        .ok_or("row reallocation overflow")?,
                    self.value.classes,
                )
            };
            let mut capacity = required
                .max(old_capacity.saturating_mul(2))
                .min(self.expected * self.value.classes);
            if peak(capacity)? > self.limit {
                capacity = required;
            }
            if peak(capacity)? > self.limit {
                return Err(
                    "complete nonuniform fallback and growth scratch exceed admitted storage"
                        .into(),
                );
            }
            // A moving reallocation can temporarily retain the entire old buffer.
            self.value.dense.try_reserve_exact(capacity - offset)?;
            if peak(self.value.dense.capacity())? > self.limit {
                return Err("actual row reallocation exceeds admitted storage".into());
            }
        }
        self.check()?;
        self.value.dense.extend_from_slice(values);
        self.value.rows.push(Row {
            value: values[0],
            offset,
        });
        Ok(())
    }
    pub fn finish(self) -> Result<Rows> {
        if self.value.rows.len() != self.expected {
            return Err("missing original rows in stored table".into());
        }
        self.check()?;
        Ok(self.value)
    }
}

impl Rows {
    /// Exact word equality, established by the builder; this reports storage
    /// without restricting what values an arbitrary row may contain.
    pub(crate) fn repeated(&self, row: usize) -> Option<f64> {
        let row = self.rows.get(row)?;
        (row.offset == SHARED).then_some(row.value)
    }
    pub fn try_clone(&self, limit: u64) -> Result<Self> {
        let mut out = Builder::new(self.classes, self.rows(), limit)?;
        for row in &self.rows {
            if row.offset == SHARED {
                out.push_repeated(row.value)?;
            } else {
                out.push(&self.dense[row.offset..row.offset + self.classes])?;
            }
        }
        out.finish()
    }
    pub fn from_dense(classes: usize, values: &[f64], limit: u64) -> Result<Self> {
        if classes == 0 || !values.len().is_multiple_of(classes) {
            return Err("incomplete flattened class table".into());
        }
        let mut out = Builder::new(classes, values.len() / classes, limit)?;
        for row in values.chunks_exact(classes) {
            out.push(row)?;
        }
        out.finish()
    }
    pub fn classes(&self) -> usize {
        self.classes
    }
    pub fn rows(&self) -> usize {
        self.rows.len()
    }
    pub fn len(&self) -> usize {
        self.rows.len() * self.classes
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    pub fn dense_rows(&self) -> usize {
        self.dense.len() / self.classes
    }
    pub fn stored_values(&self) -> usize {
        self.dense.len() + self.rows.len() - self.dense_rows()
    }
    /// Actual owned capacities, not the count of logical coordinates.
    pub fn allocation_bytes(&self) -> u64 {
        (size_of::<Self>()
            + self.rows.capacity() * size_of::<Row>()
            + self.dense.capacity() * size_of::<f64>()) as u64
    }
    pub fn get(&self, row: usize, class: usize) -> Option<&f64> {
        let stored = self.rows.get(row)?;
        if class >= self.classes {
            return None;
        }
        Some(if stored.offset == SHARED {
            &stored.value
        } else {
            &self.dense[stored.offset + class]
        })
    }
    pub fn iter(&self) -> Values<'_> {
        Values {
            table: self,
            next: 0,
            end: self.len(),
        }
    }
    pub fn row(&self, row: usize) -> Option<Values<'_>> {
        if row >= self.rows.len() {
            return None;
        }
        Some(Values {
            table: self,
            next: row * self.classes,
            end: (row + 1) * self.classes,
        })
    }
    /// Visit every original coordinate in its original order, even in a
    /// shared row. Stateful closures and repeated floating additions therefore
    /// receive the same sequence as the complete explicit representation.
    pub fn zip_map(
        &self,
        other: &Self,
        limit: u64,
        mut f: impl FnMut(f64, f64) -> f64,
    ) -> Result<Self> {
        if self.rows() != other.rows() || self.classes != other.classes {
            return Err("logical row dimensions differ".into());
        }
        let mut out = Builder::new(self.classes, self.rows(), limit)?;
        let mut scratch = Vec::new();
        scratch.try_reserve_exact(self.classes)?;
        if scratch.capacity() != self.classes {
            return Err("scratch allocator capacity differs from its admitted size".into());
        }
        for r in 0..self.rows() {
            scratch.clear();
            for c in 0..self.classes {
                scratch.push(f(*self.get(r, c).unwrap(), *other.get(r, c).unwrap()));
            }
            out.push(&scratch)?;
        }
        out.finish()
    }
}

pub struct Values<'a> {
    table: &'a Rows,
    next: usize,
    end: usize,
}
impl<'a> Iterator for Values<'a> {
    type Item = &'a f64;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.end {
            return None;
        }
        let index = self.next;
        self.next += 1;
        self.table
            .get(index / self.table.classes, index % self.table.classes)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.end - self.next;
        (n, Some(n))
    }
}
impl ExactSizeIterator for Values<'_> {}
