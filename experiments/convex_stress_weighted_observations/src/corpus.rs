//! Exact feature/observed-label fibers, CM60 Basic.lean at819d4547. Every
//! original ID and its order are retained; each batch recounts its own fibers.
use crate::*;
use joint::data::Record;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, ops::Range};

pub const MAX_ORIGINAL_RECORDS: usize = 2_000_000;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub records: usize,
    pub groups: usize,
    pub programs: usize,
    pub vocabulary: usize,
}
impl Limits {
    /// Includes feature keys, representatives, original IDs, counts, constraint
    /// copies and complete coefficients. Optimizer traces are charged separately.
    pub fn buffer_bytes(&self) -> Result<u64> {
        if !(2..=MAX_ORIGINAL_RECORDS).contains(&self.records)
            || !(1..=joint::MAX_RECORDS).contains(&self.groups)
            || self.groups > self.records
            || !(1..=joint::MAX_PROGRAMS).contains(&self.programs)
            || !(2..=1024).contains(&self.vocabulary)
        {
            return Err("weighted corpus dimensions refused".into());
        }
        let (n, g, p, v) = (
            self.records as u128,
            self.groups as u128,
            self.programs as u128,
            self.vocabulary as u128,
        );
        let bytes = 256 * 1024 * 1024 + 64 * (32 * n + g * (256 + 128 * p) + 256 * p * v);
        if bytes > joint::MAX_BYTES {
            return Err("weighted corpus buffer forecast exceeds56GiB".into());
        }
        Ok(bytes as u64)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Occurrence {
    pub id: usize,
    pub group: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Corpus {
    limits: Limits,
    groups: Vec<Record>,
    occurrences: Vec<Occurrence>,
    totals: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Batch {
    pub start: usize,
    pub end: usize,
    pub counts: Vec<(u32, u64)>,
}
impl Batch {
    pub fn records(&self) -> usize {
        self.end - self.start
    }
}

fn key(row: &Record) -> (usize, Vec<u64>) {
    (
        row.label,
        row.features
            .iter()
            .flatten()
            .map(|&x| {
                if x == 0. {
                    0.0_f64.to_bits()
                } else {
                    x.to_bits()
                }
            })
            .collect(),
    )
}

impl Corpus {
    /// Consume each original record once. Strictly increasing IDs make omitted,
    /// repeated or reordered source IDs visible; no ID is silently deduplicated.
    pub fn build(limits: Limits, rows: impl IntoIterator<Item = Result<Record>>) -> Result<Self> {
        limits.buffer_bytes()?;
        let mut out = Self {
            occurrences: Vec::with_capacity(limits.records),
            groups: vec![],
            totals: vec![],
            limits,
        };
        let mut keys = BTreeMap::new();
        for item in rows {
            if out.occurrences.len() == out.limits.records {
                return Err("original stream exceeds declared count".into());
            }
            let mut row = item?;
            joint::data::features(&row.features, out.limits.programs)?;
            if row.label >= out.limits.vocabulary
                || out.occurrences.last().is_some_and(|last| last.id >= row.id)
            {
                return Err("invalid observed label or nonincreasing original ID".into());
            }
            for x in row.features.iter_mut().flatten() {
                if *x == 0. {
                    *x = 0.;
                }
            }
            let row_key = key(&row);
            let group = if let Some(&g) = keys.get(&row_key) {
                g
            } else {
                if out.groups.len() == out.limits.groups {
                    return Err("exact feature/label group cap exceeded".into());
                }
                let g = out.groups.len() as u32;
                keys.insert(row_key, g);
                out.groups.push(row.clone());
                out.totals.push(0);
                g
            };
            out.totals[group as usize] += 1;
            out.occurrences.push(Occurrence { id: row.id, group });
        }
        drop(keys);
        out.validate()?;
        Ok(out)
    }

    pub fn validate(&self) -> Result<()> {
        self.limits.buffer_bytes()?;
        if self.occurrences.len() != self.limits.records
            || self.groups.is_empty()
            || self.groups.len() > self.limits.groups
            || self.totals.len() != self.groups.len()
            || self.occurrences.windows(2).any(|w| w[0].id >= w[1].id)
        {
            return Err("saved corpus dimensions or original IDs changed".into());
        }
        let mut keys = BTreeMap::new();
        for (g, row) in self.groups.iter().enumerate() {
            joint::data::features(&row.features, self.limits.programs)?;
            if row.label >= self.limits.vocabulary || keys.insert(key(row), g).is_some() {
                return Err("invalid or duplicated saved feature/label group".into());
            }
        }
        let mut counts = vec![0; self.groups.len()];
        let mut first = vec![None; self.groups.len()];
        for r in &self.occurrences {
            let g = r.group as usize;
            if g >= self.groups.len() {
                return Err("original record refers to an absent group".into());
            }
            counts[g] += 1;
            first[g].get_or_insert(r.id);
        }
        if counts != self.totals
            || self
                .groups
                .iter()
                .enumerate()
                .any(|(g, r)| first[g] != Some(r.id))
        {
            return Err("saved original multiplicities or representatives changed".into());
        }
        Ok(())
    }

    pub fn batch(&self, range: Range<usize>) -> Result<Batch> {
        if range.start >= range.end || range.end > self.occurrences.len() {
            return Err("empty or out-of-range original minibatch".into());
        }
        let counts = if range == (0..self.occurrences.len()) {
            self.totals
                .iter()
                .enumerate()
                .map(|(g, &n)| (g as u32, n))
                .collect()
        } else {
            let mut counts = BTreeMap::<u32, u64>::new();
            for row in &self.occurrences[range.clone()] {
                *counts.entry(row.group).or_default() += 1;
            }
            counts.into_iter().collect()
        };
        Ok(Batch {
            start: range.start,
            end: range.end,
            counts,
        })
    }

    pub fn check_batch(&self, batch: &Batch) -> Result<()> {
        if *batch != self.batch(batch.start..batch.end)? {
            return Err("batch multiplicities differ from original record range".into());
        }
        Ok(())
    }
    pub fn limits(&self) -> &Limits {
        &self.limits
    }
    pub fn groups(&self) -> &[Record] {
        &self.groups
    }
    pub fn occurrences(&self) -> &[Occurrence] {
        &self.occurrences
    }

    /// Bind original IDs, order, every complete feature, observed label and count
    /// without materializing another serialized copy of a million-record pool.
    pub fn fingerprint(&self) -> Result<String> {
        struct Sink(Sha256);
        impl std::io::Write for Sink {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.update(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut sink = Sink(Sha256::new());
        serde_json::to_writer(&mut sink, self)?;
        const HEX: &[u8; 16] = b"0123456789abcdef";
        Ok(sink
            .0
            .finalize()
            .iter()
            .flat_map(|b| {
                [
                    HEX[(b >> 4) as usize] as char,
                    HEX[(b & 15) as usize] as char,
                ]
            })
            .collect())
    }
}
