//! Bounded-memory storage of original supervised prefixes. Fixed-width records
//! retain every label and repeated observation; minibatches address complete
//! original examples. File creation is exclusive and neither truncates nor
//! resumes an existing cache. Cache construction is part of training cost.
use crate::{Result, data, features::Bank, head};
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

pub mod objective;

pub const IO_BUFFER_BYTES: usize = 64 * 1024;
pub const DISK_LIMIT: u64 = 16 * 1024 * 1024 * 1024;
const HEADER_LIMIT: usize = 16384;

#[derive(Debug, Serialize)]
pub struct Allocation {
    pub explicit_buffer_bytes: usize,
    pub cache_bytes_upper_bound: u64,
    pub eligible: bool,
}

/// Charge two dense tensors, projection scratch, bounded I/O, one decoded row,
/// example offsets and two example orders. The OS cap must also cover the
/// original data, allocator overhead and temporary path states.
pub fn allocation(bank: &Bank, observations: usize, examples: usize) -> Result<Allocation> {
    bank.validate()?;
    if observations == 0 || examples == 0 || examples > observations {
        return Err("invalid streamed observation counts".into());
    }
    let bytes = bank
        .allocation_bytes(0)?
        .checked_add(IO_BUFFER_BYTES + 2 * HEADER_LIMIT + 16 * bank.paths() + 8)
        .and_then(|n| examples.checked_mul(32).and_then(|e| n.checked_add(e)))
        .ok_or("streamed memory estimate overflow")?;
    let cache_bytes = (observations as u64)
        .checked_mul(4 * (2 * bank.paths() as u64 + 1))
        .and_then(|n| {
            (examples as u64)
                .checked_mul(8)
                .and_then(|e| n.checked_add(e))
        })
        .and_then(|n| n.checked_add(HEADER_LIMIT as u64 + 24))
        .ok_or("streamed disk estimate overflow")?;
    Ok(Allocation {
        explicit_buffer_bytes: bytes,
        cache_bytes_upper_bound: cache_bytes,
        eligible: bytes <= head::BUFFER_LIMIT && cache_bytes <= DISK_LIMIT,
    })
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    bank: Bank,
    observations: usize,
    examples: usize,
}

pub struct StreamCache {
    reader: BufReader<File>,
    bank: Bank,
    example_offsets: Vec<usize>,
    record_start: u64,
    next_record: usize,
    record_bytes: Vec<u8>,
    features: Vec<u32>,
}

impl StreamCache {
    pub fn build(path: &Path, bank: &Bank, examples: &[data::Example]) -> Result<Self> {
        bank.validate()?;
        let mut n = 0usize;
        for e in examples {
            let count = e.targets.iter().filter(|y| **y >= 0).count();
            if e.tokens.len() != e.targets.len()
                || count == 0
                || e.targets.iter().any(|y| *y >= bank.vocab as i64)
            {
                return Err("invalid streamed training example".into());
            }
            n = n.checked_add(count).ok_or("observation count overflow")?;
        }
        if !allocation(bank, n, examples.len())?.eligible {
            return Err("streamed cache exceeds explicit memory or disk budget".into());
        }
        let header = serde_json::to_vec(&Header {
            bank: bank.clone(),
            observations: n,
            examples: examples.len(),
        })?;
        if header.len() > HEADER_LIMIT {
            return Err("oversized generated cache header".into());
        }
        let mut writer = BufWriter::with_capacity(
            IO_BUFFER_BYTES,
            OpenOptions::new().write(true).create_new(true).open(path)?,
        );
        writer.write_all(b"CDFEAT01")?;
        writer.write_all(&(header.len() as u64).to_le_bytes())?;
        writer.write_all(&header)?;
        let mut offset = 0u64;
        writer.write_all(&offset.to_le_bytes())?;
        for e in examples {
            offset += e.targets.iter().filter(|y| **y >= 0).count() as u64;
            writer.write_all(&offset.to_le_bytes())?;
        }
        for e in examples {
            for (position, &label) in e.targets.iter().enumerate() {
                if label < 0 {
                    continue;
                }
                writer.write_all(&(label as u32).to_le_bytes())?;
                for k in bank.encode(&e.tokens[..=position])? {
                    writer.write_all(&u32::try_from(k)?.to_le_bytes())?;
                }
            }
        }
        writer.flush()?;
        drop(writer);
        Self::open(path)
    }

    /// Check metadata, example boundaries and exact length before accessing
    /// records. Targets and feature ranges are checked as each row is read.
    pub fn open(path: &Path) -> Result<Self> {
        let mut reader = BufReader::with_capacity(IO_BUFFER_BYTES, File::open(path)?);
        let mut bytes = [0u8; 8];
        reader.read_exact(&mut bytes)?;
        if &bytes != b"CDFEAT01" {
            return Err("invalid streamed cache magic".into());
        }
        reader.read_exact(&mut bytes)?;
        let size = u64::from_le_bytes(bytes);
        if size > HEADER_LIMIT as u64 {
            return Err("oversized streamed cache header".into());
        }
        let mut raw = vec![0; size as usize];
        reader.read_exact(&mut raw)?;
        let header: Header = serde_json::from_slice(&raw)?;
        let plan = allocation(&header.bank, header.observations, header.examples)?;
        if !plan.eligible {
            return Err("saved streamed cache exceeds memory or disk budget".into());
        }
        let record_start = 16 + size + 8 * (header.examples as u64 + 1);
        let width = 2 * header.bank.paths();
        let record_size = 4 * (width + 1);
        let expected = record_start + record_size as u64 * header.observations as u64;
        if reader.get_ref().metadata()?.len() != expected {
            return Err("truncated or trailing streamed cache bytes".into());
        }
        let mut example_offsets = Vec::with_capacity(header.examples + 1);
        for _ in 0..=header.examples {
            reader.read_exact(&mut bytes)?;
            example_offsets.push(usize::try_from(u64::from_le_bytes(bytes))?);
        }
        if example_offsets[0] != 0
            || example_offsets.last() != Some(&header.observations)
            || example_offsets.windows(2).any(|p| p[0] >= p[1])
        {
            return Err("invalid streamed example boundaries".into());
        }
        Ok(Self {
            reader,
            bank: header.bank,
            example_offsets,
            record_start,
            next_record: 0,
            record_bytes: vec![0; record_size],
            features: vec![0; width],
        })
    }
    pub fn bank(&self) -> &Bank {
        &self.bank
    }
    pub fn observations(&self) -> usize {
        *self.example_offsets.last().unwrap()
    }
    pub fn full_order(&self) -> Vec<usize> {
        (0..self.example_offsets.len() - 1).collect()
    }
    pub fn width(&self) -> usize {
        self.features.len()
    }
    pub fn visit(
        &mut self,
        examples: &[usize],
        mut visit: impl FnMut(&[u32], usize) -> Result<()>,
    ) -> Result<usize> {
        if examples.is_empty() {
            return Err("empty streamed example order".into());
        }
        let mut count = 0;
        for &e in examples {
            if e >= self.example_offsets.len() - 1 {
                return Err("invalid streamed example index".into());
            }
            let start = self.example_offsets[e];
            if self.next_record != start {
                self.reader.seek(SeekFrom::Start(
                    self.record_start + start as u64 * self.record_bytes.len() as u64,
                ))?;
                self.next_record = start;
            }
            for _ in start..self.example_offsets[e + 1] {
                self.reader.read_exact(&mut self.record_bytes)?;
                self.next_record += 1;
                let label = u32::from_le_bytes(self.record_bytes[..4].try_into()?) as usize;
                for (feature, raw) in self
                    .features
                    .iter_mut()
                    .zip(self.record_bytes[4..].as_chunks::<4>().0)
                {
                    *feature = u32::from_le_bytes(*raw);
                }
                if label >= self.bank.vocab
                    || self.features.iter().enumerate().any(|(channel, k)| {
                        let base = (channel / 2) * self.bank.stride();
                        let range = if channel % 2 == 0 {
                            base..base + self.bank.vocab + 1
                        } else {
                            base + self.bank.vocab + 1..base + self.bank.stride()
                        };
                        !range.contains(&(*k as usize))
                    })
                {
                    return Err("invalid streamed target or feature coordinate".into());
                }
                visit(&self.features, label)?;
                count += 1;
            }
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests;
