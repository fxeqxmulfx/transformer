//! Original training rows and supervised positions, without source/task fields.
//! Caller reservation must include the full owning data capacities, metadata and
//! other simultaneously live storage. Slice lengths supply only a lower bound.
use crate::{
    Result, Work,
    features::{self, Alphabet, Program},
    partition::{self, Partition},
};
use joint::admission::Budget;
use probes::boundary::{Training, TrainingRow};
use serde::{Deserialize, Serialize};
use std::mem::size_of;
pub const SCRATCH: u128 = 1024 * 1024;
pub const PROCESS: u64 = 4 * 1024 * 1024 * 1024;

pub(crate) fn vector<T>(n: usize) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(n)?;
    Ok(out)
}
pub(crate) fn bytes<T>(v: &Vec<T>) -> u128 {
    v.capacity() as u128 * size_of::<T>() as u128
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub row: usize,
    pub position: usize,
    pub label: usize,
}
pub struct Prepared<'a> {
    pub(crate) alphabet: Alphabet,
    pub(crate) programs: Vec<Program>,
    pub(crate) partition: Partition,
    rows: Vec<TrainingRow<'a>>,
    observations: Vec<Observation>,
    labels: Vec<usize>,
    caller_bytes: u128,
    process_bytes: u64,
}
impl<'a> Prepared<'a> {
    pub fn new(
        training: Training<'a>,
        caller_bytes: u128,
        process_bytes: u64,
        work: &mut Work,
    ) -> Result<Self> {
        if !(2..=1024).contains(&training.vocabulary)
            || training.rows.is_empty()
            || training.rows.len() > 8192
        {
            return Err("training vocabulary or original row count outside admission".into());
        }
        let mut count = 0usize;
        let mut tokens = 0usize;
        for (index, row) in training.rows.iter().enumerate() {
            if row.original_index != index || row.tokens.len() != row.targets.len() {
                return Err("original training row identity or mask length changed".into());
            }
            features::validate_prefix(row.tokens, training.vocabulary)?;
            tokens = tokens
                .checked_add(row.tokens.len())
                .ok_or("training token count overflow")?;
            if tokens > 2_097_152 {
                return Err("training token admission exceeded".into());
            }
            for &label in row.targets {
                if label == -100 {
                    continue;
                }
                if label < 0 || label as usize >= training.vocabulary {
                    return Err("invalid original target".into());
                }
                count += 1;
            }
        }
        if count == 0 || count > 65536 || caller_bytes < 16 * tokens as u128 {
            return Err("supervised count or owning input reservation outside admission".into());
        }
        let forecast = bytes(&training.rows)
            + count as u128 * (size_of::<Observation>() + size_of::<usize>()) as u128
            + features::PROGRAMS as u128
                * (size_of::<Program>() + 2 * size_of::<partition::Decision>() + size_of::<usize>())
                    as u128
            + 16 * training.vocabulary as u128
            + SCRATCH;
        Budget::new(
            process_bytes,
            caller_bytes
                .checked_add(forecast)
                .ok_or("input reservation overflow")?,
        )?;
        work.charge(64 * tokens as u128)?;
        let mut present = vector::<bool>(training.vocabulary)?;
        present.resize(training.vocabulary, false);
        let mut observations = vector(count)?;
        let mut labels = vector(count)?;
        for (row_id, row) in training.rows.iter().enumerate() {
            // A public separated sequence contributes only the input payload.
            // Dense sequences contribute only prefixes visible at supervision.
            let last = row.targets.iter().rposition(|&x| x != -100);
            if let Some(last) = last {
                let visible = &row.tokens[..=last];
                let input = features::separator(visible).map_or(visible, |sep| &visible[1..sep]);
                for &token in input {
                    present[token as usize] = true;
                }
            }
            for (position, &label) in row.targets.iter().enumerate() {
                if label != -100 {
                    observations.push(Observation {
                        row: row_id,
                        position,
                        label: label as usize,
                    });
                    labels.push(label as usize);
                }
            }
        }
        let mut symbols = vector(training.vocabulary)?;
        for (token, &seen) in present.iter().enumerate() {
            if seen {
                symbols.push(token as i64);
            }
        }
        drop(present);
        let alphabet = Alphabet {
            vocabulary: training.vocabulary,
            symbols,
        };
        alphabet.validate()?;
        let mut out = Self {
            alphabet,
            programs: features::catalog(),
            partition: Partition::default(),
            rows: training.rows,
            observations,
            labels,
            caller_bytes,
            process_bytes,
        };
        out.partition = partition::build(&out, work)?;
        out.budget()?;
        Ok(out)
    }
    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }
    pub fn alphabet(&self) -> &Alphabet {
        &self.alphabet
    }
    pub fn programs(&self) -> &[Program] {
        &self.programs
    }
    pub fn partition(&self) -> &Partition {
        &self.partition
    }
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }
    pub fn prefix(&self, index: usize) -> Result<&[i64]> {
        let at = self
            .observations
            .get(index)
            .ok_or("original observation index outside input")?;
        Ok(&self.rows[at.row].tokens[..=at.position])
    }
    pub fn allocation_bytes(&self) -> u128 {
        size_of::<Self>() as u128
            + bytes(&self.rows)
            + bytes(&self.observations)
            + bytes(&self.labels)
            + bytes(&self.programs)
            + bytes(&self.alphabet.symbols)
            + self.partition.allocation_bytes()
    }
    pub fn budget(&self) -> Result<Budget> {
        Budget::new(
            self.process_bytes,
            self.caller_bytes
                .checked_add(self.allocation_bytes() + SCRATCH)
                .ok_or("complete external reservation overflow")?,
        )
    }
}
