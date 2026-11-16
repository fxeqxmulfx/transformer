//! CM70 native allocation ledger, not a theorem about process RSS. Keep all
//! input capacities and caller-owned storage alongside every nested local
//! training buffer. The prospective runner must provide its actual external
//! reservation, enforce the process cap and account for other live processes.
use crate::{
    domain::{self, Domain},
    memory,
    records::{self, Packed, Source},
    training::{self, Config, Trained},
    *,
};

pub const PROCESS_BYTES: u64 = 110_000_000_000;
pub const CORE_BYTES: u64 = 96_000_000_000;
pub const MIN_RUNTIME_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug)]
pub struct Budget {
    process_bytes: u64,
    core_bytes: u64,
    external_bytes: u128,
}
impl Budget {
    /// External storage includes every caller-owned candidate/interpreter,
    /// original source record, metadata, reader and serialization buffer that
    /// remains live. A zero reservation is only valid when none is present.
    pub fn new(process_bytes: u64, external_bytes: u128) -> Result<Self> {
        if !(MIN_RUNTIME_BYTES + 1..=PROCESS_BYTES).contains(&process_bytes) {
            return Err("invalid complete process storage cap".into());
        }
        let core_bytes = CORE_BYTES.min(process_bytes - MIN_RUNTIME_BYTES);
        memory::remaining(core_bytes, external_bytes)?;
        Ok(Self {
            process_bytes,
            core_bytes,
            external_bytes,
        })
    }
    pub fn process_bytes(&self) -> u64 {
        self.process_bytes
    }
    pub fn core_bytes(&self) -> u64 {
        self.core_bytes
    }
    pub fn runtime_bytes(&self) -> u64 {
        self.process_bytes - self.core_bytes
    }
    pub fn external_bytes(&self) -> u128 {
        self.external_bytes
    }
    /// Subtract all additional live owned storage before a local constructor.
    /// A caller cannot charge only logical lengths where capacity is larger.
    pub fn remaining(&self, live_bytes: u128) -> Result<u64> {
        memory::remaining(
            self.core_bytes,
            self.external_bytes
                .checked_add(live_bytes)
                .ok_or("complete storage accounting overflow")?,
        )
    }
    /// This forecasts inputs only; it does not admit the model or its update.
    /// Subsequent constructors enforce actual shared or arbitrary fallback
    /// capacities and all simultaneous old/new buffers.
    pub fn after_input_forecast(
        &self,
        shape: Shape,
        records: usize,
        unknown: usize,
    ) -> Result<u64> {
        shape.validate()?;
        self.remaining(
            domain::dictionary_bytes(shape, shape.rows(), unknown)?
                + records::packed_bytes(shape, records)?,
        )
    }
    pub fn pack<F>(
        &self,
        shape: Shape,
        labels: &[usize],
        state: F,
        work: &mut Work,
    ) -> Result<Packed>
    where
        F: FnMut(usize, usize, &mut Work) -> Result<usize>,
    {
        Packed::from_columns(shape, labels, state, self.remaining(0)?, work)
    }
    pub fn dictionary(
        &self,
        records: &Packed,
        minimum: f64,
        maximum: f64,
        work: &mut Work,
    ) -> Result<Domain> {
        domain::from_source(
            records.shape(),
            Source::Packed(records),
            minimum,
            maximum,
            self.remaining(records.allocation_bytes())?,
            work,
        )
    }
    pub fn model_room(&self, domain: &Domain, records: &Packed) -> Result<u64> {
        Source::Packed(records).check_shape(domain.shape)?;
        self.remaining(domain::allocation_bytes(domain) + records.allocation_bytes())
    }
    pub fn train(
        &self,
        domain: &Domain,
        records: &Packed,
        config: Config,
        work: &mut Work,
    ) -> Result<Trained> {
        training::train_source(
            domain,
            Source::Packed(records),
            config,
            self.model_room(domain, records)?,
            work,
        )
    }
    pub fn train_from(
        &self,
        domain: &Domain,
        records: &Packed,
        config: Config,
        initial: Parameters,
        work: &mut Work,
    ) -> Result<Trained> {
        // The local trainer owns initial and charges it before any allocation.
        training::train_from_source(
            domain,
            Source::Packed(records),
            config,
            initial,
            self.model_room(domain, records)?,
            work,
        )
    }
}

#[cfg(test)]
mod tests;
