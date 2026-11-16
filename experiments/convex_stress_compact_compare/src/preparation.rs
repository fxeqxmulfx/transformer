//! Full input materialization from wide_latents encoding at e489dbf8, using
//! CM68 words and CM70 capacities. Verification and construction retain all
//! original work charges in two separately capped40-trillion-work passes.
//! The original largest probe alone charges38.7 trillion; its verification
//! and the subsequent construction cannot share one40-trillion allowance.
use crate::*;
use joint::{admission::Budget, domain::Domain, records::Packed};
use pairs::{
    Training,
    family::{Evaluator, programs},
    selection::Selection,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub(crate) mod memory;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Phases {
    pub selection: Work,
    pub materialization: Work,
}
impl Phases {
    pub fn new() -> Result<Self> {
        Ok(Self {
            selection: Work::new(pairs::MAX_WORK)?,
            materialization: Work::new(pairs::MAX_WORK)?,
        })
    }
    pub fn total(&self) -> Result<u64> {
        if [&self.selection, &self.materialization]
            .iter()
            .any(|w| w.limit != pairs::MAX_WORK || w.used > w.limit)
        {
            return Err("invalid complete preparation phase counters".into());
        }
        Ok(self
            .selection
            .used
            .checked_add(self.materialization.used)
            .ok_or("preparation work overflow")?)
    }
}
pub struct Prepared {
    pub selection: Selection,
    pub domain: Domain,
    pub records: Packed,
    pub budget: Budget,
}
fn header(hash: &mut Sha256, program: usize, size: usize) {
    hash.update((program as u64).to_le_bytes());
    hash.update((size as u64).to_le_bytes());
}
fn hex(value: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(64);
    for b in value {
        write!(out, "{b:02x}").unwrap();
    }
    out
}
pub fn prepare(
    training: &Training,
    expected: &Selection,
    process_bytes: u64,
    caller_bytes: u128,
    phases: &mut Phases,
) -> Result<Prepared> {
    if phases.total()? != 0 {
        return Err("preparation requires fresh bounded phase counters".into());
    }
    let caller = Budget::new(process_bytes, caller_bytes)?;
    let external =
        caller.external_bytes() + memory::training(training) + memory::selection(expected);
    let initial = Budget::new(process_bytes, external)?;
    initial.remaining(memory::SELECTION_SCRATCH)?;
    let selected = pairs::selection::select(training, &mut phases.selection)?;
    if selected != *expected {
        return Err(
            "compact selection omits or changes an original candidate or observation".into(),
        );
    }
    let external = external + memory::selection(&selected);
    let construction = Budget::new(process_bytes, external)?;
    construction.remaining(memory::SELECTION_SCRATCH)?;
    let shape = selected.shape();
    construction.after_input_forecast(shape, training.rows.len(), selected.audit.unknown_rows)?;
    let family = programs();
    let evaluator = Evaluator::new(training.vocabulary)?;
    let labels: Vec<_> = training.rows.iter().map(|r| r[2] as usize).collect();
    let building = Budget::new(
        process_bytes,
        external + memory::vector(&family) + memory::vector(&labels) + memory::INTERPRETER_SCRATCH,
    )?;
    let work = &mut phases.materialization;
    let records = building.pack(
        shape,
        &labels,
        |row, p, work| {
            let program = family[selected.retained[p]];
            let [a, b, y] = training.rows[row];
            work.charge(512 + evaluator.reference_cost(program))?;
            let state = evaluator.evaluate(program, a as usize, b as usize);
            if evaluator.reference(program, a as usize, b as usize) != state
                || labels[row] != y as usize
            {
                return Err("compact independent state interpretation differs".into());
            }
            Ok(state)
        },
        work,
    )?;
    work.charge(16 * shape.programs as u128 * training.rows.len() as u128)?;
    let mut encoded = Sha256::new();
    for (p, &id) in selected.retained.iter().enumerate() {
        header(&mut encoded, id, training.rows.len());
        for (row, original) in training.rows.iter().enumerate() {
            if records.label(row)? != original[2] as usize {
                return Err("compact original raw label changed".into());
            }
            encoded.update((records.state(row, p)? as u32).to_le_bytes());
        }
    }
    let domain = building.dictionary(&records, 1., 32., work)?;
    work.charge(1024 * shape.rows() as u128)?;
    let mut dictionary = Sha256::new();
    for (p, &id) in selected.retained.iter().enumerate() {
        header(&mut dictionary, id, shape.states);
        for label in &domain.labels[p * shape.states..(p + 1) * shape.states] {
            dictionary.update(label.map_or(u32::MAX, |x| x as u32).to_le_bytes());
        }
    }
    if hex(&encoded.finalize()) != selected.audit.program_major_states_sha256
        || hex(&dictionary.finalize()) != selected.audit.dictionary_sha256
        || domain.unknown.len() != selected.audit.unknown_rows
        || domain.known.iter().sum::<usize>() != selected.audit.known_rows
        || domain.head_columns_bound != selected.audit.head_columns_bound
    {
        return Err("compact materialization differs from the complete frozen probe audit".into());
    }
    // Family/label temporaries end here. The returned budget keeps both live
    // selections and the caller's original training capacity reserved.
    drop((family, labels, evaluator));
    let budget = Budget::new(process_bytes, external + memory::INTERPRETER_SCRATCH)?;
    budget.model_room(&domain, &records)?;
    phases.total()?;
    Ok(Prepared {
        selection: selected,
        domain,
        records,
        budget,
    })
}
