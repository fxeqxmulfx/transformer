//! The materializing boundary rechecks the complete candidate partition and
//! exact streamed audits, then admits all records and every unknown row.
use crate::{
    family::{Evaluator, programs},
    selection::{Selection, hash_header},
    *,
};
use joint::{Record, domain::Domain};
use sha2::{Digest, Sha256};

pub fn encode(
    training: &Training,
    selected: &Selection,
    work: &mut Work,
) -> Result<(Domain, Vec<Record>)> {
    if *selected != selection::select(training, work)? {
        return Err("wide selection omits or changes an original candidate or observation".into());
    }
    let expected = selected
        .audit
        .complete_buffer_forecast
        .ok_or("wide full model was not admitted")?;
    let shape = selected.shape();
    if shape.admit(
        training.rows.len(),
        selected.audit.unknown_rows,
        selected.audit.head_columns_bound,
    )? != expected
    {
        return Err("wide reduced-buffer admission differs".into());
    }
    let family = programs();
    let evaluator = Evaluator::new(training.vocabulary)?;
    work.charge(16 * shape.programs as u128 * training.rows.len() as u128)?;
    let mut records: Vec<_> = training
        .rows
        .iter()
        .map(|x| Record {
            states: Vec::with_capacity(shape.programs),
            label: x[2] as usize,
        })
        .collect();
    let mut encoded = Sha256::new();
    for &id in &selected.retained {
        hash_header(&mut encoded, id, training.rows.len());
        for (row, &[a, b, y]) in records.iter_mut().zip(&training.rows) {
            work.charge(512 + evaluator.reference_cost(family[id]))?;
            let state = evaluator.evaluate(family[id], a as usize, b as usize);
            if evaluator.reference(family[id], a as usize, b as usize) != state
                || row.label != y as usize
            {
                return Err("materialized wide interpretation differs".into());
            }
            encoded.update((state as u32).to_le_bytes());
            row.states.push(state);
        }
    }
    let domain = Domain::new(shape, &records, 1., 32., work)?;
    work.charge(1024 * shape.rows() as u128)?;
    let mut dictionary = Sha256::new();
    for (p, &id) in selected.retained.iter().enumerate() {
        hash_header(&mut dictionary, id, shape.states);
        for label in &domain.labels[p * shape.states..(p + 1) * shape.states] {
            dictionary.update(label.map_or(u32::MAX, |x| x as u32).to_le_bytes());
        }
    }
    if hex(&encoded.finalize()) != selected.audit.program_major_states_sha256
        || hex(&dictionary.finalize()) != selected.audit.dictionary_sha256
        || domain.unknown.len() != selected.audit.unknown_rows
        || domain.known.iter().sum::<usize>() != selected.audit.known_rows
        || domain.head_columns_bound != selected.audit.head_columns_bound
        || domain.admit(records.len())? != expected
    {
        return Err("streamed and materialized complete domain audits differ".into());
    }
    Ok((domain, records))
}

/// CM68 packed observations preserve the original probe's complete audit.
/// Its legacy memory refusal stays intact; packed admission is computed separately.
pub fn encode_packed(
    training: &Training,
    selected: &Selection,
    work: &mut Work,
) -> Result<(Domain, joint::records::Packed)> {
    use joint::records::{Packed, Source};
    if *selected != selection::select(training, work)? {
        return Err(
            "packed selection omits or changes an original candidate or observation".into(),
        );
    }
    let shape = selected.shape();
    let expected = shape.admit_packed(
        training.rows.len(),
        selected.audit.unknown_rows,
        selected.audit.head_columns_bound,
    )?;
    let family = programs();
    let evaluator = Evaluator::new(training.vocabulary)?;
    let labels: Vec<_> = training.rows.iter().map(|r| r[2] as usize).collect();
    let records = Packed::from_columns(
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
                return Err("packed independent state interpretation differs".into());
            }
            Ok(state)
        },
        work,
    )?;
    // Hash the actual stored words after decoding them. A hash of the values
    // before storage would fail to audit the materialized representation.
    work.charge(16 * shape.programs as u128 * training.rows.len() as u128)?;
    let mut encoded = Sha256::new();
    for (p, &id) in selected.retained.iter().enumerate() {
        hash_header(&mut encoded, id, training.rows.len());
        for (row, original) in training.rows.iter().enumerate() {
            if records.label(row)? != original[2] as usize {
                return Err("packed original raw class changed".into());
            }
            encoded.update((records.state(row, p)? as u32).to_le_bytes());
        }
    }
    let domain = Domain::from_source(shape, Source::Packed(&records), 1., 32., work)?;
    work.charge(1024 * shape.rows() as u128)?;
    let mut dictionary = Sha256::new();
    for (p, &id) in selected.retained.iter().enumerate() {
        hash_header(&mut dictionary, id, shape.states);
        for label in &domain.labels[p * shape.states..(p + 1) * shape.states] {
            dictionary.update(label.map_or(u32::MAX, |x| x as u32).to_le_bytes());
        }
    }
    if hex(&encoded.finalize()) != selected.audit.program_major_states_sha256
        || hex(&dictionary.finalize()) != selected.audit.dictionary_sha256
        || domain.unknown.len() != selected.audit.unknown_rows
        || domain.known.iter().sum::<usize>() != selected.audit.known_rows
        || domain.head_columns_bound != selected.audit.head_columns_bound
        || Source::Packed(&records).admit(shape, domain.unknown.len(), domain.head_columns_bound)?
            != expected
    {
        return Err("packed materialization differs from the complete frozen probe audit".into());
    }
    Ok((domain, records))
}
