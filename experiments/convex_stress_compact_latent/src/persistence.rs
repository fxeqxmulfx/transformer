//! CM70 lossless logical streaming using the exact RDLT6401 format of
//! reduced_latent at47db6ff5. Every original coordinate and byte remains.
//! Read admission includes its complete dictionary, compact parameters and
//! construction scratch; the reader's own storage belongs to the caller.
//! Guarded file writes retain the prior both-disk/repeat/overwrite rules.
use crate::domain::Domain;
use crate::{memory, *};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
const MAGIC: &[u8; 8] = b"RDLT6401";
#[cfg(test)]
mod tests;

pub fn write_file(
    domain: &Domain,
    q: &Parameters,
    workspace: &std::path::Path,
    storage: &std::path::Path,
    path: &std::path::Path,
    remaining_cohort_bytes: u64,
    work: &mut Work,
) -> Result<[u8; 32]> {
    domain::validate(domain, work)?;
    q.check(domain, 1e-8, work)?;
    let length = bytes(domain);
    let reserve =
        remaining_cohort_bytes.max(length.checked_mul(2).ok_or("checkpoint reserve overflow")?);
    let mut writer =
        legacy::resources::Guard::new(workspace, storage, reserve)?.writer(path, length)?;
    let hash = write(domain, q, &mut writer, work)?;
    writer.finish()?;
    Ok(hash)
}

pub fn bytes(domain: &Domain) -> u64 {
    56 + 4 * domain.shape.rows() as u64 + 8 * domain.parameters() as u64 + 32
}
fn emit<W: Write>(writer: &mut W, hash: &mut Sha256, bytes: &[u8]) -> Result<()> {
    writer.write_all(bytes)?;
    hash.update(bytes);
    Ok(())
}
pub fn write<W: Write>(
    domain: &Domain,
    q: &Parameters,
    mut writer: W,
    work: &mut Work,
) -> Result<[u8; 32]> {
    domain::validate(domain, work)?;
    q.check(domain, 1e-8, work)?;
    work.charge(64 * bytes(domain) as u128)?;
    let mut hash = Sha256::new();
    emit(&mut writer, &mut hash, MAGIC)?;
    for n in [
        domain.shape.programs,
        domain.shape.states,
        domain.shape.classes,
        domain.unknown.len(),
    ] {
        emit(&mut writer, &mut hash, &(n as u64).to_le_bytes())?;
    }
    for x in [domain.minimum_mass, domain.maximum_mass] {
        emit(&mut writer, &mut hash, &x.to_le_bytes())?;
    }
    for label in &domain.labels {
        emit(
            &mut writer,
            &mut hash,
            &label.map_or(u32::MAX, |x| x as u32).to_le_bytes(),
        )?;
    }
    for x in q.values() {
        emit(&mut writer, &mut hash, &x.to_le_bytes())?;
    }
    let digest: [u8; 32] = hash.finalize().into();
    writer.write_all(&digest)?;
    writer.flush()?;
    Ok(digest)
}
fn read_array<R: Read, const N: usize>(reader: &mut R, hash: &mut Sha256) -> Result<[u8; N]> {
    let mut bytes = [0; N];
    reader.read_exact(&mut bytes)?;
    hash.update(bytes);
    Ok(bytes)
}
pub fn read<R: Read>(
    mut reader: R,
    expected: [u8; 32],
    length: u64,
    limit: u64,
    work: &mut Work,
) -> Result<(Domain, Parameters)> {
    if length < 88 || length as u128 > legacy::MAX_BYTES {
        return Err("invalid reduced checkpoint length".into());
    }
    work.charge(64 * length as u128)?;
    let mut hash = Sha256::new();
    if &read_array::<_, 8>(&mut reader, &mut hash)? != MAGIC {
        return Err("invalid reduced checkpoint version".into());
    }
    let mut dims = [0usize; 4];
    for d in &mut dims {
        *d = usize::try_from(u64::from_le_bytes(read_array(&mut reader, &mut hash)?))?;
    }
    let sh = Shape {
        programs: dims[0],
        states: dims[1],
        classes: dims[2],
    };
    sh.validate()?;
    let dictionary_bytes = domain::dictionary_bytes(sh, sh.rows(), dims[3])?;
    // Count every dictionary vector and mass before reading or allocating
    // them; the row builder charges all class scratch and fallback growth.
    let row_room = memory::remaining(
        limit,
        dictionary_bytes + memory::vector_bytes::<f64>(sh.programs) + 4096,
    )?;
    let count = sh.programs + dims[3] * sh.classes;
    if 56 + 4 * sh.rows() as u64 + 8 * count as u64 + 32 != length {
        return Err("checkpoint dimensions do not match file length".into());
    }
    let minimum = f64::from_le_bytes(read_array(&mut reader, &mut hash)?);
    let maximum = f64::from_le_bytes(read_array(&mut reader, &mut hash)?);
    let mut labels = memory::vector(sh.rows(), limit)?;
    for _ in 0..sh.rows() {
        let y = u32::from_le_bytes(read_array(&mut reader, &mut hash)?);
        labels.push((y != u32::MAX).then_some(y as usize));
    }
    if labels.iter().filter(|label| label.is_none()).count() != dims[3] {
        return Err("unknown-row count differs from dictionary before allocation".into());
    }
    let domain = domain::from_labels(sh, labels, minimum, maximum, limit, work)?;
    if domain.unknown.len() != dims[3] {
        return Err("unknown-row count differs from dictionary".into());
    }
    let actual_dictionary = domain::allocation_bytes(&domain);
    if actual_dictionary != dictionary_bytes {
        return Err("checkpoint dictionary capacities differ from admission".into());
    }
    let mut masses = memory::vector(sh.programs, limit)?;
    for _ in 0..sh.programs {
        masses.push(f64::from_le_bytes(read_array(&mut reader, &mut hash)?));
    }
    let mut values = rows::Builder::new(sh.classes, dims[3], row_room)?;
    let mut row = memory::vector(sh.classes, row_room)?;
    for _ in 0..dims[3] {
        row.clear();
        for _ in 0..sh.classes {
            row.push(f64::from_le_bytes(read_array(&mut reader, &mut hash)?));
        }
        values.push(&row)?;
    }
    let digest: [u8; 32] = hash.finalize().into();
    let mut footer = [0; 32];
    reader.read_exact(&mut footer)?;
    let mut extra = [0; 1];
    if digest != expected || footer != expected || reader.read(&mut extra)? != 0 {
        return Err("checkpoint hash or exact length differs from its pinned value".into());
    }
    let q = Parameters::from_parts(
        &domain,
        masses,
        values.finish()?,
        memory::remaining(limit, dictionary_bytes + 4096)?,
    )?;
    q.check(&domain, 1e-8, work)?;
    Ok((domain, q))
}
