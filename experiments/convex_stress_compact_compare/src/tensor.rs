//! Port of reduced_compare at f5015cec, with complete CM70 local admission.
//! The caller subtracts original inputs/metadata before passing this limit.
//! Read admission includes the additional stored dictionary until comparison;
//! head construction includes retained parameters, I/O and class audit scratch.
use crate::*;
use joint::{Parameters, Shape, Work, domain::Domain, head::Head, persistence};
use std::{fs::File, io::BufReader, path::Path};

#[derive(Clone, Debug)]
pub struct Saved {
    pub artifact: disk::Artifact,
    pub body_sha256: [u8; 32],
}
pub fn length(shape: Shape, unknown: usize) -> Result<u64> {
    shape.validate()?;
    if unknown > shape.rows() {
        return Err("unknown rows exceed the complete model dictionary".into());
    }
    let parameters = shape.programs as u128 + unknown as u128 * shape.classes as u128;
    Ok(u64::try_from(
        88 + 4 * shape.rows() as u128 + 8 * parameters,
    )?)
}
pub fn save(
    path: &Path,
    domain: &Domain,
    parameters: &Parameters,
    head: &Head,
    guard: &resources::Guard,
    limit: u64,
    work: &mut Work,
) -> Result<Saved> {
    guard.check()?;
    // Inputs remain caller-owned. Writer and audit scratch belong here.
    crate::memory::remaining(limit, scratch(domain.shape))?;
    head.audit(domain, parameters, work)?;
    let length = length(domain.shape, domain.unknown.len())?;
    if length != persistence::bytes(domain) {
        return Err("complete checkpoint forecast differs from stream codec".into());
    }
    let mut writer = disk::Writer::new(path, length, guard)?;
    let body_sha256 = persistence::write(domain, parameters, &mut writer, work)?;
    let artifact = writer.finish()?;
    if artifact.bytes != length {
        return Err("incomplete reduced model write".into());
    }
    Ok(Saved {
        artifact,
        body_sha256,
    })
}
pub fn load(
    path: &Path,
    domain: &Domain,
    expected: &disk::Artifact,
    body_sha256: [u8; 32],
    limit: u64,
    work: &mut Work,
) -> Result<(Parameters, Head)> {
    let local = crate::memory::remaining(limit, scratch(domain.shape))?;
    joint::domain::validate(domain, work)?;
    if expected.bytes != length(domain.shape, domain.unknown.len())?
        || disk::hash(path)? != *expected
    {
        return Err("complete reduced checkpoint length or file hash changed".into());
    }
    let (stored, parameters) = persistence::read(
        BufReader::new(File::open(path)?),
        body_sha256,
        expected.bytes,
        local,
        work,
    )?;
    if stored != *domain {
        return Err("checkpoint dictionary differs from original observed records".into());
    }
    drop(stored);
    let head = Head::build(
        domain,
        &parameters,
        crate::memory::remaining(local, parameters.allocation_bytes() as u128)?,
        work,
    )?;
    head.audit(domain, &parameters, work)?;
    Ok((parameters, head))
}

/// One 1MiB hash/writer buffer, a bounded reader, classwise spectral audit,
/// hash/path/container metadata. Hashing precedes reading; reserve both anyway.
fn scratch(shape: Shape) -> u128 {
    2 * 1024 * 1024 + 64 * shape.classes as u128
}
