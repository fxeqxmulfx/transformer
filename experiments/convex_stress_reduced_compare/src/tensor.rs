//! Bind every free coordinate and anchor dictionary, then reconstruct the full
//! complex head with the CM65 implementation's fixed binary64 arithmetic.
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
    work: &mut Work,
) -> Result<Saved> {
    guard.check()?;
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
    work: &mut Work,
) -> Result<(Parameters, Head)> {
    domain.admit(0)?;
    if expected.bytes != length(domain.shape, domain.unknown.len())?
        || disk::hash(path)? != *expected
    {
        return Err("complete reduced checkpoint length or file hash changed".into());
    }
    let (stored, parameters) = persistence::read(
        BufReader::new(File::open(path)?),
        body_sha256,
        expected.bytes,
        work,
    )?;
    if stored != *domain {
        return Err("checkpoint dictionary differs from original observed records".into());
    }
    drop(stored);
    let head = Head::build(domain, &parameters, work)?;
    head.audit(domain, &parameters, work)?;
    Ok((parameters, head))
}
