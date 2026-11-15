//! Exact little-endian binary64 persistence of masses, gamma and every real/
//! imaginary head entry. No quantization, sparse coefficient omission or JSON
//! rounding is used. Length/shape/hash checks precede model acceptance.
use crate::{
    disk::{Artifact, Writer},
    resources::Guard,
    *,
};
use joint::{Parameters, Shape, Work, head::Head};
use std::{
    fs::File,
    io::{BufReader, Read, Write},
    path::Path,
};
const MAGIC: &[u8; 8] = b"BLTC6401";
pub fn length(shape: Shape) -> Result<u64> {
    shape.admit(0)?;
    Ok(u64::try_from(
        32 + 8 * (shape.programs as u128 + 3 * shape.cells() as u128),
    )?)
}
pub fn save(
    path: &Path,
    q: &Parameters,
    head: &Head,
    guard: &Guard,
    work: &mut Work,
) -> Result<Artifact> {
    let shape = head.shape;
    q.validate(shape)?;
    head.validate()?;
    work.charge(32 * (shape.parameters() as u128 + 2 * shape.cells() as u128))?;
    let mut writer = Writer::new(path, length(shape)?, guard)?;
    writer.write_all(MAGIC)?;
    for n in [shape.programs, shape.states, shape.classes] {
        writer.write_all(&(n as u64).to_le_bytes())?;
    }
    let mut buffer = Vec::with_capacity(65536);
    for values in [&q.values, &head.real, &head.imaginary] {
        for chunk in values.chunks(8192) {
            buffer.clear();
            for x in chunk {
                buffer.extend_from_slice(&x.to_le_bytes());
            }
            writer.write_all(&buffer)?;
        }
    }
    let artifact = writer.finish()?;
    if artifact.bytes != length(shape)? {
        return Err("complete tensor output length differs".into());
    }
    Ok(artifact)
}
fn array(reader: &mut impl Read, len: usize) -> Result<Vec<f64>> {
    let mut result = Vec::with_capacity(len);
    let mut buffer = [0u8; 65536];
    while result.len() < len {
        let count = (len - result.len()).min(8192);
        reader.read_exact(&mut buffer[..count * 8])?;
        for bytes in buffer[..count * 8].as_chunks::<8>().0 {
            result.push(f64::from_le_bytes(*bytes));
        }
    }
    Ok(result)
}
pub fn load(
    path: &Path,
    shape: Shape,
    expected: &Artifact,
    work: &mut Work,
) -> Result<(Parameters, Head)> {
    if expected.bytes != length(shape)? || File::open(path)?.metadata()?.len() != expected.bytes {
        return Err("complete tensor length or shape differs".into());
    }
    if crate::disk::hash(path)? != *expected {
        return Err("complete tensor hash differs".into());
    }
    work.charge(32 * (shape.parameters() as u128 + 2 * shape.cells() as u128))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err("unsupported complete tensor encoding".into());
    }
    for n in [shape.programs, shape.states, shape.classes] {
        let mut bytes = [0u8; 8];
        reader.read_exact(&mut bytes)?;
        if u64::from_le_bytes(bytes) != n as u64 {
            return Err("complete tensor header differs".into());
        }
    }
    let parameters = Parameters {
        values: array(&mut reader, shape.parameters())?,
    };
    let head = Head {
        shape,
        real: array(&mut reader, shape.cells())?,
        imaginary: array(&mut reader, shape.cells())?,
    };
    parameters.validate(shape)?;
    head.validate()?;
    let rebuilt = Head::build(shape, &parameters, work)?;
    if head
        .real
        .iter()
        .zip(&rebuilt.real)
        .chain(head.imaginary.iter().zip(&rebuilt.imaginary))
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err("saved complete head does not match its learned joint tensor".into());
    }
    Ok((parameters, head))
}
