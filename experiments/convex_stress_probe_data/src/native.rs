//! Bounded binary transport from the original native CPU tensor operations.
//! This module supplies data only and cannot be used as a learned predictor.
use crate::Result;
use std::{
    io::Read,
    path::Path,
    process::{Command, Stdio},
};
const LIMIT: u64 = 32 * 1024 * 1024;
pub struct Shard {
    pub inputs: Vec<Vec<i64>>,
    pub labels: Vec<Vec<i64>>,
    pub teacher: Option<Vec<Vec<u64>>>,
    pub vocabulary: usize,
}
fn take<'a>(bytes: &mut &'a [u8], n: usize) -> Result<&'a [u8]> {
    let value = bytes.get(..n).ok_or("truncated native data")?;
    *bytes = &bytes[n..];
    Ok(value)
}
fn word(bytes: &mut &[u8]) -> Result<u64> {
    Ok(u64::from_le_bytes(take(bytes, 8)?.try_into()?))
}
pub fn parse(mut bytes: &[u8]) -> Result<Shard> {
    if bytes.len() as u64 > LIMIT || take(&mut bytes, 8)? != b"GPTPRB01" {
        return Err("invalid native probe header or size".into());
    }
    let n = usize::try_from(word(&mut bytes)?)?;
    let width = usize::try_from(word(&mut bytes)?)?;
    let outputs = usize::try_from(word(&mut bytes)?)?;
    let teachers = usize::try_from(word(&mut bytes)?)?;
    let vocabulary = usize::try_from(word(&mut bytes)?)?;
    if !(1..=2048).contains(&n)
        || !(1..=160).contains(&width)
        || !(1..=32).contains(&outputs)
        || !(1..=293).contains(&vocabulary)
        || (teachers != 0 && teachers != n * outputs)
        || bytes.len() != 8 * (n * (width + outputs) + teachers)
    {
        return Err("native complete shard dimensions differ".into());
    }
    let mut rows = |width: usize| -> Result<Vec<Vec<i64>>> {
        (0..n)
            .map(|_| (0..width).map(|_| Ok(word(&mut bytes)? as i64)).collect())
            .collect()
    };
    let inputs = rows(width)?;
    let labels = rows(outputs)?;
    let teacher = if teachers == 0 {
        None
    } else {
        Some(
            (0..n)
                .map(|_| (0..outputs).map(|_| word(&mut bytes)).collect())
                .collect::<Result<_>>()?,
        )
    };
    if inputs
        .iter()
        .chain(&labels)
        .flatten()
        .any(|&x| x < 0 || x as usize >= vocabulary)
        || !bytes.is_empty()
    {
        return Err("native output is out of vocabulary or has trailing bytes".into());
    }
    Ok(Shard {
        inputs,
        labels,
        teacher,
        vocabulary,
    })
}
pub fn generate(helper: &Path, name: &str, split: &str) -> Result<Shard> {
    let mut child = Command::new(helper)
        .args([name, split])
        // Declare one reproducible native kernel family; no unsupported AVX512.
        .env("ATEN_CPU_CAPABILITY", "avx2")
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let mut bytes = vec![];
    child
        .stdout
        .take()
        .ok_or("native data stdout")?
        .take(LIMIT + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > LIMIT {
        child.kill()?;
        child.wait()?;
        return Err("native data exceeds declared output limit".into());
    }
    if !child.wait()?.success() {
        return Err("native original data construction failed".into());
    }
    parse(&bytes)
}
