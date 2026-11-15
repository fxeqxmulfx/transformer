//! Bounded JSON reads, buffered immutable writes and streaming SHA256. Large
//! retained reports and raw vectors are never duplicated just for hashing.
use crate::*;
use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

pub fn write(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut w = BufWriter::with_capacity(
        1024 * 1024,
        OpenOptions::new().write(true).create_new(true).open(path)?,
    );
    serde_json::to_writer(&mut w, value)?;
    w.write_all(b"\n")?;
    w.flush()?;
    w.get_ref().sync_all()?;
    Ok(())
}
pub fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::metadata(path)?.len() as u128;
    if bytes > 1024u128.pow(3) || 64 * bytes + 2 * 1024u128.pow(3) > joint::MAX_BYTES {
        return Err("JSON parse buffer forecast refused".into());
    }
    Ok(serde_json::from_reader(BufReader::with_capacity(
        1024 * 1024,
        fs::File::open(path)?,
    ))?)
}
pub fn digest(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
struct HashWriter(Sha256);
impl Write for HashWriter {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.0.update(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
pub fn canonical_digest(value: &impl Serialize) -> Result<String> {
    let mut w = HashWriter(Sha256::new());
    serde_json::to_writer(&mut w, value)?;
    w.write_all(b"\n")?;
    Ok(w.0.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
