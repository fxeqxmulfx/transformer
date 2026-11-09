//! Immutable source ledger and little-endian evidence for task #202.
use crate::Result;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
pub fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
pub fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
pub fn word(bytes: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap())
}
pub fn scalar(bytes: &[u8], i: usize) -> f64 {
    f64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap())
}
pub fn write(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec(value)?)?;
    Ok(())
}
#[derive(Default)]
pub struct Ledger(pub BTreeMap<PathBuf, String>);
impl Ledger {
    pub fn read(&mut self, path: &Path) -> Result<Vec<u8>> {
        let path = path.canonicalize()?;
        let bytes = fs::read(&path)?;
        let digest = hash(&bytes);
        if let Some(old) = self.0.insert(path, digest.clone()) {
            require(old == digest, "source changed during repeated read")?;
        }
        Ok(bytes)
    }
    pub fn json(&mut self, path: &Path) -> Result<Value> {
        Ok(serde_json::from_slice(&self.read(path)?)?)
    }
    pub fn verify(&self) -> Result<()> {
        for (path, digest) in &self.0 {
            require(
                hash(&fs::read(path)?) == *digest,
                "source changed during audit",
            )?;
        }
        Ok(())
    }
}
