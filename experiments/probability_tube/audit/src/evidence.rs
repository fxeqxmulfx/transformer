//! Explicit file ledger; the constructor's call sites permit only old inputs.
use crate::Result;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
pub fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
pub fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
pub fn word(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}
pub fn write(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec(value)?)?;
    Ok(())
}
#[derive(Default)]
pub struct Ledger(pub Vec<Value>);
impl Ledger {
    pub fn read(&mut self, path: &Path) -> Result<Vec<u8>> {
        let path = path.canonicalize()?;
        let bytes = fs::read(&path)?;
        self.0.push(json!({"path":path,"sha256":hash(&bytes)}));
        Ok(bytes)
    }
    pub fn json(&mut self, path: &Path) -> Result<Value> {
        Ok(serde_json::from_slice(&self.read(path)?)?)
    }
    pub fn verify(&self) -> Result<()> {
        for item in &self.0 {
            require(
                hash(&fs::read(item["path"].as_str().ok_or("ledger path")?)?) == item["sha256"],
                "input changed",
            )?;
        }
        Ok(())
    }
}
