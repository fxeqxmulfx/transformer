//! CM28 certificates around immutable original inputs, models and complete parents.
use crate::adapter;
use convex_stress_modular::{Result, coding, search};
use convex_stress_relabel::{data, hash, metrics};
use serde_json::{Value, json};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};
mod evaluate;
mod fingerprint;
#[cfg(test)]
mod fixtures;
mod inputs;
mod inventory;
mod parent;
mod run;
mod summary;
#[cfg(test)]
mod tests;
pub use fingerprint::fingerprint;
use inputs::{Inputs, check_input, inputs};
use inventory::inventory;
pub use inventory::prepare;
pub use run::{replay, run};
pub use summary::summarize;
const PROTOCOL: &str = include_str!("../../protocol.json");
const SUITE: &str = include_str!("../../../../convex_stress/suite.json");
const LIMIT: u64 = 2 * 1024 * 1024 * 1024;
fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
fn tasks() -> Result<Vec<String>> {
    Ok(serde_json::from_value(protocol()?["tasks"].clone())?)
}
fn modes() -> Result<Vec<String>> {
    Ok(serde_json::from_value(
        protocol()?["reference_modes"].clone(),
    )?)
}
fn policy() -> Result<adapter::Policy> {
    let policy: adapter::Policy = serde_json::from_value(protocol()?["certificate"].clone())?;
    policy.validate()?;
    Ok(policy)
}
fn bytes(path: &Path) -> Result<Vec<u8>> {
    if fs::metadata(path)?.len() > LIMIT {
        return Err("identified cohort artifact exceeds2GiB".into());
    }
    Ok(fs::read(path)?)
}
fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&bytes(path)?)?)
}
fn write_raw(path: &Path, raw: &[u8]) -> Result<()> {
    if raw.len() as u64 > LIMIT {
        return Err("identified artifact serialization exceeds2GiB".into());
    }
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(raw)?;
    Ok(())
}
fn write<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    write_raw(path, &serde_json::to_vec(value)?)
}
fn binary_hash() -> Result<String> {
    Ok(hash(&bytes(&std::env::current_exe()?)?))
}
