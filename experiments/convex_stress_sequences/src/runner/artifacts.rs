use crate::{Result, hash};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

pub fn write(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

pub fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    if fs::metadata(path)?.len() > 4 * 1024 * 1024 * 1024 {
        return Err("saved JSON exceeds4GiB serialization cap".into());
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

pub fn digest(path: &Path) -> Result<String> {
    Ok(hash(&fs::read(path)?))
}

pub fn same_json<T: Serialize>(a: &T, b: &T) -> Result<bool> {
    Ok(serde_json::to_vec(a)? == serde_json::to_vec(b)?)
}
