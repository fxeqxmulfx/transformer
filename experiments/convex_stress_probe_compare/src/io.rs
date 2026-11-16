//! Bounded immutable artifacts; binary64 checkpoints/traces stream without
//! whole-file allocation. Paths and full SHA256 bytes are part of the seal.
use crate::{
    Result,
    resources::{self, Guard},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Read, Write},
    path::Path,
};
pub const JSON_MAX: u64 = 32 * 1024 * 1024;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub bytes: u64,
    pub sha256: String,
}
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|x| format!("{x:02x}")).collect()
}
pub fn digest(text: &str) -> Result<[u8; 32]> {
    if text.len() != 64 || !text.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid digest".into());
    }
    let mut out = [0; 32];
    for (i, x) in out.iter_mut().enumerate() {
        *x = u8::from_str_radix(&text[2 * i..2 * i + 2], 16)?;
    }
    Ok(out)
}
pub fn artifact(path: &Path) -> Result<Artifact> {
    let meta = fs::symlink_metadata(path)?;
    if !meta.is_file() || meta.len() > resources::FILE_MAX {
        return Err("artifact is nonregular or oversized".into());
    }
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut count = 0u64;
    let mut buf = [0; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        count += n as u64;
        if count > resources::FILE_MAX {
            return Err("artifact grew beyond admission".into());
        }
        hash.update(&buf[..n]);
    }
    if count != meta.len() {
        return Err("artifact length changed while hashing".into());
    }
    Ok(Artifact {
        bytes: count,
        sha256: hex(&hash.finalize()),
    })
}
pub fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let meta = fs::symlink_metadata(path)?;
    if !meta.is_file() || meta.len() > JSON_MAX {
        return Err("invalid bounded JSON artifact".into());
    }
    let mut bytes = Vec::new();
    File::open(path)?
        .take(JSON_MAX + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > JSON_MAX {
        return Err("JSON grew beyond admission".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}
pub fn json(path: &Path, value: &impl Serialize, guard: &Guard) -> Result<Artifact> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    if bytes.len() as u64 > JSON_MAX {
        return Err("JSON exceeds admission".into());
    }
    let mut file = Writer::new(path, JSON_MAX, guard)?;
    file.write_all(&bytes)?;
    file.finish()
}
pub struct Writer<'a> {
    file: BufWriter<File>,
    hash: Sha256,
    bytes: u64,
    limit: u64,
    checked: u64,
    guard: &'a Guard,
}
impl<'a> Writer<'a> {
    pub fn new(path: &Path, limit: u64, guard: &'a Guard) -> Result<Self> {
        if limit == 0 || limit > resources::FILE_MAX {
            return Err("invalid binary artifact admission".into());
        }
        guard.check()?;
        Ok(Self {
            file: BufWriter::with_capacity(
                262144,
                OpenOptions::new().write(true).create_new(true).open(path)?,
            ),
            hash: Sha256::new(),
            bytes: 0,
            limit,
            checked: 0,
            guard,
        })
    }
    pub fn finish(mut self) -> Result<Artifact> {
        self.flush()?;
        self.file.get_ref().sync_all()?;
        self.guard.check()?;
        Ok(Artifact {
            bytes: self.bytes,
            sha256: hex(&self.hash.finalize()),
        })
    }
}
impl Write for Writer<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if (bytes.len() as u64)
            .checked_add(self.bytes)
            .is_none_or(|n| n > self.limit)
        {
            return Err(std::io::Error::other("artifact exceeds frozen allocation"));
        }
        if self.bytes - self.checked >= 4 * 1024 * 1024 {
            self.guard
                .check()
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            self.checked = self.bytes;
        }
        let n = self.file.write(bytes)?;
        self.hash.update(&bytes[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}
pub fn copy(source: &Path, target: &Path, guard: &Guard) -> Result<Artifact> {
    let expected = artifact(source)?;
    let mut input = File::open(source)?;
    let mut output = Writer::new(target, expected.bytes, guard)?;
    std::io::copy(&mut input, &mut output)?;
    let actual = output.finish()?;
    if actual != expected {
        return Err("input copy differs from original artifact".into());
    }
    Ok(actual)
}
