//! Bounded, create-new artifact writes. Admission is repeated every64MiB and
//! after finishing; partial evidence remains if an I/O or resource stop occurs.
use crate::{
    resources::{Guard, ResourceStop},
    *,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Read, Write},
    path::Path,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    pub bytes: u64,
    pub sha256: String,
}
fn hex(bytes: impl IntoIterator<Item = u8>) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(64);
    for b in bytes {
        write!(s, "{b:02x}").unwrap();
    }
    s
}
pub fn hash(path: &Path) -> Result<Artifact> {
    let mut file = File::open(path)?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut hash = Sha256::new();
    let mut bytes = 0;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        bytes += n as u64;
    }
    Ok(Artifact {
        bytes,
        sha256: hex(hash.finalize()),
    })
}
pub struct Writer<'a> {
    file: BufWriter<File>,
    guard: &'a Guard,
    limit: u64,
    written: u64,
    next_check: u64,
    hash: Sha256,
}
impl<'a> Writer<'a> {
    pub fn new(path: &Path, limit: u64, guard: &'a Guard) -> Result<Self> {
        guard.contains(path)?;
        guard.check()?;
        Ok(Self {
            file: BufWriter::with_capacity(
                1024 * 1024,
                OpenOptions::new().write(true).create_new(true).open(path)?,
            ),
            guard,
            limit,
            written: 0,
            next_check: 64 * 1024 * 1024,
            hash: Sha256::new(),
        })
    }
    pub fn finish(mut self) -> Result<Artifact> {
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
        self.guard.check()?;
        Ok(Artifact {
            bytes: self.written,
            sha256: hex(self.hash.finalize()),
        })
    }
}
impl Write for Writer<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() as u64 > self.limit - self.written {
            return Err(std::io::Error::other(ResourceStop(
                "artifact exceeds its admitted byte limit".into(),
            )));
        }
        let n = self.file.write(bytes)?;
        self.hash.update(&bytes[..n]);
        self.written += n as u64;
        if self.written >= self.next_check {
            self.file.flush()?;
            self.guard
                .check()
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            self.next_check = self.written + 64 * 1024 * 1024;
        }
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}
pub fn json<T: Serialize>(path: &Path, value: &T, limit: u64, guard: &Guard) -> Result<Artifact> {
    let mut out = Writer::new(path, limit, guard)?;
    serde_json::to_writer(&mut out, value)?;
    out.write_all(b"\n")?;
    out.finish()
}
pub fn copy(input: &Path, output: &Path, expected: &Artifact, guard: &Guard) -> Result<Artifact> {
    if hash(input)? != *expected {
        return Err("input artifact hash changed".into());
    }
    let mut reader = File::open(input)?;
    let mut writer = Writer::new(output, expected.bytes, guard)?;
    std::io::copy(&mut reader, &mut writer)?;
    let actual = writer.finish()?;
    if actual != *expected {
        return Err("copied artifact differs".into());
    }
    Ok(actual)
}
