//! Streaming artifact bounds, ported from paths_compare at92343371.
use crate::*;
use sha2::{Digest, Sha256};
use std::io::{BufReader, BufWriter, Read, Write};
const LIMIT: u64 = 8 * 1024 * 1024 * 1024;
pub fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
pub fn digest(path: &Path) -> Result<String> {
    digest_limit(path, LIMIT)
}
pub(crate) fn digest_limit(path: &Path, limit: u64) -> Result<String> {
    let mut file = fs::File::open(path)?;
    if limit > 16 * 1024 * 1024 * 1024 || file.metadata()?.len() > limit {
        return Err("artifact exceeds its file limit".into());
    }
    let mut hash = Sha256::new();
    let (mut buffer, mut total) = ([0; 65536], 0u64);
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if total > limit {
            return Err("artifact grew past its file limit".into());
        }
        hash.update(&buffer[..n]);
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
pub fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = fs::File::open(path)?;
    if file.metadata()?.len() > LIMIT {
        return Err("JSON exceeds8GiB".into());
    }
    Ok(serde_json::from_reader(BufReader::new(
        file.take(LIMIT + 1),
    ))?)
}
struct Limited<W> {
    inner: W,
    bytes: u64,
}
impl<W: Write> Write for Limited<W> {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        if buffer.len() as u64 > LIMIT - self.bytes {
            return Err(std::io::Error::other("JSON output exceeds8GiB"));
        }
        let n = self.inner.write(buffer)?;
        self.bytes += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
pub fn write<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    let mut writer = Limited {
        inner: BufWriter::new(file),
        bytes: 0,
    };
    serde_json::to_writer(&mut writer, value)?;
    writer.flush()?;
    Ok(())
}
