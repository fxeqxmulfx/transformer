use super::*;
use sha2::{Digest, Sha256};
use std::io::{BufReader, BufWriter, Read, Write};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Artifact {
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
}
pub(super) fn digest(path: &Path) -> Result<String> {
    if fs::metadata(path)?.len() > LIMIT {
        return Err("artifact exceeds8GiB".into());
    }
    let mut file = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        h.update(&buffer[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
pub(super) fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    if fs::metadata(path)?.len() > LIMIT {
        return Err("JSON artifact exceeds8GiB".into());
    }
    Ok(serde_json::from_reader(BufReader::new(fs::File::open(
        path,
    )?))?)
}
struct Limited<W> {
    inner: W,
    bytes: u64,
}
impl<W: Write> Write for Limited<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.len() as u64 > LIMIT - self.bytes {
            return Err(std::io::Error::other("serialized artifact exceeds8GiB"));
        }
        let n = self.inner.write(buf)?;
        self.bytes += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
pub(super) fn write<T: Serialize>(path: &Path, value: &T) -> Result<()> {
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
pub(super) fn record(dir: &Path, file: &str) -> Result<Artifact> {
    if Path::new(file).components().count() != 1 {
        return Err("artifact filename must be local".into());
    }
    let path = dir.join(file);
    Ok(Artifact {
        file: file.into(),
        bytes: fs::metadata(&path)?.len(),
        sha256: digest(&path)?,
    })
}
pub(super) fn verify(dir: &Path, saved: &Artifact) -> Result<()> {
    if record(dir, &saved.file)? != *saved {
        return Err("retained artifact hash or size differs".into());
    }
    Ok(())
}
