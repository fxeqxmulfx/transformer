//! Bounded streaming hashes; complete parents remain byte-for-byte snapshots.
use crate::*;
use sha2::{Digest, Sha256};
use std::io::{BufReader, BufWriter, Read, Write};

const FILE_LIMIT: u64 = 16 * 1024 * 1024 * 1024;
const JSON_LIMIT: u64 = 8 * 1024 * 1024 * 1024;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}
pub fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
pub fn digest(path: &Path) -> Result<String> {
    digest_limit(path, FILE_LIMIT)
}
fn digest_limit(path: &Path, limit: u64) -> Result<String> {
    let mut file = fs::File::open(path)?;
    if file.metadata()?.len() > limit {
        return Err("artifact exceeds its file limit".into());
    }
    let mut hash = Sha256::new();
    let (mut buffer, mut total) = ([0; 65536], 0u64);
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        total = total.checked_add(n as u64).ok_or("file size overflow")?;
        if total > limit {
            return Err("artifact grew past its file limit".into());
        }
        hash.update(&buffer[..n]);
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
pub fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = fs::File::open(path)?;
    if file.metadata()?.len() > JSON_LIMIT {
        return Err("JSON exceeds8GiB".into());
    }
    Ok(serde_json::from_reader(BufReader::new(
        file.take(JSON_LIMIT + 1),
    ))?)
}
struct Limited<W> {
    inner: W,
    bytes: u64,
}
impl<W: Write> Write for Limited<W> {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        if buffer.len() as u64 > JSON_LIMIT - self.bytes {
            return Err(std::io::Error::other("serialized output exceeds8GiB"));
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
    let mut writer = Limited {
        bytes: 0,
        inner: BufWriter::new(
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)?,
        ),
    };
    serde_json::to_writer(&mut writer, value)?;
    writer.flush()?;
    Ok(())
}
pub fn local(name: &str) -> Result<&str> {
    if name.is_empty()
        || Path::new(name).components().count() != 1
        || !matches!(
            Path::new(name).components().next(),
            Some(std::path::Component::Normal(_))
        )
    {
        return Err("artifact name must be a local filename".into());
    }
    Ok(name)
}
#[derive(Default)]
pub struct Tracker(pub BTreeMap<PathBuf, Artifact>);
impl Tracker {
    pub fn bind(&mut self, path: &Path, expected: Option<&str>) -> Result<Artifact> {
        let path = fs::canonicalize(path)?;
        if !self.0.contains_key(&path) {
            let item = Artifact {
                bytes: fs::metadata(&path)?.len(),
                sha256: digest(&path)?,
                path: path.clone(),
            };
            self.0.insert(path.clone(), item);
        }
        let item = &self.0[&path];
        if expected.is_some_and(|s| s != item.sha256) {
            return Err(format!("artifact hash differs: {}", path.display()).into());
        }
        Ok(item.clone())
    }
    pub fn load(&mut self, path: &Path) -> Result<Value> {
        self.bind(path, None)?;
        read(path)
    }
}
pub fn verify(item: &Artifact) -> Result<()> {
    if fs::metadata(&item.path)?.len() != item.bytes || digest(&item.path)? != item.sha256 {
        return Err(format!("sealed artifact changed: {}", item.path.display()).into());
    }
    Ok(())
}
pub fn snapshot(source: &Artifact, destination: &Path) -> Result<()> {
    let mut input = fs::File::open(&source.path)?;
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    if std::io::copy(&mut input, &mut output)? != source.bytes
        || digest(destination)? != source.sha256
    {
        return Err("complete parent changed while copying".into());
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_limits_and_local_names_are_checked_before_admission() -> Result<()> {
        let temp = crate::fixtures::Temp::new()?;
        let path = temp.0.join("large");
        fs::write(&path, vec![7; 131073])?;
        assert_eq!(digest_limit(&path, 131073)?, hash(&vec![7; 131073]));
        assert!(digest_limit(&path, 131072).is_err());
        for name in ["", ".", "..", "a/b", "/a"] {
            assert!(local(name).is_err());
        }
        let item = Tracker::default().bind(&path, None)?;
        fs::write(&path, b"changed")?;
        assert!(verify(&item).is_err());
        Ok(())
    }
}
