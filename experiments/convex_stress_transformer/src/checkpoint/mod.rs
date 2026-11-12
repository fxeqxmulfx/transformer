//! Immutable streamed f32 checkpoints for the bounded native transformer.
//! Extends backend 8eaf3a24 and the checkpoint/RNG practice in
//! convex_recall_compare/src/main.rs at a74ce44b. Unlike that earlier f64 copy,
//! this format streams native tensor bits and verifies bounds and a checksum.
//! Parameters, both Adam moments and the complete sampling RNG are preserved.
//! A checksum covers dimensions, optimizer constants, provenance binding and
//! progress as well as tensor bits. Allocation guards run before tensor reads.
use crate::{Result, engine::Transformer, mt::Mt};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};
mod header;
use header::Header;
pub use header::Progress;
#[cfg(test)]
mod tests;
const MAGIC: &[u8; 8] = b"CSTRANS1";
const HEADER_LIMIT: u64 = 64 * 1024;

fn digest_text(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}
fn write_hashed(w: &mut impl Write, sha: &mut Sha256, bytes: &[u8]) -> Result<()> {
    w.write_all(bytes)?;
    sha.update(bytes);
    Ok(())
}
fn read_hashed<const N: usize>(r: &mut impl Read, sha: &mut Sha256) -> Result<[u8; N]> {
    let mut bytes = [0; N];
    r.read_exact(&mut bytes)?;
    sha.update(bytes);
    Ok(bytes)
}
fn arrays_valid(net: &Transformer) -> bool {
    [&net.p, &net.m, &net.v]
        .iter()
        .all(|a| a.len() == net.dims.len() && a.iter().all(|x| x.is_finite()))
        && net.v.iter().all(|x| *x >= 0.)
}

/// Stream a new immutable checkpoint. An existing destination or partial file
/// is an error. Failed writes leave their partial file for explicit recovery;
/// publication uses a hard link so an existing destination cannot be replaced.
pub fn save_new(
    path: &Path,
    net: &Transformer,
    rng: &Mt,
    progress: Progress,
    binding: &str,
) -> Result<String> {
    let header = Header::new(net, rng, progress, binding)?;
    if !arrays_valid(net) {
        return Err("invalid checkpoint parameter/moment arrays".into());
    }
    let bytes = serde_json::to_vec(&header)?;
    if bytes.len() as u64 > HEADER_LIMIT {
        return Err("checkpoint header exceeds bound".into());
    }
    if path.exists() {
        return Err("checkpoint destination already exists".into());
    }
    let mut name = path
        .file_name()
        .ok_or("missing checkpoint filename")?
        .to_os_string();
    name.push(".partial");
    let partial = path.with_file_name(name);
    let mut writer = BufWriter::new(fs::File::create_new(&partial)?);
    let mut sha = Sha256::new();
    write_hashed(&mut writer, &mut sha, MAGIC)?;
    write_hashed(&mut writer, &mut sha, &(bytes.len() as u64).to_le_bytes())?;
    write_hashed(&mut writer, &mut sha, &bytes)?;
    for array in [&net.p, &net.m, &net.v] {
        for value in array {
            write_hashed(&mut writer, &mut sha, &value.to_le_bytes())?;
        }
    }
    let digest = sha.finalize();
    writer.write_all(&digest)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    fs::hard_link(&partial, path)?;
    fs::remove_file(&partial)?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::File::open(parent)?.sync_all()?;
    Ok(digest_text(&digest))
}

pub struct Restored {
    pub model: Transformer,
    pub rng: Mt,
    pub progress: Progress,
    pub body_sha256: String,
}

/// Load one model without a whole-file or duplicate tensor buffer. The caller
/// should drop its previous model before loading a replacement. This function
/// never resumes training itself; the runner must authorize and record that.
pub fn load(path: &Path, expected_binding: &str) -> Result<Restored> {
    if !header::valid_binding(expected_binding) {
        return Err("invalid expected checkpoint binding".into());
    }
    let file = fs::File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let mut sha = Sha256::new();
    if read_hashed::<8>(&mut reader, &mut sha)? != *MAGIC {
        return Err("not a native stress transformer checkpoint".into());
    }
    let header_len = u64::from_le_bytes(read_hashed(&mut reader, &mut sha)?);
    if header_len == 0 || header_len > HEADER_LIMIT || file_size < 16 + header_len + 32 {
        return Err("invalid bounded checkpoint header length".into());
    }
    let mut bytes = vec![0; header_len as usize];
    reader.read_exact(&mut bytes)?;
    sha.update(&bytes);
    let header: Header = serde_json::from_slice(&bytes)?;
    header.validate(expected_binding)?;
    let dims = header.dims();
    if file_size != 16 + header_len + 12 * dims.len() as u64 + 32 {
        return Err("checkpoint tensor payload size differs from dimensions".into());
    }
    let mut model = Transformer::new(dims, 0, header.microbatch, header.hyper())?;
    for (index, array) in [&mut model.p, &mut model.m, &mut model.v]
        .into_iter()
        .enumerate()
    {
        for value in array {
            *value = f32::from_le_bytes(read_hashed(&mut reader, &mut sha)?);
            if !value.is_finite() || (index == 2 && *value < 0.) {
                return Err("invalid saved transformer tensor value".into());
            }
        }
    }
    let mut saved_digest = [0; 32];
    reader.read_exact(&mut saved_digest)?;
    let digest = sha.finalize();
    if digest[..] != saved_digest {
        return Err("checkpoint checksum mismatch".into());
    }
    let mut tail = [0; 1];
    if reader.read(&mut tail)? != 0 {
        return Err("trailing checkpoint bytes".into());
    }
    Ok(Restored {
        model,
        rng: Mt::from_words(&header.rng),
        progress: header.progress,
        body_sha256: digest_text(&digest),
    })
}
