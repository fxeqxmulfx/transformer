//! Reading a checkpoint's parameters, for the binaries that examine saved runs. The trainer
//! writes and restores checkpoints in `run.rs`; this file is not among its sources.

use std::path::Path;

use sha2::{Digest, Sha256};

use crate::layout::Dims;

/// The parameters of a checkpoint, in doubles, after its digest is checked.
pub fn parameters(path: &Path, dims: Dims) -> Result<Vec<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 96 || &bytes[..8] != b"GROKDIV1" {
        return Err(format!("{}: not a checkpoint", path.display()));
    }
    let (body, digest) = bytes.split_at(bytes.len() - 32);
    if Sha256::digest(body).as_slice() != digest {
        return Err(format!("{}: the digest does not match the bytes", path.display()));
    }
    let len = u64::from_le_bytes(body[56..64].try_into().expect("eight bytes")) as usize;
    if len != dims.len() {
        return Err(format!("{}: another model", path.display()));
    }
    Ok(body[64..64 + 4 * len]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")) as f64)
        .collect())
}
