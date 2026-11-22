//! Resume a clean observation boundary of the pinned AdamW/WSD trainer.
//! The state layout follows the pinned native runner's checkpoint and restore.
use super::*;

const FILE: &str = "progress.bin";
const MAGIC: &[u8; 8] = b"PHASOR01";
const DIGEST_LEN: usize = 64;

pub(super) fn finish(dir: &Path) -> Result<()> {
    let path = dir.join(FILE);
    if path.try_exists()? {
        fs::remove_file(path)?;
    }
    Ok(())
}

pub(super) struct Progress {
    pub completed: usize,
    pub elapsed: f64,
    pub history: String,
}

fn protocol(row: &Value, seed: u32, e: &Engine) -> Value {
    json!({"case":row,"seed":seed,"model":MODEL,"suite_sha256":suite_hash(row),
        "kernite_version":KERNITE_VERSION,"gpu":e.cuda.name(),"max_batch":e.config.max_batch,
        "max_length":e.config.max_length,"vocab":e.config.vocab,"parameter_count":e.parameters.len(),
        "fp32_affines":precision::FP32_AFFINE_REFERENCE,"bf16":precision::BF16,
        "tf32_backward":precision::TF32_BACKWARD})
}

pub(super) fn save(
    dir: &Path,
    row: &Value,
    seed: u32,
    e: &Engine,
    point: &Value,
    elapsed: f64,
) -> Result<()> {
    let history = fs::read_to_string(dir.join("observations.jsonl"))?;
    let header = serde_json::to_vec(&json!({"protocol":protocol(row,seed,e),"point":point,
        "elapsed_seconds":elapsed,"history":history}))?;
    e.cuda.sync()?;
    let params = e.parameters.read()?;
    let first = e.first_moment.read()?;
    let second = e.second_moment.read()?;
    layout::validate_state(&e.config, &params, &first, &second)?;
    let mut bytes = Vec::with_capacity(16 + header.len() + 12 * params.len() + DIGEST_LEN);
    bytes.extend(MAGIC);
    bytes.extend((header.len() as u64).to_le_bytes());
    bytes.extend(header);
    for x in params.into_iter().chain(first).chain(second) {
        bytes.extend(x.to_le_bytes());
    }
    let digest = hash(&bytes);
    bytes.extend(digest.as_bytes());
    publish(dir, FILE, &bytes)
}

impl Progress {
    pub(super) fn load(dir: &Path, row: &Value, seed: u32, e: &mut Engine) -> Result<Option<Self>> {
        let path = dir.join(FILE);
        if !path.try_exists()? {
            return Ok(None);
        }
        let bytes = fs::read(&path)?;
        let length = bytes
            .len()
            .checked_sub(DIGEST_LEN)
            .ok_or("truncated progress file")?;
        let (payload, digest) = bytes.split_at(length);
        if digest != hash(payload).as_bytes() || payload.get(..8) != Some(MAGIC) {
            return Err("progress checksum or format mismatch".into());
        }
        let header_length = u64::from_le_bytes(
            payload
                .get(8..16)
                .ok_or("missing progress header")?
                .try_into()?,
        );
        let (header, state) = payload[16..]
            .split_at_checked(usize::try_from(header_length)?)
            .ok_or("truncated progress header")?;
        let header: Value = serde_json::from_slice(header)?;
        let n = e.parameters.len();
        if header["protocol"] != protocol(row, seed, e) || state.len() != 12 * n {
            return Err("progress belongs to a different model, training protocol or shape".into());
        }
        let completed = usize::try_from(
            header["point"]["step"]
                .as_u64()
                .ok_or("missing progress step")?,
        )?;
        let history = header["history"]
            .as_str()
            .ok_or("missing progress observations")?;
        let elapsed = header["elapsed_seconds"]
            .as_f64()
            .ok_or("missing progress timing")?;
        let restore = Self {
            completed,
            elapsed,
            history: history.to_owned(),
        };
        if !elapsed.is_finite()
            || elapsed < 0.
            || header["point"]["pass"] != false
            || restore.observation(completed)? != header["point"]
        {
            return Err("invalid progress endpoint".into());
        }
        let data: Vec<_> = state
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        layout::validate_state(&e.config, &data[..n], &data[n..2 * n], &data[2 * n..])?;
        e.parameters.write(&data[..n])?;
        e.first_moment.write(&data[n..2 * n])?;
        e.second_moment.write(&data[2 * n..])?;
        Ok(Some(restore))
    }

    pub(super) fn observation(&self, at: usize) -> Result<Value> {
        for line in self.history.lines().rev() {
            let point: Value = serde_json::from_str(line)?;
            if point["step"] == at {
                return Ok(point);
            }
        }
        Err("scheduled observation missing from progress file".into())
    }
}
