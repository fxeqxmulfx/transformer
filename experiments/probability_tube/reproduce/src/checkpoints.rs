//! Frozen inventories and full source-state checkpoint serialization.
use crate::{Result, application::State, infrastructure::storage};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::Path};
pub const SEEDS: [u64; 8] = [7, 11, 1, 3, 5, 13, 17, 19];
pub const ANCHORS: [usize; 6] = [0, 1000, 4000, 6000, 8000, 9500];
pub const HORIZONS: [usize; 13] = [0, 1, 2, 4, 8, 16, 32, 64, 128, 250, 500, 1000, 4000];
pub fn probes() -> Vec<usize> {
    [0, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000]
        .into_iter()
        .chain((5000..=10000).step_by(250))
        .collect()
}
pub fn endpoints() -> Vec<usize> {
    ANCHORS
        .into_iter()
        .flat_map(|a| HORIZONS.into_iter().map(move |h| a + h))
        .filter(|&s| s <= 10000)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
pub fn save(path: &Path, step: usize, state: &State) -> Result<Value> {
    fs::create_dir_all(path)?;
    let file = path.join(format!("step-{step:06}.bin"));
    crate::inputs::require(!file.exists(), "refusing to overwrite checkpoint")?;
    let bytes = storage::state_bytes(step, state);
    let sha = storage::hash(&bytes);
    fs::write(&file, &bytes)?;
    let (got, decoded) = storage::decode(&fs::read(&file)?, state.parameters.len())?;
    crate::inputs::require(
        got == step && storage::state_bytes(got, &decoded) == bytes,
        "checkpoint round-trip mismatch",
    )?;
    Ok(
        json!({"step":step,"status":"pass","file":file.file_name().unwrap().to_str(),"bytes":bytes.len(),"sha256":sha}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_declared_forecast_endpoint_is_retained() {
        let saved = endpoints();
        assert_eq!(probes().len(), 31);
        assert!(saved.windows(2).all(|w| w[0] < w[1]));
        for a in ANCHORS {
            for h in HORIZONS {
                if a + h <= 10000 {
                    assert!(saved.contains(&(a + h)));
                }
            }
        }
        assert!(saved.contains(&9501) && saved.contains(&4016));
    }
}
