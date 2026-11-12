use crate::{Result, hash};
use serde_json::Value;
pub const SUITE: &str = include_str!("../../convex_stress/suite.json");
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub fn fingerprint() -> String {
    let sources: &[&[u8]] = &[
        include_bytes!("lib.rs"),
        include_bytes!("main.rs"),
        include_bytes!("meta.rs"),
        include_bytes!("training.rs"),
        include_bytes!("replay.rs"),
        include_bytes!("selection.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        include_bytes!("../../convex_stress_schatten/src/lib.rs"),
        include_bytes!("../../convex_stress_schatten/src/solver.rs"),
        include_bytes!("../../convex_stress_schatten/src/projection.rs"),
        include_bytes!("../../convex_stress_moments/src/lib.rs"),
        include_bytes!("../../convex_stress_moments/src/fft.rs"),
        include_bytes!("../../convex_stress_moments/src/projection.rs"),
        include_bytes!("../../convex_stress_moments/src/solver.rs"),
        include_bytes!("../../convex_stress_spectral/src/lib.rs"),
        include_bytes!("../../convex_stress_spectral/src/linalg.rs"),
        include_bytes!("../../convex_stress_counts/src/lib.rs"),
        include_bytes!("../../convex_stress_counts/src/counts.rs"),
        include_bytes!("../../convex_stress_counts/src/persistent.rs"),
        include_bytes!("../../convex_stress_repair/src/lib.rs"),
        include_bytes!("../../convex_stress_minibatch/src/lib.rs"),
        include_bytes!("../../convex_stress_minibatch/src/input.rs"),
        include_bytes!("../../convex_stress_minibatch/src/persistent.rs"),
        include_bytes!("../../convex_stress_minibatch/src/metrics.rs"),
        include_bytes!("../../convex_stress/src/features.rs"),
        include_bytes!("../../convex_stress/src/channel.rs"),
        include_bytes!("../../convex_stress/src/vendor/data.rs"),
        include_bytes!("../../convex_stress/src/vendor/arithmetic_data.rs"),
        include_bytes!("../../convex_stress/src/vendor/random.rs"),
        include_bytes!("../../convex_stress/src/vendor/pcg.rs"),
        include_bytes!("../../convex_stress/src/vendor/blake.rs"),
        SUITE.as_bytes(),
        PROTOCOL.as_bytes(),
    ];
    let mut bytes = vec![];
    for source in sources {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    hash(&bytes)
}
pub fn rows() -> Result<Vec<Value>> {
    let rows = serde_json::from_str::<Value>(SUITE)?["tasks"]
        .as_array()
        .ok_or("missing tasks")?
        .clone();
    if rows.len() != 35 {
        return Err("expected original 35-task suite".into());
    }
    Ok(rows)
}
pub fn selected(name: &str) -> Result<Value> {
    rows()?
        .into_iter()
        .find(|r| r["name"] == name)
        .ok_or_else(|| "unknown task".into())
}
pub fn protocol() -> Result<Value> {
    Ok(serde_json::from_str(PROTOCOL)?)
}
pub fn allowed(p: &Value, key: &str, name: &str) -> bool {
    p[key].as_array().unwrap().iter().any(|v| v == name)
}
pub fn number(v: &Value, key: &str) -> Result<f64> {
    v[key]
        .as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| format!("missing finite {key}").into())
}
pub fn limit(p: &Value, arm: &str, candidate: &str) -> usize {
    let prefix = if candidate == "counts" {
        "counts"
    } else {
        "spectral"
    };
    let suffix = match arm {
        "full_batch" => "full",
        "minibatch" => "minibatch",
        _ => "forward",
    };
    p[format!("{prefix}_{suffix}_updates")].as_u64().unwrap() as usize
}
