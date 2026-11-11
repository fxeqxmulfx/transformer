//! Minibatch training on the unchanged frozen stress data and causal feature map.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
use sha2::{Digest, Sha256};
pub fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
#[expect(
    clippy::should_implement_trait,
    reason = "Preserve the frozen generator's public next method"
)]
#[path = "../../convex_stress/src/vendor/arithmetic_data.rs"]
pub mod arithmetic_data;
#[path = "../../convex_stress/src/vendor/blake.rs"]
pub mod blake;
#[path = "../../convex_stress/src/channel.rs"]
pub mod channel;
#[path = "../../convex_stress/src/vendor/data.rs"]
pub mod data;
#[path = "../../convex_stress/src/features.rs"]
pub mod features;
#[path = "../../convex_stress/src/forward.rs"]
pub mod forward;
pub mod input;
pub mod metrics;
#[path = "../../convex_stress/src/vendor/pcg.rs"]
pub mod pcg;
pub mod persistent;
#[path = "../../convex_stress/src/vendor/random.rs"]
pub mod random;
#[path = "../../convex_stress/src/metrics.rs"]
pub mod reference_metrics;
pub mod stochastic;
#[path = "../../convex_stress/src/train.rs"]
pub mod train;
