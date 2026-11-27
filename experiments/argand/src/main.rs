//! Argand, a complex decoder with a tied vocabulary, on the frozen GPT
//! stress tiers. Training follows the Lean `fullGradientMomentumRun`.
use arithmetic::{hash, reference as arithmetic_data};

include!(concat!(env!("OUT_DIR"), "/upstream.rs"));
mod bench;
mod engine;
mod runner;
mod suite;

/// The error type shared with the pinned modules.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const KERNITE_VERSION: &str = env!("KERNITE_CUDA_VERSION");

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    runner::command(&args)
}
