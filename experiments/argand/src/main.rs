//! Argand, a complex decoder with a tied vocabulary, on the frozen GPT
//! stress tiers. Training follows the Lean `fullGradientMomentumRun`.
use arithmetic::{hash, reference as arithmetic_data};

include!(concat!(env!("OUT_DIR"), "/upstream.rs"));
mod and_audit;
mod bench;
mod checkpoint_proposal;
mod division;
mod division_certificate;
#[allow(dead_code)]
#[path = "../../grok_division/src/data.rs"]
mod division_equations;
mod engine;
mod margins;
#[allow(dead_code)]
#[path = "../../grok_division/src/mt.rs"]
mod mt;
mod page_confirmation;
mod page_runs;
mod rank_one_loss;
mod runner;
mod suite;
mod sum8;
mod sum8_spectrum;
#[path = "../../signed_transport/src/svd.rs"]
mod svd;

/// The error type shared with the pinned modules.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const KERNITE_VERSION: &str = env!("KERNITE_CUDA_VERSION");

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    runner::command(&args)
}
