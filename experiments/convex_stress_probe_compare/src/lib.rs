//! Complete six-probe comparison of the fixed causal CM64/70 model atba159ae6.
//! Freeze before fitting; seal every training endpoint before any heldout call.
//! Numeric teachers and original transformer histories remain outside the model.
pub use joint::{Result, Work};
mod forecast;
pub mod inputs;
pub mod io;
pub mod resources;
pub mod schema;
pub mod scoring;
pub mod seal;
pub mod stage;
#[cfg(test)]
mod tests;
pub mod trace;
pub mod verification;
pub const DATA_HASH: &str = "12fd36f6076b922aa959ab0b9c0a9ace11d34a0df3f0579e5445ac107e3f9033";
pub const WORK_LIMIT: u64 = 2_000_000_000_000;
pub const MODES: [joint::training::Mode; 2] =
    [joint::training::Mode::Forward, joint::training::Mode::Mini];
pub fn name(mode: joint::training::Mode) -> Result<&'static str> {
    match mode {
        joint::training::Mode::Forward => Ok("forward"),
        joint::training::Mode::Mini => Ok("mini"),
        _ => Err("full batch is outside this frozen comparison".into()),
    }
}
pub(crate) fn probe<T>(result: probes::Result<T>) -> Result<T> {
    result.map_err(|error| error.to_string().into())
}
