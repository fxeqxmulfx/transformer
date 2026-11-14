//! CM38 importer and evaluator ported unchanged from 3852e0b0 except for the
//! immediate reviewed parent: refined 246 already contains causal 245.
use super::*;
use component::{binding, training};
use convex_stress_fallback::{self as fallback, causal as component};
pub(crate) mod imported;
pub(crate) mod inputs;
pub(crate) mod outcomes;
fn mode(m: training::Mode) -> &'static str {
    match m {
        training::Mode::Full => MODES[0],
        training::Mode::Mini => MODES[1],
        training::Mode::Forward => MODES[2],
    }
}
mod comparison {
    use super::*;
    pub(crate) fn causal(parent: &Value) -> &Value {
        &parent["complete_parent"]
    }
    pub(crate) fn cell<'a>(
        parent: &'a Value,
        policy: &str,
        task: &str,
        mode: &str,
    ) -> Result<&'a Value> {
        crate::cell(parent, policy, task, mode)
    }
}
