//! Retain reviewed 276 imports and ancestry; apply the actual CM41 wrapper.
use super::*;
use component::{binding, training};
use convex_stress_partial_fallback::causal as component;
mod fallback {
    pub use convex_stress_fallback::digest;
    pub use convex_stress_partial_fallback::*;
}
#[path = "../../../convex_stress_shared_compare/src/causal/imported.rs"]
pub(crate) mod imported;
#[path = "../../../convex_stress_shared_compare/src/causal/inputs.rs"]
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
