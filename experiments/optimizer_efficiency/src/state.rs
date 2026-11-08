//! Source signed first-moment and unsigned second-moment storage contract,
//! from forward_training/config.rs at c9995ba5; no local training is invoked.
use crate::{
    application::State,
    domain::model::{self, Arm},
};
pub fn valid(arm: Arm, state: &State) -> bool {
    model::valid(arm, &state.parameters)
        && model::valid(arm, &state.first)
        && state.second.len() == arm.count()
        && state.second.iter().all(|&v| v.is_finite() && v >= 0.)
        && (arm == Arm::Swiglu
            || model::pairs().iter().all(|&[ar, bi, br, ai, kind]| {
                kind == 1
                    || (state.second[ar as usize] == state.second[ai as usize]
                        && state.second[bi as usize] == state.second[br as usize])
            }))
}
