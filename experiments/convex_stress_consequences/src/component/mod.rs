//! CM27 checked unit normalization and residual lifts, followed by bounded
//! branching. Class-coordinate multiplicities are an explicit proposal filter,
//! not a claim that observed sample frequencies are balanced. Discovery is
//! discrete and bounded; decoded identification still needs CM29 certificates.
use crate::{Result, affine, finite_system, lifts};
use convex_stress_modular::{System, coding::Factor};
use serde::{Deserialize, Serialize};
mod audit;
mod plan;
mod run;
#[cfg(test)]
mod tests;
pub use audit::check;
pub use run::run;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub stages: usize,
    pub states: usize,
    pub total_scalar_updates: u64,
    pub elimination: finite_system::Policy,
    pub enumeration: affine::Policy,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            stages: 16,
            states: 512,
            total_scalar_updates: 2_000_000_000,
            elimination: finite_system::Policy {
                nullity: 8,
                ..finite_system::Policy::default()
            },
            enumeration: affine::Policy::default(),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub n: usize,
    pub classes: usize,
    pub factor: Factor,
    pub system: System,
    pub policy: Policy,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Origin {
    Base { point: usize },
    Digit { stage: usize, point: usize },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    NoClassUnit,
    Multiplicity {
        residue: usize,
        count: usize,
        expected: usize,
    },
    Duplicate {
        state: usize,
    },
    StateCap,
    Pending,
    Expanded {
        stage: usize,
    },
    StageCap,
    ScalarCap,
    Complete,
}
impl Decision {
    fn retained(&self) -> bool {
        matches!(
            self,
            Self::Pending
                | Self::Expanded { .. }
                | Self::StageCap
                | Self::ScalarCap
                | Self::Complete
        )
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct State {
    pub origin: Origin,
    pub modulus: u16,
    pub anchor: Option<usize>,
    pub inverse: Option<u16>,
    pub point: Vec<u16>,
    pub counts: Vec<usize>,
    pub decision: Decision,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stage {
    pub parent: usize,
    pub report: lifts::Report,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub stages: usize,
    pub finite_scalar_updates: u64,
    pub finite_row_operations: usize,
    pub finite_pivot_tests: usize,
    pub finite_inverse_tests: usize,
    pub affine_parameter_digits: usize,
    pub affine_coordinate_products: u64,
    pub affine_original_products: u64,
    pub lift_original_products: u64,
    pub inverse_tests: usize,
    pub normalization_products: usize,
    pub class_coordinates: usize,
    pub endpoint_products: u64,
    pub ce_gradients: usize,
}
impl Work {
    fn charge(&mut self, r: &affine::Report) {
        self.stages += 1;
        self.finite_scalar_updates += r.elimination.work.scalar_updates;
        self.finite_row_operations += r.elimination.work.row_operations;
        self.finite_pivot_tests += r.elimination.work.pivot_tests;
        self.finite_inverse_tests += r.elimination.work.inverse_tests;
        self.affine_parameter_digits += r.work.parameter_digits;
        self.affine_coordinate_products += r.work.coordinate_products;
        self.affine_original_products += r.work.original_products;
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub input: Request,
    pub buffer_preflight_bytes: u64,
    pub base: Option<affine::Report>,
    pub stages: Vec<Stage>,
    pub states: Vec<State>,
    pub complete: Vec<usize>,
    pub reason: String,
    pub work: Work,
}
pub fn replay(input: &Request, out: &Report) -> Result<()> {
    if run(input.clone())? != *out {
        return Err("component original-data/branch/cap/order/work replay differs".into());
    }
    Ok(())
}
