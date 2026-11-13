//! CM27 fits_recombine/product_dvd: bounded products of the actual component
//! banks. Every CRT witness and final original equation is checked. Injective
//! class codes are proposals; CM29 decoded identification remains separate.
use crate::{Result, component};
use convex_stress_modular::{System, coding::Factor, recombine};
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
    pub points: usize,
    pub original_products: u64,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            points: 64,
            original_products: 500_000_000,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub n: usize,
    pub classes: usize,
    pub system: System,
    pub component_policy: component::Policy,
    pub policy: Policy,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Branch {
    pub ordinal: usize,
    /// Actual state indices in the component reports, not supplied codes.
    pub component_states: Vec<usize>,
    pub crt: Vec<recombine::Step>,
    pub point: Vec<u16>,
    pub collision: Option<[usize; 3]>,
    pub duplicate: Option<usize>,
    pub retained: bool,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub components: Vec<component::Work>,
    pub tuples: usize,
    pub parameter_digits: usize,
    pub crt_steps: usize,
    pub inverse_tests: usize,
    pub crt_original_products: u64,
    pub endpoint_original_products: u64,
    pub class_coordinates: usize,
    pub duplicate_tests: usize,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub input: Request,
    pub buffer_preflight_bytes: u64,
    pub components: Vec<component::Report>,
    /// Product of emitted complete component banks, not all possible codes.
    pub cardinality: u64,
    pub branches: Vec<Branch>,
    pub candidates: Vec<usize>,
    pub entire_emitted_product: bool,
    pub reason: String,
    pub work: Work,
}
pub fn replay(input: &Request, out: &Report) -> Result<()> {
    if run(input.clone())? != *out {
        return Err("composite original-data/component/CRT/branch/work replay differs".into());
    }
    Ok(())
}
