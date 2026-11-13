//! CM28 / Lean9cd95708, anchoredMatrix and cyclic_of_row_certificate.
//! Only observed names enter the matrix. The certificate proves uniqueness
//! within the bijective cyclic model; structural realizability remains explicit.
use crate::{Result, ring};
use serde::{Deserialize, Serialize};
mod audit;
mod run;
#[cfg(test)]
mod tests;
pub use audit::check;
pub use run::run;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Data {
    pub n: usize,
    pub vocabulary: usize,
    /// Cell = row + n*column; labels are categorical identities only.
    pub observations: Vec<[usize; 2]>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proposal {
    pub rows: Vec<u16>,
    pub columns: Vec<u16>,
    /// Code coordinates follow the classes' first observed occurrence order.
    pub codes: Vec<u16>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub data: Data,
    pub proposal: Proposal,
    pub retained_equations: usize,
    pub policy: ring::Policy,
}
impl Request {
    pub fn validate(&self) -> Result<(Vec<usize>, u64)> {
        let d = &self.data;
        if !(2..=257).contains(&d.n)
            || !(2..=1024).contains(&d.vocabulary)
            || d.observations.is_empty()
            || d.observations.len() > d.n * d.n
            || !(1..=4096).contains(&self.retained_equations)
        {
            return Err("categorical certificate data/equation limits exceeded".into());
        }
        let mut cells = vec![false; d.n * d.n];
        let mut classes = vec![];
        for &[cell, label] in &d.observations {
            if cell >= cells.len() || label >= d.vocabulary || cells[cell] {
                return Err("categorical certificate has invalid/repeated original cells".into());
            }
            cells[cell] = true;
            if !classes.contains(&label) {
                classes.push(label);
            }
        }
        let q = classes.len();
        let p = &self.proposal;
        if !(2..=257).contains(&q)
            || p.rows.len() != d.n
            || p.columns.len() != d.n
            || p.codes.len() != q
            || p.rows
                .iter()
                .chain(&p.columns)
                .chain(&p.codes)
                .any(|&x| x as usize >= q)
        {
            return Err("categorical proposal shape/domain/residue limits exceeded".into());
        }
        // Shape-only forecast: these temporary zero rows are never a certificate.
        let bound = ring::Problem {
            modulus: q as u16,
            variables: 2 * d.n + q,
            rows: vec![vec![0; 2 * d.n + q]; d.observations.len().min(self.retained_equations) + 3],
            policy: self.policy.clone(),
        }
        .validate()?
            + 512 * 1024 * 1024;
        Ok((classes, bound))
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub observation_checks: usize,
    pub normalized_coordinates: usize,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Report {
    pub input: Request,
    pub buffer_bound_bytes: u64,
    pub classes: Vec<usize>,
    pub anchor_row: usize,
    pub anchor_class: Option<usize>,
    pub missing_rows: Vec<usize>,
    pub missing_columns: Vec<usize>,
    pub collision: Option<[usize; 2]>,
    pub conflict: Option<usize>,
    pub normalized: Option<Vec<u16>>,
    pub kernel: Option<ring::Outcome>,
    pub certified: bool,
    pub reason: String,
    pub work: Work,
}
pub fn replay(input: &Request, out: &Report) -> Result<()> {
    if run(input.clone())? != *out {
        return Err("categorical certificate original-data/state/work replay differs".into());
    }
    Ok(())
}
