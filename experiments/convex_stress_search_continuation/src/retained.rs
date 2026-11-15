//! Deterministic bounded collection of completed, independently checked fits.
//! No observed-signature deduplication is used for successful programs: equal
//! training signatures do not imply equal predictions on new inputs (CM63).
use crate::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stats {
    pub successful_queries: u64,
    pub fitting_roots_offered: u64,
    pub first_fit_work: Option<u64>,
    pub first_fit_minimum_nodes: Option<usize>,
    pub minimum_nodes: Option<usize>,
    pub interrupted_after_fit: bool,
}
pub struct Retained {
    capacity: usize,
    formulas: Vec<Formula>,
    pub stats: Stats,
}
impl Retained {
    pub fn new(capacity: usize) -> Result<Self> {
        if !(1..=32).contains(&capacity) {
            return Err("continuation retention capacity refused".into());
        }
        Ok(Self {
            capacity,
            formulas: vec![],
            stats: Stats::default(),
        })
    }
    pub fn offer(&mut self, formulas: Vec<Formula>, work: u64) -> Result<()> {
        if formulas.is_empty()
            || formulas.len() > self.capacity
            || formulas.iter().any(|f| f.nodes() > 127)
        {
            return Err("invalid completed fitting query".into());
        }
        let minimum = formulas
            .iter()
            .map(Formula::nodes)
            .min()
            .ok_or("empty fitting query")?;
        if self.stats.first_fit_work.is_none() {
            self.stats.first_fit_work = Some(work);
            self.stats.first_fit_minimum_nodes = Some(minimum);
        }
        self.stats.successful_queries += 1;
        self.stats.fitting_roots_offered += formulas.len() as u64;
        self.formulas.extend(formulas);
        self.formulas
            .sort_unstable_by(|a, b| (a.nodes(), a).cmp(&(b.nodes(), b)));
        self.formulas.dedup();
        self.formulas.truncate(self.capacity);
        self.stats.minimum_nodes = self.formulas.first().map(Formula::nodes);
        Ok(())
    }
    pub fn formulas(&self) -> &[Formula] {
        &self.formulas
    }
    pub fn into_formulas(self) -> Vec<Formula> {
        self.formulas
    }
}
