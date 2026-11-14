//! CM39 at2ca6d2cf: actual bounded complex heads with a conservative fixed gate.
//! The raw head predicts on its support and on joint refusal. The grouped head
//! predicts only newly covered cells. CM36/37 coverage is checked on the actual
//! complete codes; neither acceptance nor flatness identifies an unknown target.
pub use convex_stress_readout_kernel as grouped;
pub use grouped::{C, Result, kernel};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
mod binding;
pub mod bounds;
mod gradient;
pub use bounds::Bounds;
pub use gradient::Gradient;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub prior_sha256: String,
    pub supplement_sha256: String,
    pub raw_discovery_sha256: String,
    pub grouped_discovery_sha256: String,
    pub records_sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Source {
    Prior,
    Supplement,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Route {
    pub source: Source,
    pub prior_supported: bool,
    pub supplement_supported: bool,
    pub prior_axes_encoded: bool,
    pub supplement_axes_encoded: bool,
    pub exact_observed_input: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Coverage {
    pub prior_supported: usize,
    pub newly_covered: usize,
    pub jointly_refused: usize,
    pub unknown_prior_axes: usize,
    pub unmatched_prior_cells: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Model {
    prior: kernel::Model,
    supplement: kernel::Model,
    records: Vec<[usize; 2]>,
    binding: Binding,
    bounds: Bounds,
    coverage: Coverage,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Saved {
    prior: kernel::Model,
    supplement: kernel::Model,
    records: Vec<[usize; 2]>,
    binding: Binding,
    bounds: Bounds,
    coverage: Coverage,
}

pub fn digest<T: Serialize>(value: &T) -> Result<String> {
    Ok(Sha256::digest(serde_json::to_vec(value)?)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

impl Model {
    pub fn load(
        value: serde_json::Value,
        raw: &kernel::discovery::Report,
        grouped: &grouped::discovery::Report,
        expected: &Binding,
    ) -> Result<Self> {
        let saved: Saved = serde_json::from_value(value)?;
        if &saved.binding != expected {
            return Err("saved extension differs from external bindings".into());
        }
        let rebuilt = Self::new(saved.prior, saved.supplement, raw, grouped, saved.records)?;
        if rebuilt.binding != *expected
            || rebuilt.bounds != saved.bounds
            || rebuilt.coverage != saved.coverage
        {
            return Err("saved extension models, records, budgets or coverage changed".into());
        }
        Ok(rebuilt)
    }
    pub fn prior(&self) -> &kernel::Model {
        &self.prior
    }
    pub fn supplement(&self) -> &kernel::Model {
        &self.supplement
    }
    pub fn records(&self) -> &[[usize; 2]] {
        &self.records
    }
    pub fn binding(&self) -> &Binding {
        &self.binding
    }
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }
    pub fn coverage(&self) -> &Coverage {
        &self.coverage
    }
    pub fn route(&self, cell: usize) -> Result<Route> {
        if cell >= self.prior.n * self.prior.n {
            return Err("cell outside the shared categorical domain".into());
        }
        let prior_supported = self.prior.known(cell);
        let supplement_supported = self.supplement.known(cell);
        let (block, state) = self.prior.location(cell);
        Ok(Route {
            source: if !prior_supported && supplement_supported {
                Source::Supplement
            } else {
                Source::Prior
            },
            prior_supported,
            supplement_supported,
            prior_axes_encoded: self.prior.latent(cell).is_some(),
            supplement_axes_encoded: self.supplement.latent(cell).is_some(),
            exact_observed_input: state < self.prior.blocks[block].observed.len(),
        })
    }
    pub fn logits_cell(&self, cell: usize) -> Result<Vec<f64>> {
        Ok(match self.route(cell)?.source {
            Source::Prior => self.prior.logits_cell(cell),
            Source::Supplement => self.supplement.logits_cell(cell),
        })
    }
    /// The original pair head reads the last two categorical tokens. Preserve
    /// that feature map while checking the complete supplied prefix domain.
    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        if !(2..=4096).contains(&prefix.len())
            || prefix
                .iter()
                .any(|&t| t < 0 || t as usize >= self.prior.vocabulary)
        {
            return Err("prefix outside the shared pair prediction domain".into());
        }
        let row = prefix[prefix.len() - 1] as usize;
        let column = prefix[prefix.len() - 2] as usize;
        if row >= self.prior.n || column >= self.prior.n {
            return Err("pair token outside the categorical axis domain".into());
        }
        self.logits_cell(row + self.prior.n * column)
    }
}

impl kernel::metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.prior.vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.logits(prefix)
            .expect("validated coverage-extension domain")
    }
}
