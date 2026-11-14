//! CM42 at81f4e46e: fixed context weights over actual complex source features.
//! The declared ordinal metric and training-only canonical neighbor selection
//! add an explicit locality bias. Flatness bounds the source head; it supplies
//! no unseen raw labels. All original coefficients and observed rows are kept.
pub use convex_stress_shared_roles::{C, Result, kernel, records};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
mod binding;
mod gradient;
mod selection;
pub use gradient::Gradient;
pub use selection::{Candidate, Selection};
#[cfg(test)]
mod tests;

pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const BANK: [usize; 5] = [0, 1, 3, 5, 9];
pub const MARGIN: f64 = 24.;
pub const PREDICTION_LIMIT: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub source_sha256: String,
    pub records_sha256: String,
    pub protocol_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bounds {
    pub source: kernel::limits::Bounds,
    pub extra_buffer_bytes: u64,
    pub buffer_bytes: u64,
    pub construction_scalar_work: u64,
    pub interpolation_entries: usize,
    pub changed_cells: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Model {
    source: kernel::Model,
    records: Vec<records::Record>,
    binding: Binding,
    selection: Selection,
    /// None is the exact original source row; Some lists uniformly weighted
    /// source-state indices within the original latent block.
    rows: Vec<Option<Vec<usize>>>,
    bounds: Bounds,
}

pub fn digest<T: Serialize>(value: &T) -> Result<String> {
    Ok(Sha256::digest(serde_json::to_vec(value)?)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

pub(crate) fn ambiguous(b: &kernel::model::Block) -> bool {
    b.observed_labels.iter().collect::<BTreeSet<_>>().len() > 1
}

/// The metric uses input IDs only; no target arithmetic or class distance.
pub(crate) fn nearest(source: &kernel::Model, block: usize, cell: usize, k: usize) -> Vec<usize> {
    let mut distances: Vec<_> = source.blocks[block]
        .observed
        .iter()
        .enumerate()
        .filter(|&(_, &other)| other != cell)
        .map(|(state, &other)| {
            let dx = (cell % source.n).abs_diff(other % source.n);
            let dy = (cell / source.n).abs_diff(other / source.n);
            (dx * dx + dy * dy, other, state)
        })
        .collect();
    distances.sort_unstable();
    distances.into_iter().take(k).map(|(_, _, s)| s).collect()
}

impl Model {
    pub fn source(&self) -> &kernel::Model {
        &self.source
    }
    pub fn records(&self) -> &[records::Record] {
        &self.records
    }
    pub fn binding(&self) -> &Binding {
        &self.binding
    }
    pub fn selection(&self) -> &Selection {
        &self.selection
    }
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    pub fn neighbors(&self, cell: usize) -> Result<Option<&[usize]>> {
        self.rows
            .get(cell)
            .map(|r| r.as_deref())
            .ok_or_else(|| "context cell outside domain".into())
    }

    /// Actual complex feature coordinates, before their unchanged output head.
    pub fn features(&self, cell: usize) -> Result<(usize, Vec<C>)> {
        let row = self.neighbors(cell)?;
        let (block, state) = self.source.location(cell);
        let b = &self.source.blocks[block];
        let n = b.states();
        let mut phi = vec![C::new(0., 0.); n];
        if let Some(states) = row {
            let weight = 1. / states.len() as f64;
            for &s in states {
                for (f, value) in phi.iter_mut().enumerate() {
                    *value += weight * b.code[s * n + f];
                }
            }
        } else {
            phi.copy_from_slice(&b.code[state * n..(state + 1) * n]);
        }
        Ok((block, phi))
    }

    pub fn logits_cell(&self, cell: usize) -> Result<Vec<f64>> {
        if self.neighbors(cell)?.is_none() {
            return Ok(self.source.logits_cell(cell));
        }
        let (block, phi) = self.features(cell)?;
        let b = &self.source.blocks[block];
        let v = self.source.vocabulary;
        let mut logits = vec![0.; v];
        for (f, p) in phi.iter().enumerate() {
            let h = 2 * (b.offset + f) * v;
            for (c, logit) in logits.iter_mut().enumerate() {
                *logit += p.re * self.source.coefficients[h + c]
                    - p.im * self.source.coefficients[h + v + c];
            }
        }
        Ok(logits)
    }

    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        if !(2..=PREDICTION_LIMIT).contains(&prefix.len())
            || prefix
                .iter()
                .any(|&t| t < 0 || t as usize >= self.source.vocabulary)
            || prefix[prefix.len() - 2..]
                .iter()
                .any(|&t| t as usize >= self.source.n)
        {
            return Err("prefix outside the fixed context prediction domain".into());
        }
        self.logits_cell(
            prefix[prefix.len() - 1] as usize + self.source.n * prefix[prefix.len() - 2] as usize,
        )
    }
}

impl kernel::metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.source.vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.logits(prefix)
            .expect("validated context prediction domain")
    }
}
