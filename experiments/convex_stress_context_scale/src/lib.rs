//! CM43 at2dcaa38a: fixed confidence scaling after reviewed CM42 features.
//! Selection uses original training labels and canonical logits only. The
//! actual prediction uses all unchanged complex coefficients. No new optimizer.
pub use context::{C, Result, digest, kernel, records};
pub use convex_stress_context as context;
use serde::{Deserialize, Serialize};
mod binding;
mod selection;
pub use selection::{Candidate, Selection};
#[cfg(test)]
mod tests;

pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const BANK: [f64; 6] = [1., 0.75, 0.5, 0.375, 0.25, 0.125];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub context_sha256: String,
    pub protocol_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bounds {
    pub context: context::Bounds,
    pub extra_buffer_bytes: u64,
    pub buffer_bytes: u64,
    pub calibration_scalar_work: u64,
    pub scaled_cells: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Model {
    context: context::Model,
    binding: Binding,
    selection: Selection,
    bounds: Bounds,
}

impl Model {
    pub fn context(&self) -> &context::Model {
        &self.context
    }
    pub fn source(&self) -> &kernel::Model {
        self.context.source()
    }
    pub fn records(&self) -> &[records::Record] {
        self.context.records()
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

    pub fn scale(&self, cell: usize) -> Result<f64> {
        Ok(if self.context.neighbors(cell)?.is_some() {
            self.selection.selected_scale
        } else {
            1.
        })
    }

    /// CM43 features: scale the actual fixed complex coordinates, not labels.
    pub fn features(&self, cell: usize) -> Result<(usize, Vec<C>)> {
        let (block, mut phi) = self.context.features(cell)?;
        let scale = self.scale(cell)?;
        if scale != 1. {
            for p in &mut phi {
                *p *= scale;
            }
        }
        Ok((block, phi))
    }

    pub fn logits_cell(&self, cell: usize) -> Result<Vec<f64>> {
        if self.scale(cell)? == 1. {
            return self.context.logits_cell(cell);
        }
        let (block, phi) = self.features(cell)?;
        let source = self.source();
        let b = &source.blocks[block];
        let v = source.vocabulary;
        let mut logits = vec![0.; v];
        for (f, p) in phi.iter().enumerate() {
            let h = 2 * (b.offset + f) * v;
            for (c, z) in logits.iter_mut().enumerate() {
                *z += p.re * source.coefficients[h + c] - p.im * source.coefficients[h + v + c];
            }
        }
        Ok(logits)
    }

    /// Same validated finite prediction domain as the reviewed context head.
    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        let source = self.source();
        if !(2..=context::PREDICTION_LIMIT).contains(&prefix.len())
            || prefix
                .iter()
                .any(|&t| t < 0 || t as usize >= source.vocabulary)
            || prefix[prefix.len() - 2..]
                .iter()
                .any(|&t| t as usize >= source.n)
        {
            return Err("prefix outside the context-scale prediction domain".into());
        }
        self.logits_cell(
            prefix[prefix.len() - 1] as usize + source.n * prefix[prefix.len() - 2] as usize,
        )
    }

    /// CM43 batch_fderiv_eq. Construction checks scale one at every original
    /// record, so these are the actual complete scaled-model derivatives too.
    pub fn batch_gradient(&self, positions: &[usize]) -> Result<context::Gradient> {
        self.context.batch_gradient(positions)
    }
}

impl kernel::metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.source().vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.logits(prefix)
            .expect("validated context-scale prediction domain")
    }
}
