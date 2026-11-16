//! Model-facing fields: original training rows or one visible causal prefix.
//! Source identities, codecs, teacher values and heldout labels stay outside.
use crate::{Result, dataset::Dataset};

pub struct TrainingRow<'a> {
    pub original_index: usize,
    pub tokens: &'a [i64],
    pub targets: &'a [i64],
}
pub struct Training<'a> {
    pub vocabulary: usize,
    pub rows: Vec<TrainingRow<'a>>,
}
impl Dataset {
    pub fn training(&self) -> Result<Training<'_>> {
        let rows = self.splits.get("train").ok_or("missing training split")?;
        Ok(Training {
            vocabulary: self.vocabulary,
            rows: rows
                .records
                .iter()
                .enumerate()
                .map(|(original_index, row)| TrainingRow {
                    original_index,
                    tokens: &row.tokens,
                    targets: &row.targets,
                })
                .collect(),
        })
    }
}

/// Ordinary class logits at the last position of this visible causal prefix.
/// The same interface serves dense teacher-forced CE and own-prefix generation.
pub trait Predictor {
    fn logits(&mut self, prefix: &[i64]) -> Result<Vec<f64>>;
}
impl<F: FnMut(&[i64]) -> Result<Vec<f64>>> Predictor for F {
    fn logits(&mut self, prefix: &[i64]) -> Result<Vec<f64>> {
        self(prefix)
    }
}
