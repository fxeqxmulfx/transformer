//! Feasible mixture of the complete learned old parent and one new trainable block.
//! The retained parent is held only during this prospective new-block phase.
use crate::{Result, block::Block};
use rotary::atom::{State, row};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub parent: rotary::model::Model,
    pub block: Block,
    pub fraction: f64,
}
impl Model {
    pub fn validate(&self) -> Result<f64> {
        if self.parent.objects != self.block.objects()
            || !self.fraction.is_finite()
            || !(0. ..=1.).contains(&self.fraction)
        {
            return Err("invalid mixed function family".into());
        }
        Ok(self.parent.validate()?.max(self.block.validate()?))
    }
    pub fn mass(&self) -> f64 {
        self.fraction * self.parent.budget
    }
    pub fn parent_logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        Ok(self
            .parent
            .logits(prefix)?
            .iter()
            .map(|x| (1. - self.fraction) * x)
            .collect())
    }
    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        let z = self.block.atom.state(prefix)?;
        combine(&self.parent_logits(prefix)?, &self.block, z, self.mass())
    }
}
pub fn combine(base: &[f64], block: &Block, z: State, mass: f64) -> Result<Vec<f64>> {
    if base.len() != block.objects() || !mass.is_finite() || mass < 0. {
        return Err("incomplete mixed logits".into());
    }
    let logits: Vec<_> = base
        .iter()
        .zip(&block.atom.output)
        .map(|(&s, &u)| s + mass * row(u, z).re)
        .collect();
    if logits.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite complete mixed logits".into());
    }
    Ok(logits)
}
