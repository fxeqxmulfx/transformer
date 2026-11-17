//! CM107 finite_feasible: arbitrary complete learned blocks share one budget.
//! The old learned rotary parent remains stored, including unused components.
use crate::Result;
use hilbert::{block::Block, geometry::Geometry};
use rotary::atom::row;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Weighted {
    pub block: Block,
    pub weight: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub parent: rotary::model::Model,
    pub parent_fraction: f64,
    pub blocks: Vec<Weighted>,
}
impl Model {
    pub fn from_source(source: hilbert::model::Model) -> Result<Self> {
        source.validate()?;
        let weight = source.mass();
        let model = Self {
            parent: source.parent,
            parent_fraction: 1. - source.fraction,
            blocks: vec![Weighted {
                block: source.block,
                weight,
            }],
        };
        model.validate()?;
        Ok(model)
    }
    pub fn objects(&self) -> usize {
        self.parent.objects
    }
    pub fn budget(&self) -> f64 {
        self.parent.budget
    }
    pub fn components(&self) -> usize {
        self.parent.atoms.len() + self.blocks.len()
    }
    pub fn geometry(&self) -> Result<Geometry> {
        Ok(self
            .blocks
            .first()
            .ok_or("no learned block")?
            .block
            .geometry)
    }
    pub fn used_mass(&self) -> f64 {
        self.parent_fraction * self.budget() + self.blocks.iter().map(|w| w.weight).sum::<f64>()
    }
    pub fn validate(&self) -> Result<f64> {
        let mut excess = self.parent.validate()?;
        if !self.parent_fraction.is_finite()
            || !(0. ..=1.).contains(&self.parent_fraction)
            || (self.used_mass() - self.budget()).abs() > 2e-12 * self.budget()
        {
            return Err("mixed capacity differs from fixed budget".into());
        }
        let geometry = self.geometry()?;
        for w in &self.blocks {
            if !w.weight.is_finite()
                || w.weight < 0.
                || w.block.objects() != self.objects()
                || w.block.geometry != geometry
            {
                return Err("incomplete mixed component".into());
            }
            excess = excess.max(w.block.validate()?);
        }
        Ok(excess)
    }
    pub fn append(&mut self, block: Block, fraction: f64) -> Result<usize> {
        self.validate()?;
        block.validate()?;
        if block.geometry != self.geometry()?
            || block.objects() != self.objects()
            || !fraction.is_finite()
            || !(0. ..=1.).contains(&fraction)
        {
            return Err("invalid complete convex append".into());
        }
        self.parent_fraction *= 1. - fraction;
        for w in &mut self.blocks {
            w.weight *= 1. - fraction;
        }
        let weight = fraction * self.budget();
        self.blocks.push(Weighted { block, weight });
        self.validate()?;
        Ok(self.blocks.len() - 1)
    }
    pub fn base_logits(&self, active: Option<usize>, prefix: &[i64]) -> Result<Vec<f64>> {
        if active.is_some_and(|i| i >= self.blocks.len()) {
            return Err("active component outside mixture".into());
        }
        let mut logits = self.parent.logits(prefix)?;
        for x in &mut logits {
            *x *= self.parent_fraction;
        }
        for (j, w) in self.blocks.iter().enumerate() {
            if Some(j) == active {
                continue;
            }
            let z = w.block.atom.state(prefix)?;
            for (l, &u) in logits.iter_mut().zip(&w.block.atom.output) {
                *l += w.weight * row(u, z).re;
            }
        }
        if logits.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite complete mixture execution".into());
        }
        Ok(logits)
    }
    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        self.base_logits(None, prefix)
    }
    pub fn base_hash(&self, active: usize) -> Result<String> {
        let w = self.blocks.get(active).ok_or("active base hash")?;
        let held: Vec<_> = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != active)
            .collect();
        Ok(bilinear::experiment::hash(&serde_json::to_vec(
            &serde_json::json!({"parent":self.parent,"parent_fraction":self.parent_fraction,"held":held,"active":active,"active_weight":w.weight,"active_geometry":w.block.geometry,"objects":self.objects()}),
        )?))
    }
}
