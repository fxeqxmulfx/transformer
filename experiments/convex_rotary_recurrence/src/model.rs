//! CM95/98 finite convex function mixtures, with no removed original classes.
//! Ported from convex_recurrent_sequence model at1786526f; full matrix atoms replace scalars.
use crate::{
    C, Result,
    atom::{Atom, Family, row},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub objects: usize,
    pub family: Family,
    pub budget: f64,
    pub atoms: Vec<Atom>,
    pub weights: Vec<f64>,
}
impl Model {
    pub fn zero(objects: usize, budget: f64, family: Family) -> Result<Self> {
        let result = Self {
            objects,
            family,
            budget,
            atoms: vec![Atom::zero(objects, family)],
            weights: vec![budget],
        };
        result.validate()?;
        Ok(result)
    }
    pub fn validate(&self) -> Result<f64> {
        if self.objects == 0
            || self.atoms.is_empty()
            || self.atoms.len() != self.weights.len()
            || !self.budget.is_finite()
            || self.budget <= 0.
            || self.weights.iter().any(|w| !w.is_finite() || *w < 0.)
        {
            return Err("invalid complete recurrent mixture".into());
        }
        let zero = &self.atoms[0];
        if zero.initial != [C::default(); 2]
            || zero
                .transition
                .iter()
                .flatten()
                .flatten()
                .chain(zero.output.iter().flatten())
                .any(|&z| z != C::default())
            || zero.frequency.is_some_and(|f| f != [0.; 2])
        {
            return Err("unused-capacity component changed".into());
        }
        if (self.weights.iter().sum::<f64>() - self.budget).abs() > 2e-10 * self.budget {
            return Err("mixture mass differs from fixed capacity".into());
        }
        self.atoms.iter().try_fold(0f64, |e, a| {
            Ok(e.max(a.validate(self.objects, self.family)?))
        })
    }
    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        if prefix
            .iter()
            .any(|&o| o < 0 || o as u128 >= self.objects as u128)
        {
            return Err("object outside original vocabulary".into());
        }
        let mut logits = vec![0.; self.objects];
        for (atom, &weight) in self.atoms.iter().zip(&self.weights) {
            if weight == 0. {
                continue;
            }
            let state = atom.state(prefix)?;
            for (score, &u) in logits.iter_mut().zip(&atom.output) {
                *score += weight * row(u, state).re;
            }
        }
        Ok(logits)
    }
    pub fn next_object(&self, prefix: &[i64]) -> Result<usize> {
        let logits = self.logits(prefix)?;
        Ok((1..logits.len()).fold(0, |best, c| if logits[c] > logits[best] { c } else { best }))
    }
    pub fn shrink_for_queries(&mut self, h: f64) -> Result<()> {
        if !h.is_finite() || h <= 0. || h > self.budget {
            return Err("finite CE query outside capacity".into());
        }
        for weight in &mut self.weights {
            *weight *= 1. - h / self.budget;
        }
        self.weights[0] += h;
        self.validate()?;
        Ok(())
    }
    pub fn mix(&mut self, atom: Atom, eta: f64) -> Result<()> {
        if !eta.is_finite() || !(0. ..=1.).contains(&eta) {
            return Err("step outside convex interval".into());
        }
        atom.validate(self.objects, self.family)?;
        for weight in &mut self.weights {
            *weight *= 1. - eta;
        }
        self.atoms.push(atom);
        self.weights.push(eta * self.budget);
        self.validate()?;
        Ok(())
    }
    pub fn reservation(&self, records: usize) -> Result<u64> {
        let bytes = 128u128 * (self.atoms.len() as u128 + 16) * (6 * self.objects as u128 + 4)
            + 512u128 * records as u128 * self.objects as u128;
        u64::try_from(bytes).map_err(Into::into)
    }
}
