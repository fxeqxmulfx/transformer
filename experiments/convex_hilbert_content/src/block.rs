//! All complete initial, object-matrix and output coordinates remain trainable.
//! CM107 embeds the old plain block through zero rotary frequencies.
use crate::{
    C, Result,
    geometry::{self, Geometry},
};
use baseline::random::Random;
use rotary::atom::{Atom, Family};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Block {
    pub geometry: Geometry,
    pub atom: Atom,
}
impl Block {
    pub fn initial(
        objects: usize,
        geometry: Geometry,
        seed: u32,
        radius: f64,
        jitter: f64,
    ) -> Result<Self> {
        if objects == 0
            || !radius.is_finite()
            || radius <= 0.
            || radius > 1.
            || !jitter.is_finite()
            || jitter <= 0.
        {
            return Err("invalid generic initialization".into());
        }
        let mut rng = Random::new(seed);
        let mut atom = Atom::identity_start(objects, Family::Plain);
        for z in &mut atom.initial {
            *z = C::new(rng.random() - 0.5, rng.random() - 0.5);
        }
        let n = atom.initial[0].norm().hypot(atom.initial[1].norm());
        if n == 0. {
            return Err("zero random initializer".into());
        }
        for z in &mut atom.initial {
            *z *= radius / n;
        }
        for z in atom.transition.iter_mut().flatten().flatten() {
            *z += C::new(rng.random() - 0.5, rng.random() - 0.5) * jitter;
        }
        let mut b = Self { geometry, atom };
        b.project()?;
        Ok(b)
    }
    pub fn objects(&self) -> usize {
        self.atom.output.len()
    }
    pub fn dimensions(&self) -> usize {
        4 + 12 * self.objects()
    }
    pub fn category(&self, k: usize) -> Result<usize> {
        if k >= self.dimensions() {
            return Err("coordinate outside complete block".into());
        }
        Ok(if k < 4 {
            0
        } else if k < 4 + 8 * self.objects() {
            1
        } else {
            2
        })
    }
    pub fn responds(&self, k: usize, prefix: &[i64]) -> Result<bool> {
        Ok(self.category(k)? != 1 || prefix.contains(&(((k - 4) / 8) as i64)))
    }
    fn coordinate_mut(&mut self, k: usize) -> Result<&mut f64> {
        self.category(k)?;
        let complex = k / 2;
        let z = if complex < 2 {
            &mut self.atom.initial[complex]
        } else if complex < 2 + 4 * self.objects() {
            let i = complex - 2;
            &mut self.atom.transition[i / 4][(i % 4) / 2][i % 2]
        } else {
            let i = complex - 2 - 4 * self.objects();
            &mut self.atom.output[i / 2][i % 2]
        };
        Ok(if k.is_multiple_of(2) {
            &mut z.re
        } else {
            &mut z.im
        })
    }
    pub fn coordinates(&self) -> Vec<f64> {
        self.atom
            .initial
            .iter()
            .chain(self.atom.transition.iter().flatten().flatten())
            .chain(self.atom.output.iter().flatten())
            .flat_map(|z| [z.re, z.im])
            .collect()
    }
    pub fn perturb(&self, k: usize, delta: f64) -> Result<Self> {
        if !delta.is_finite() {
            return Err("nonfinite coordinate query".into());
        }
        let mut b = self.clone();
        *b.coordinate_mut(k)? += delta;
        match b.category(k)? {
            0 => b.atom.initial = b.geometry.initial(b.atom.initial)?,
            1 => {
                let o = (k - 4) / 8;
                b.atom.transition[o] = b.geometry.matrix(b.atom.transition[o])?;
            }
            _ => {
                let c = (k - 4 - 8 * b.objects()) / 4;
                b.atom.output[c] = b.geometry.output(b.atom.output[c])?;
            }
        }
        Ok(b)
    }
    pub fn project(&mut self) -> Result<()> {
        self.atom.initial = self.geometry.initial(self.atom.initial)?;
        for m in &mut self.atom.transition {
            *m = self.geometry.matrix(*m)?;
        }
        for u in &mut self.atom.output {
            *u = self.geometry.output(*u)?;
        }
        self.validate()?;
        Ok(())
    }
    pub fn update(&mut self, mean: &[f64], rate: f64) -> Result<()> {
        if mean.len() != self.dimensions()
            || !rate.is_finite()
            || rate <= 0.
            || mean.iter().any(|x| !x.is_finite())
        {
            return Err("invalid simultaneous complete update".into());
        }
        let mut next = self.clone();
        for (k, &g) in mean.iter().enumerate() {
            *next.coordinate_mut(k)? -= rate * g;
        }
        next.project()?;
        *self = next;
        Ok(())
    }
    pub fn validate(&self) -> Result<f64> {
        if self.objects() == 0
            || self.atom.transition.len() != self.objects()
            || self.atom.frequency.is_some()
        {
            return Err("incomplete plain block".into());
        }
        geometry::finite(self.atom.initial)?;
        let mut excess = (self.geometry.state_norm(self.atom.initial) - 1.).max(0.);
        for &m in &self.atom.transition {
            geometry::finite(m[0])?;
            geometry::finite(m[1])?;
            excess = excess.max(self.geometry.matrix_norm(m)? - 1.);
        }
        for &u in &self.atom.output {
            geometry::finite(u)?;
            excess = excess.max(self.geometry.row_norm(u) - 1.);
        }
        if excess > 2e-12 {
            return Err("block exceeds its complete geometry".into());
        }
        Ok(excess)
    }
}
