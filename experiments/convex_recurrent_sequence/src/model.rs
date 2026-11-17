//! CM95 Scalar.run_product, domain_bound and CM96 complete scalar representation.
//! Every original object has a learned transition and readout, both complex.
use crate::{C, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Atom {
    pub initial: C,
    pub transition: Vec<C>,
    pub output: Vec<C>,
}

impl Atom {
    pub fn zero(objects: usize) -> Self {
        Self {
            initial: C::default(),
            transition: vec![C::default(); objects],
            output: vec![C::default(); objects],
        }
    }

    pub fn identity_start(objects: usize) -> Self {
        Self {
            initial: C::new(1., 0.),
            transition: vec![C::new(1., 0.); objects],
            output: vec![C::default(); objects],
        }
    }

    pub fn bias(objects: usize, class: usize, sign: f64) -> Result<Self> {
        if class >= objects || !matches!(sign, -1. | 1.) {
            return Err("invalid signed original-class query".into());
        }
        let mut atom = Self::identity_start(objects);
        atom.output[class] = C::new(sign, 0.);
        Ok(atom)
    }

    pub fn validate(&self, objects: usize) -> Result<f64> {
        if objects == 0 || self.transition.len() != objects || self.output.len() != objects {
            return Err("incomplete original-object parameters".into());
        }
        let mut excess = 0f64;
        for z in std::iter::once(&self.initial)
            .chain(&self.transition)
            .chain(&self.output)
        {
            let norm = z.norm();
            if !z.re.is_finite() || !z.im.is_finite() || norm > 1. + 2e-12 {
                return Err("nonfinite or unbounded scalar operator".into());
            }
            excess = excess.max(norm - 1.);
        }
        Ok(excess)
    }

    /// Execute every occurrence in its original order, including answer objects.
    /// Commutativity is a mathematical limitation of this scalar state family.
    pub fn state(&self, prefix: &[i64]) -> Result<C> {
        let mut state = self.initial;
        for &object in prefix {
            let index = usize::try_from(object)?;
            state *= *self
                .transition
                .get(index)
                .ok_or("object outside vocabulary")?;
        }
        if !state.re.is_finite() || !state.im.is_finite() {
            return Err("nonfinite executed recurrent state".into());
        }
        Ok(state)
    }

    pub fn complex(&self, prefix: &[i64]) -> Result<Vec<C>> {
        let state = self.state(prefix)?;
        Ok(self.output.iter().map(|&u| u * state).collect())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub objects: usize,
    pub budget: f64,
    pub atoms: Vec<Atom>,
    pub weights: Vec<f64>,
}

impl Model {
    pub fn zero(objects: usize, budget: f64) -> Result<Self> {
        let result = Self {
            objects,
            budget,
            atoms: vec![Atom::zero(objects)],
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
        if self.atoms[0].initial != C::default()
            || self.atoms[0]
                .transition
                .iter()
                .chain(&self.atoms[0].output)
                .any(|&z| z != C::default())
        {
            return Err("unused-capacity component changed".into());
        }
        if (self.weights.iter().sum::<f64>() - self.budget).abs() > 2e-10 * self.budget {
            return Err("mixture mass differs from fixed capacity".into());
        }
        self.atoms
            .iter()
            .try_fold(0f64, |e, a| Ok(e.max(a.validate(self.objects)?)))
    }

    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        // Validate IDs even for the initial all-zero mixture.
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
                *score += weight * (u * state).re;
            }
        }
        Ok(logits)
    }

    pub fn next_object(&self, prefix: &[i64]) -> Result<usize> {
        let logits = self.logits(prefix)?;
        Ok((1..logits.len()).fold(0, |best, c| if logits[c] > logits[best] { c } else { best }))
    }

    /// Same feasible centre in both modes; signed unit bias has raw response one.
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
        atom.validate(self.objects)?;
        for weight in &mut self.weights {
            *weight *= 1. - eta;
        }
        self.atoms.push(atom);
        self.weights.push(eta * self.budget);
        self.validate()?;
        Ok(())
    }

    pub fn reservation(&self, records: usize) -> Result<u64> {
        let bytes = 128u128 * (self.atoms.len() as u128 + 16) * (2 * self.objects as u128 + 1)
            + 256u128 * records as u128 * self.objects as u128;
        u64::try_from(bytes).map_err(Into::into)
    }
}
