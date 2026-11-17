//! CM92/CM93: execute every learned factor under a positive capacity geometry.
use crate::{C, Result};
use bilinear::domain::{Addresses, Shape};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Atom {
    pub output: Vec<C>,
    pub query: Vec<C>,
    pub context: Vec<C>,
}

impl Atom {
    pub fn zero(shape: Shape) -> Self {
        Self {
            output: vec![C::default(); shape.objects],
            query: vec![C::default(); shape.queries],
            context: vec![C::default(); shape.columns],
        }
    }

    pub fn bias(shape: Shape, class: usize, sign: f64) -> Result<Self> {
        if class >= shape.objects || !matches!(sign, -1. | 1.) {
            return Err("invalid signed class-bias component".into());
        }
        let mut atom = Self::zero(shape);
        atom.output[class] = C::new(sign, 0.);
        atom.query[0] = C::new(1., 0.);
        atom.context[0] = C::new(1., 0.);
        Ok(atom)
    }

    pub fn validate(&self, shape: Shape) -> Result<f64> {
        let mut maximum_excess = 0f64;
        for (vector, expected) in [
            (&self.output, shape.objects),
            (&self.query, shape.queries),
            (&self.context, shape.columns),
        ] {
            if vector.len() != expected
                || vector
                    .iter()
                    .any(|z| !z.re.is_finite() || !z.im.is_finite())
            {
                return Err("incomplete or nonfinite complex factor".into());
            }
            let size: f64 = vector.iter().map(C::norm_sqr).sum();
            if !size.is_finite() || size > 1. + 2e-12 {
                return Err("component exceeds unit capacity".into());
            }
            maximum_excess = maximum_excess.max(size - 1.);
        }
        Ok(maximum_excess)
    }

    pub fn complex(&self, input: &Addresses) -> Vec<C> {
        let q: C = input.queries.iter().map(|&i| self.query[i]).sum();
        let d: C = input
            .columns
            .iter()
            .map(|&(i, n)| self.context[i] * n as f64)
            .sum();
        let interaction = C::new(1., 1.) * q * d;
        self.output.iter().map(|&u| u * interaction).collect()
    }

    pub fn scaled_complex(&self, input: &Addresses, query: &[f64], context: &[f64]) -> Vec<C> {
        let q: C = input
            .queries
            .iter()
            .map(|&i| self.query[i] * query[i])
            .sum();
        let d: C = input
            .columns
            .iter()
            .map(|&(i, n)| self.context[i] * (context[i] * n as f64))
            .sum();
        let interaction = C::new(1., 1.) * q * d;
        self.output.iter().map(|&u| u * interaction).collect()
    }

    pub fn entry(&self, c: usize, q: usize, d: usize) -> C {
        self.output[c] * self.query[q] * self.context[d]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub shape: Shape,
    pub budget: f64,
    pub query_scale: Vec<f64>,
    pub context_scale: Vec<f64>,
    pub atoms: Vec<Atom>,
    pub weights: Vec<f64>,
}

impl Model {
    pub fn zero(shape: Shape, budget: f64) -> Result<Self> {
        if !budget.is_finite() || budget <= 0. {
            return Err("nonpositive finite capacity budget".into());
        }
        Ok(Self {
            shape,
            budget,
            query_scale: vec![1.; shape.queries],
            context_scale: vec![1.; shape.columns],
            atoms: vec![Atom::zero(shape)],
            weights: vec![budget],
        })
    }

    /// CM93: dimension-only positive geometry; every learned entry is retained.
    pub fn balanced(shape: Shape, budget: f64) -> Result<Self> {
        let mut model = Self::zero(shape, budget)?;
        let scale = 1. / (shape.objects as f64).sqrt();
        model.query_scale[0] = scale;
        model.context_scale[0] = scale;
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<f64> {
        for (scales, expected) in [
            (&self.query_scale, self.shape.queries),
            (&self.context_scale, self.shape.columns),
        ] {
            if scales.len() != expected
                || scales.is_empty()
                || scales.iter().any(|s| !s.is_finite() || *s <= 0. || *s > 1.)
            {
                return Err("incomplete or nonpositive coordinate geometry".into());
            }
        }
        let minimum_query = self.query_scale.iter().copied().fold(1f64, f64::min);
        let minimum_context = self.context_scale.iter().copied().fold(1f64, f64::min);
        if minimum_query * minimum_context == 0. {
            return Err("joint geometry underflow removes a coordinate".into());
        }
        if self.atoms.is_empty()
            || self.atoms.len() != self.weights.len()
            || !self.budget.is_finite()
            || self.budget <= 0.
            || self.weights.iter().any(|w| !w.is_finite() || *w < 0.)
        {
            return Err("invalid complete mixture state".into());
        }
        // Slot zero is the actual zero component carrying unused capacity.
        if self.atoms[0]
            .output
            .iter()
            .chain(&self.atoms[0].query)
            .chain(&self.atoms[0].context)
            .any(|&x| x != C::default())
        {
            return Err("unused capacity component is not exactly zero".into());
        }
        let total: f64 = self.weights.iter().sum();
        if (total - self.budget).abs() > 2e-10 * self.budget {
            return Err("mixture mass differs from declared capacity".into());
        }
        self.atoms
            .iter()
            .try_fold(0f64, |e, f| Ok(e.max(f.validate(self.shape)?)))
    }

    pub fn complex(&self, input: &Addresses) -> Vec<C> {
        let mut scores = vec![C::default(); self.shape.objects];
        for (atom, &weight) in self.atoms.iter().zip(&self.weights) {
            if weight == 0. {
                continue;
            } // Exact zero mass only; no small-weight threshold.
            for (score, value) in scores.iter_mut().zip(atom.scaled_complex(
                input,
                &self.query_scale,
                &self.context_scale,
            )) {
                *score += value * weight;
            }
        }
        scores
    }

    pub fn logits(&self, prefix: &[i64]) -> Result<Vec<f64>> {
        Ok(self
            .complex(&self.shape.addresses(prefix)?)
            .iter()
            .map(|z| z.re)
            .collect())
    }

    /// Output is one original immutable object ID, with the lowest-ID tie rule.
    pub fn next_object(&self, prefix: &[i64]) -> Result<usize> {
        let scores = self.logits(prefix)?;
        Ok((1..scores.len()).fold(0, |best, c| if scores[c] > scores[best] { c } else { best }))
    }

    /// CM93 accounts for the actual scaled bias response in raw logit units.
    pub fn query_mass(&self, h: f64) -> Result<f64> {
        let mass = h / (self.query_scale[0] * self.context_scale[0]);
        if !h.is_finite() || h <= 0. || !mass.is_finite() || mass <= 0. || mass > self.budget {
            return Err("loss-query step exceeds scaled capacity".into());
        }
        Ok(mass)
    }

    /// CM92/CM93 leave room for both signed loss queries. The zero component
    /// retains exactly the unused mass, without changing any class score.
    pub fn shrink_for_queries(&mut self, h: f64) -> Result<()> {
        let mass = self.query_mass(h)?;
        let eta = mass / self.budget;
        for weight in &mut self.weights {
            *weight *= 1. - eta;
        }
        self.weights[0] += mass;
        self.validate()?;
        Ok(())
    }

    pub fn mix(&mut self, atom: Atom, eta: f64) -> Result<()> {
        if !eta.is_finite() || !(0. ..=1.).contains(&eta) {
            return Err("convex step outside unit interval".into());
        }
        atom.validate(self.shape)?;
        for weight in &mut self.weights {
            *weight *= 1. - eta;
        }
        self.atoms.push(atom);
        self.weights.push(eta * self.budget);
        self.validate()?;
        Ok(())
    }

    pub fn reservation(&self, batch_records: usize) -> Result<u64> {
        let s = self.shape;
        let coordinates = s.objects as u128 * s.queries as u128 * s.columns as u128;
        let factors = (s.objects + s.queries + s.columns) as u128;
        let bytes = 32 * coordinates
            + 128 * (self.atoms.len() as u128 + 8) * factors
            + 128 * batch_records as u128 * s.objects as u128;
        u64::try_from(bytes).map_err(Into::into)
    }
}
