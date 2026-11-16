//! CM75 recovered physical computation and CM78 shared paired computation.
//! All learned representations and readouts are evaluated. Binary64 factors
//! approximate exact rational parameters; arithmetic failures are explicit.

use super::query::ComponentQuery;
use super::{Complex64, ComplexModel};
use crate::{Rational, Refusal};
use num_traits::{ToPrimitive, Zero};

fn finite(value: Complex64) -> Result<Complex64, Refusal> {
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(Refusal::NonFinite);
    }
    Ok(value)
}

fn approximate(value: &Rational) -> Result<f64, Refusal> {
    let converted = value.to_f64().ok_or(Refusal::NonFinite)?;
    if !converted.is_finite() {
        return Err(Refusal::NonFinite);
    }
    if converted == 0.0 && !value.is_zero() {
        return Err(Refusal::NumericalUnderflow);
    }
    Ok(converted)
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecoveredNeuron {
    pub amplitude: f64,
    pub representation: Vec<Complex64>,
}

impl RecoveredNeuron {
    pub(super) fn new(real: &[Rational], imag: &[Rational]) -> Result<Self, Refusal> {
        if real.len() != imag.len() {
            return Err(Refusal::Shape);
        }
        let mut weights = Vec::with_capacity(real.len());
        let mut norm = 0.0_f64;
        for (r, i) in real.iter().zip(imag) {
            let value = Complex64::new(approximate(r)?, approximate(i)?);
            let magnitude = value.re.hypot(value.im);
            if !magnitude.is_finite() {
                return Err(Refusal::NonFinite);
            }
            norm = norm.max(magnitude);
            weights.push(value);
        }
        // The Lean function-space norm is max coordinate complex magnitude.
        // It is distinct from the joint Euclidean projection metric.
        let amplitude = norm.sqrt();
        if amplitude == 0.0 {
            return Ok(Self {
                amplitude,
                representation: weights,
            });
        }
        for value in &mut weights {
            let old = *value;
            *value = finite(old / amplitude)?;
            if (old.re != 0.0 && value.re == 0.0) || (old.im != 0.0 && value.im == 0.0) {
                return Err(Refusal::NumericalUnderflow);
            }
        }
        Ok(Self {
            amplitude,
            representation: weights,
        })
    }

    fn evaluate(&self, tokens: &[usize], objects: usize) -> Result<Complex64, Refusal> {
        let mut preactivation = self.representation[0];
        for (position, &object) in tokens.iter().enumerate() {
            preactivation =
                finite(preactivation + self.representation[1 + position * objects + object])?;
        }
        let activation = Complex64::new(preactivation.re.max(0.0), preactivation.im.max(0.0));
        let readout = Complex64::new(self.amplitude, self.amplitude);
        finite(readout * activation)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Renaming {
    permutation: Vec<usize>,
}

impl Renaming {
    pub fn new(permutation: Vec<usize>) -> Result<Self, Refusal> {
        if permutation.is_empty()
            || permutation
                .iter()
                .enumerate()
                .any(|(i, &j)| j >= permutation.len() || permutation[j] != i)
        {
            return Err(Refusal::InvalidRenaming);
        }
        Ok(Self { permutation })
    }
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    complex_scores: Vec<Complex64>,
    probabilities: Vec<f64>,
    object: usize,
    max_logit: f64,
    log_shifted_sum: f64,
}

impl Output {
    pub fn complex_scores(&self) -> &[Complex64] {
        &self.complex_scores
    }
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }
    pub fn object(&self) -> usize {
        self.object
    }

    fn from_scores(complex_scores: Vec<Complex64>) -> Result<Self, Refusal> {
        if complex_scores.is_empty() {
            return Err(Refusal::EmptyObjects);
        }
        let mut object = 0;
        for (i, &score) in complex_scores.iter().enumerate() {
            finite(score)?;
            if score.re > complex_scores[object].re {
                object = i;
            }
        }
        let max_logit = complex_scores[object].re;
        let mut probabilities = Vec::with_capacity(complex_scores.len());
        for score in &complex_scores {
            let delta = score.re - max_logit;
            if !delta.is_finite() {
                return Err(Refusal::NonFinite);
            }
            let value = delta.exp();
            if value == 0.0 {
                return Err(Refusal::NumericalUnderflow);
            }
            probabilities.push(value);
        }
        let sum: f64 = probabilities.iter().sum();
        if !sum.is_finite() {
            return Err(Refusal::NonFinite);
        }
        for value in &mut probabilities {
            *value /= sum;
            if *value == 0.0 {
                return Err(Refusal::NumericalUnderflow);
            }
        }
        Ok(Self {
            complex_scores,
            probabilities,
            object,
            max_logit,
            log_shifted_sum: sum.ln(),
        })
    }

    /// Ordinary full-vocabulary CE. This is the only forward-layer method
    /// accepting a target; inference and factor recovery never receive one.
    pub fn cross_entropy(&self, target: usize) -> Result<f64, Refusal> {
        let target_score = self
            .complex_scores
            .get(target)
            .ok_or(Refusal::InvalidObject)?
            .re;
        let loss = (self.max_logit - target_score) + self.log_shifted_sum;
        if !loss.is_finite() || loss < 0.0 {
            return Err(Refusal::NonFinite);
        }
        Ok(loss)
    }
}

impl ComplexModel {
    fn scores(
        &self,
        tokens: &[usize],
        query: Option<&ComponentQuery<'_>>,
    ) -> Result<Vec<Complex64>, Refusal> {
        self.domain.input_index(tokens)?;
        let plan = self.domain.plan();
        let mut scores = vec![Complex64::new(0.0, 0.0); plan.objects];
        for mask in 0..plan.patterns {
            for (class, score) in scores.iter_mut().enumerate() {
                let index = (mask * plan.objects + class) * 2;
                let neuron = |i| match query {
                    Some(query) => query.neuron(i),
                    None => &self.recovered[i],
                };
                let positive = neuron(index).evaluate(tokens, plan.objects)?;
                let negative = neuron(index + 1).evaluate(tokens, plan.objects)?;
                *score = finite(*score + positive - negative)?;
            }
        }
        Ok(scores)
    }

    pub fn forward(&self, tokens: &[usize]) -> Result<Output, Refusal> {
        Output::from_scores(self.scores(tokens, None)?)
    }

    pub fn paired_forward(&self, tokens: &[usize], renaming: &Renaming) -> Result<Output, Refusal> {
        self.paired_output(tokens, renaming, None)
    }

    fn paired_output(
        &self,
        tokens: &[usize],
        renaming: &Renaming,
        query: Option<&ComponentQuery<'_>>,
    ) -> Result<Output, Refusal> {
        if renaming.permutation.len() != self.domain.plan().objects {
            return Err(Refusal::InvalidRenaming);
        }
        let mut first = self.scores(tokens, query)?;
        let renamed: Vec<_> = tokens
            .iter()
            .map(|&object| renaming.permutation[object])
            .collect();
        let second = self.scores(&renamed, query)?;
        for (class, score) in first.iter_mut().enumerate() {
            *score = finite((*score + second[renaming.permutation[class]]) * 0.5)?;
        }
        Output::from_scores(first)
    }
}

impl ComponentQuery<'_> {
    pub(super) fn forward(&self, tokens: &[usize]) -> Result<Output, Refusal> {
        Output::from_scores(self.model().scores(tokens, Some(self))?)
    }

    pub(super) fn paired_forward(
        &self,
        tokens: &[usize],
        renaming: &Renaming,
    ) -> Result<Output, Refusal> {
        self.model().paired_output(tokens, renaming, Some(self))
    }
}
