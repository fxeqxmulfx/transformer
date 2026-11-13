//! Actual complex Fourier features with one real/imaginary coefficient tensor.
//! CFt4 fixes categorical inputs; CD8 applies with features (Re phi,-Im phi).
//! AddReadoutSpectrum proves F*F=N I. Native audits inspect the saved code basis,
//! not a claim that the empirical task matrix has a flat spectrum.
use crate::*;
mod audit;
mod initialize;
#[cfg(test)]
mod tests;
pub use audit::Audit;
pub use initialize::{Initialization, check as check_initialization, initialize};
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub query: usize,
    pub decoder: discovery::Decoder,
    pub first_row: usize,
    pub code: Vec<C>,
}
impl Block {
    pub fn states(&self) -> usize {
        self.decoder.values.len() + 1
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub vocabulary: usize,
    pub context: usize,
    pub blocks: Vec<Block>,
    /// Rows alternate real and imaginary coordinates of the complex head.
    pub coefficients: Vec<f64>,
    pub radius: f64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Encoded {
    pub block: usize,
    pub state: usize,
    pub label: usize,
}
pub(crate) fn phase(states: usize, state: usize, frequency: usize) -> C {
    C::from_polar(
        1. / (states as f64).sqrt(),
        std::f64::consts::TAU * ((state * frequency) % states) as f64 / states as f64,
    )
}
pub fn memory_bound(vocabulary: usize, states: usize, phase_entries: usize) -> Result<u64> {
    if !(2..=1024).contains(&vocabulary)
        || !(1..=65536).contains(&states)
        || phase_entries > 4_000_000
        || 2 * states * vocabulary > 16_000_000
    {
        return Err("complex head exceeds state/phase/coefficient caps".into());
    }
    Ok(512 * 1024 * 1024 + 128 * phase_entries as u64 + 128 * (2 * states * vocabulary) as u64)
}
impl Model {
    pub fn validate(&self) -> Result<()> {
        if !(2..=1024).contains(&self.vocabulary)
            || !(1..=4096).contains(&self.context)
            || self.blocks.is_empty()
            || self.blocks.len() > self.vocabulary
        {
            return Err("invalid complex model dimensions".into());
        }
        let keys = features::library(self.vocabulary, self.context).len();
        let (mut rows, mut entries, mut prior) = (0, 0, None);
        for b in &self.blocks {
            let n = b.states();
            if b.query >= self.vocabulary
                || prior.is_some_and(|p| p >= b.query)
                || b.first_row != rows
                || !(2..=513).contains(&n)
                || b.decoder.labels.len() + 1 != n
                || b.decoder.labels.iter().any(|&c| c >= self.vocabulary)
                || b.decoder.values.windows(2).any(|w| w[0] >= w[1])
                || !b.decoder.candidate.valid(keys)
                || b.code.len() != n * n
            {
                return Err("invalid saved query/code dictionary".into());
            }
            for state in 0..n {
                for f in 0..n {
                    if b.code[state * n + f] != phase(n, state, f) {
                        return Err(
                            "saved complex features differ from fixed categorical Fourier code"
                                .into(),
                        );
                    }
                }
            }
            prior = Some(b.query);
            rows += 2 * n;
            entries += n * n;
        }
        memory_bound(self.vocabulary, rows / 2, entries)?;
        if self.coefficients.len() != rows * self.vocabulary || self.radius != 32. * rows as f64 {
            return Err("complex head shape or fixed radius differs".into());
        }
        self.check_coefficients()
    }
    pub(crate) fn check_coefficients(&self) -> Result<()> {
        if self.coefficients.iter().any(|x| !x.is_finite())
            || (0..self.vocabulary).any(|c| {
                self.coefficients
                    .iter()
                    .skip(c)
                    .step_by(self.vocabulary)
                    .map(|x| x.abs())
                    .sum::<f64>()
                    > self.radius + 1e-8
            })
        {
            return Err("complex coefficient tensor exceeds finite per-class L1 budget".into());
        }
        Ok(())
    }
    fn location(&self, tokens: &[usize]) -> Option<(usize, usize, bool)> {
        assert!(
            !tokens.is_empty()
                && tokens.len() <= 4096
                && tokens.iter().all(|&t| t < self.vocabulary)
        );
        let block = self
            .blocks
            .binary_search_by_key(tokens.last().unwrap(), |b| b.query)
            .ok()?;
        let b = &self.blocks[block];
        let library = features::library(self.vocabulary, self.context);
        let value = b
            .decoder
            .candidate
            .original(&library, self.vocabulary, tokens);
        let state = b.decoder.state(value);
        Some((block, state.unwrap_or(b.states() - 1), state.is_some()))
    }
    pub fn known(&self, tokens: &[usize]) -> bool {
        self.location(tokens).is_some_and(|p| p.2)
    }
    pub fn logits_tokens(&self, tokens: &[usize]) -> Vec<f64> {
        self.location(tokens).map_or_else(
            || vec![0.; self.vocabulary],
            |(b, s, _)| self.logits_state(b, s),
        )
    }
    pub fn logits_state(&self, block: usize, state: usize) -> Vec<f64> {
        let b = &self.blocks[block];
        let n = b.states();
        assert!(state < n);
        let mut logits = vec![0.; self.vocabulary];
        for f in 0..n {
            let phi = b.code[state * n + f];
            let row = (b.first_row + 2 * f) * self.vocabulary;
            for (c, z) in logits.iter_mut().enumerate() {
                *z += self.coefficients[row + c] * phi.re
                    - self.coefficients[row + self.vocabulary + c] * phi.im;
            }
        }
        logits
    }
    pub fn observations(&self, input: &Input) -> Result<Vec<Encoded>> {
        self.validate()?;
        if input.vocabulary != self.vocabulary
            || input.context != self.context
            || input.observations.is_empty()
        {
            return Err("original input shape differs from complex model".into());
        }
        input
            .observations
            .iter()
            .map(|s| {
                if s.tokens.is_empty()
                    || s.tokens.len() > self.context
                    || s.label >= self.vocabulary
                    || s.tokens.iter().any(|&t| t >= self.vocabulary)
                {
                    return Err("invalid original head observation".into());
                }
                let (block, state, known) = self
                    .location(&s.tokens)
                    .ok_or("original query absent from model")?;
                if !known {
                    return Err("original feature value absent from model".into());
                }
                Ok(Encoded {
                    block,
                    state,
                    label: s.label,
                })
            })
            .collect()
    }
}
impl metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        assert!(prefix.iter().all(|&x| x >= 0));
        self.logits_tokens(&prefix.iter().map(|&x| x as usize).collect::<Vec<_>>())
    }
}
