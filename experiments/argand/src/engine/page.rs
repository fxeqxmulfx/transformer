//! PAGE on a finite sum of observed word losses, with coupled L2 and an
//! observed-objective line search. The rate and penalty are calibrated from
//! the start. The base estimator is the one in Optimization.PAGE.FiniteSum.
use super::{Engine, F, Result, grid};
use crate::pcg::Pcg;
use serde_json::{Value, json};

#[cfg(test)]
mod tests;

/// Padded sequences. Each supervised position is one summand of the mean CE.
#[derive(Clone, Copy)]
pub(crate) struct Observations<'a> {
    pub tokens: &'a [u32],
    pub labels: &'a [u32],
    pub length: usize,
}

impl Observations<'_> {
    fn words(self) -> usize {
        self.tokens.len() / self.length
    }

    fn count(self, engine: &Engine) -> Result<usize> {
        if self.length == 0
            || self.tokens.is_empty()
            || self.tokens.len() != self.labels.len()
            || !self.tokens.len().is_multiple_of(self.length)
            || self.words() > engine.config.max_batch
            || self.tokens.len() > (u32::MAX / 2) as usize
        {
            return Err("invalid padded PAGE reference batch".into());
        }
        let count = engine.labeled(self.labels);
        if count == 0 {
            return Err("the reference batch has no supervised position".into());
        }
        Ok(count)
    }

    /// Copy uniform supervised positions, preserving each causal context.
    /// A row with several targets can contribute several distinct summands.
    fn sample(self, rng: &mut Pcg, n: u32, size: u32, vocab: usize) -> crate::metrics::Packed {
        let mut chosen = Vec::with_capacity(size as usize);
        while chosen.len() < size as usize {
            let i = rng.below(n);
            if !chosen.contains(&i) {
                chosen.push(i);
            }
        }
        chosen.sort_unstable();
        let mut chosen = chosen.into_iter().peekable();
        let mut packed = crate::metrics::Packed {
            length: self.length,
            ..Default::default()
        };
        for (i, (pos, &target)) in self
            .labels
            .iter()
            .enumerate()
            .filter(|(_, y)| **y < vocab as u32)
            .enumerate()
        {
            if chosen.peek() != Some(&(i as u32)) {
                continue;
            }
            chosen.next();
            let start = pos / self.length * self.length;
            packed
                .tokens
                .extend_from_slice(&self.tokens[start..start + self.length]);
            let old_len = packed.labels.len();
            packed.labels.resize(old_len + self.length, u32::MAX);
            packed.labels[old_len + pos % self.length] = target;
            if chosen.peek().is_none() {
                break;
            }
        }
        packed
    }

    fn forward(self, engine: &mut Engine) -> Result<()> {
        engine.forward(self.tokens, self.labels, self.words(), self.length)
    }

    fn mean_loss(self, engine: &Engine, labeled: usize) -> Result<f64> {
        let sum: f64 = engine
            .losses(self.tokens.len())?
            .into_iter()
            .map(f64::from)
            .sum();
        Ok(sum / labeled as f64)
    }
}

impl<'a> From<&'a crate::metrics::Packed> for Observations<'a> {
    fn from(p: &'a crate::metrics::Packed) -> Self {
        Self {
            tokens: &p.tokens,
            labels: &p.labels,
            length: p.length,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PageRule {
    Single,
    SquareRoot,
    Retry,
}

impl PageRule {
    pub fn label(self) -> &'static str {
        match self {
            Self::Single => "one",
            Self::SquareRoot => "ceil-sqrt",
            Self::Retry => "one-retry-previous-rate",
        }
    }
}

fn norm_squared(engine: &Engine, values: &F) -> Result<f64> {
    let norm = engine
        .parameters
        .reduce_norm(&engine.optimizer, values)?
        .read()?[0];
    Ok(f64::from(norm).powi(2))
}

fn regularized_backward(engine: &mut Engine, penalty: f32) -> Result<()> {
    engine.backward()?;
    engine.parameters.add_penalty(&engine.kernel, penalty)
}

pub(crate) struct AdaptivePage {
    correction: PageRule,
    sampler: Pcg,
    saved: F,
    estimate: F,
    old_component: F,
    exact: bool,
    penalty: f32,
    loss: f64,
    objective: f64,
    rate: f32,
    steps: usize,
    refreshes: usize,
    corrections: usize,
    forced_refreshes: usize,
    estimated_retries: usize,
    retries_accepted: usize,
    rejected: usize,
}

impl AdaptivePage {
    pub fn new(
        engine: &mut Engine,
        train: Observations<'_>,
        sampling_seed: u32,
        correction: PageRule,
    ) -> Result<Self> {
        let n = train.count(engine)?;
        train.forward(engine)?;
        let loss = train.mean_loss(engine, n)?;
        let norm = norm_squared(engine, &engine.parameters.value)?;
        // One observation's share of the initial mean CE sets the starting
        // penalty energy. The unit floor also covers an exactly zero start.
        let penalty = (2.0 * loss / (n as f64 * norm.max(1.0))) as f32;
        let objective = loss + 0.5 * f64::from(penalty) * norm;
        if !objective.is_finite() || loss < 0.0 {
            return Err("invalid initial PAGE objective".into());
        }
        regularized_backward(engine, penalty)?;
        let n = engine.parameters.len();
        let mut estimate = engine.cuda.alloc(n)?;
        estimate.copy_from(&engine.parameters.grad)?;
        Ok(Self {
            correction,
            sampler: Pcg::new(&[sampling_seed]),
            saved: engine.cuda.alloc(n)?,
            old_component: engine.cuda.alloc(n)?,
            estimate,
            loss,
            objective,
            penalty,
            exact: true,
            rate: 0.0,
            steps: 0,
            refreshes: 0,
            corrections: 0,
            forced_refreshes: 0,
            estimated_retries: 0,
            retries_accepted: 0,
            rejected: 0,
        })
    }

    fn refresh(&mut self, engine: &mut Engine) -> Result<()> {
        regularized_backward(engine, self.penalty)?;
        self.estimate.copy_from(&engine.parameters.grad)?;
        self.exact = true;
        Ok(())
    }

    fn trial_rate(&self, squared: f64, expand: bool) -> f32 {
        let trial = if self.steps == 0 {
            self.objective / squared
        } else if expand {
            2.0 * f64::from(self.rate)
        } else {
            f64::from(self.rate)
        };
        trial.min(f64::from(f32::MAX)) as f32
    }

    /// A rejected estimated direction is refreshed at the original weights.
    /// All retries start from the same snapshot. A false return means that
    /// the refreshed direction gave no representable objective decrease.
    pub fn step(
        &mut self,
        engine: &mut Engine,
        train: Observations<'_>,
        changed_batch: bool,
    ) -> Result<bool> {
        let n = train.count(engine)? as u32;
        if changed_batch {
            train.forward(engine)?;
            self.loss = train.mean_loss(engine, n as usize)?;
            self.objective = self.loss
                + 0.5 * f64::from(self.penalty) * norm_squared(engine, &engine.parameters.value)?;
            self.exact = false;
        }
        let size = match self.correction {
            PageRule::Single | PageRule::Retry => 1,
            PageRule::SquareRoot => {
                (f64::from(n).sqrt().ceil() as u32).min(engine.config.max_batch as u32)
            }
        };
        let refresh = n == size || self.sampler.below(n + size) < size;
        let packed =
            (!refresh).then(|| train.sample(&mut self.sampler, n, size, engine.config.vocab));
        let sample = packed.as_ref().map(Observations::from);
        if let Some(row) = sample {
            row.forward(engine)?;
            regularized_backward(engine, self.penalty)?;
            self.old_component.copy_from(&engine.parameters.grad)?;
        }
        self.saved.copy_from(&engine.parameters.value)?;
        let mut squared = norm_squared(engine, &self.estimate)?;
        let mut rate = self.trial_rate(squared, true);
        let mut retried = false;
        loop {
            if !squared.is_finite()
                || !rate.is_finite()
                || squared == 0.0
                || rate == 0.0
                || self.objective - 0.5 * f64::from(rate) * squared == self.objective
            {
                engine.parameters.value.copy_from(&self.saved)?;
                if self.exact {
                    if !squared.is_finite() {
                        return Err("nonfinite full PAGE gradient".into());
                    }
                    return Ok(false);
                }
            } else {
                // SAFETY: all three buffers cover the same set of native coordinates.
                unsafe {
                    engine.kernel.page_trial(
                        grid(self.saved.len()),
                        engine.parameters.value.ptr(),
                        self.saved.ptr(),
                        self.estimate.ptr(),
                        self.saved.len() as u32,
                        rate,
                    )?;
                }
                train.forward(engine)?;
                let loss = train.mean_loss(engine, n as usize)?;
                let norm = norm_squared(engine, &engine.parameters.value)?;
                let candidate = loss + 0.5 * f64::from(self.penalty) * norm;
                if candidate.is_finite()
                    && candidate <= self.objective - 0.5 * f64::from(rate) * squared
                {
                    self.loss = loss;
                    self.objective = candidate;
                    self.rate = rate;
                    self.retries_accepted += usize::from(retried && !self.exact);
                    break;
                }
                self.rejected += 1;
                if self.exact {
                    rate *= 0.5;
                    continue;
                }
                // A rejected speculative doubling can still descend at the
                // previously accepted rate. Each trial reads the same snapshot.
                if self.correction == PageRule::Retry && rate > self.rate && self.rate > 0.0 {
                    rate = self.rate;
                    retried = true;
                    self.estimated_retries += 1;
                    continue;
                }
                engine.parameters.value.copy_from(&self.saved)?;
            }
            train.forward(engine)?;
            self.refresh(engine)?;
            self.forced_refreshes += 1;
            squared = norm_squared(engine, &self.estimate)?;
            rate = self.trial_rate(squared, false);
        }
        if let Some(row) = sample {
            row.forward(engine)?;
            regularized_backward(engine, self.penalty)?;
            // SAFETY: the two gradients and the persistent estimate have equal lengths.
            unsafe {
                engine.kernel.page_correct(
                    grid(self.saved.len()),
                    self.estimate.ptr(),
                    engine.parameters.grad.ptr(),
                    self.old_component.ptr(),
                    self.saved.len() as u32,
                )?;
            }
            self.exact = false;
            self.corrections += 1;
        } else {
            // The accepted full observed forward pass is still on the tape.
            self.refresh(engine)?;
            self.refreshes += 1;
        }
        self.steps += 1;
        Ok(true)
    }

    pub fn metrics(&self) -> Value {
        json!({"step": self.steps, "observed_ce": self.loss, "objective": self.objective,
            "rate": self.rate, "penalty": self.penalty, "scheduled_refreshes": self.refreshes,
            "forced_refreshes": self.forced_refreshes, "corrections": self.corrections,
            "estimated_retries":self.estimated_retries,"retries_accepted":self.retries_accepted,
            "rejected_trials": self.rejected})
    }
}
