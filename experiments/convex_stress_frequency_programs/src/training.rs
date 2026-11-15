//! Ordinary full-vocabulary CE on CM56's observed-anchor ray. The only free
//! parameter is the shared positive margin; scaled slope/intercept remain
//! linear in it. Real batches retain original IDs and their true denominator.
use crate::{
    coordinates::{Fit, Record},
    head::Head,
    *,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    Forward,
    Full,
    Mini,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub mode: Mode,
    pub iterations: usize,
    pub batch: usize,
    pub learning_rate: f64,
    pub initial_margin: f64,
    pub minimum_margin: f64,
    pub radius: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Forecast {
    pub scalar_work: u64,
    pub buffer_bytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub gradient_batches: usize,
    pub gradient_records: usize,
    pub complete_products: u64,
    pub effective_updates: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub epoch: usize,
    pub record_ids: Vec<usize>,
    pub before_ce: f64,
    pub after_ce: f64,
    pub gradient: f64,
    pub margin: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    pub policy: Policy,
    pub forecast: Forecast,
    pub work: Work,
    pub initial_ce: f64,
    pub final_ce: f64,
    pub steps: Vec<Step>,
    pub head: Head,
}

impl Policy {
    pub fn forecast(&self, n: usize, v: usize) -> Result<Forecast> {
        if !(2..=200_000).contains(&n)
            || !(2..=1024).contains(&v)
            || self.iterations > 1024
            || !(1..=4096).contains(&self.batch)
            || (self.mode == Mode::Forward) != (self.iterations == 0)
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.
            || self.learning_rate > 1e12
            || !self.minimum_margin.is_finite()
            || self.minimum_margin <= 0.
            || !self.initial_margin.is_finite()
            || self.initial_margin < self.minimum_margin
            || !self.radius.is_finite()
            || self.radius <= 0.
            || self.radius > 1e12
        {
            return Err("invalid bounded coordinate CE policy".into());
        }
        let batches = if self.mode == Mode::Mini {
            n.div_ceil(self.batch)
        } else {
            1
        };
        let n = n as u128;
        let v = v as u128;
        let passes = self.iterations as u128;
        let steps = batches as u128 * passes;
        // Fit/revalidation, initial/final scores, complete gradient and post-step
        // products, all class exponentials and complete head reconstructions.
        let scalar =
            (4 * n + 4 * n * passes) * (128 * v + 128) + (steps + 4) * (256 * v + 512) + 4 * v * v;
        // Original IDs in every history step and conservative serialization
        // copies; no caller-owned sequence data are silently included here.
        let bytes = 256 * 1024 * 1024 + 64 * (32 * n + 192 * v + 192 * steps + 16 * n * passes);
        if scalar > 96_000_000_000 || bytes > 56 * 1024 * 1024 * 1024 {
            return Err("coordinate CE work or memory forecast refused".into());
        }
        Ok(Forecast {
            scalar_work: scalar as u64,
            buffer_bytes: bytes as u64,
        })
    }
}

pub fn score(head: &Head, rows: &[Record]) -> f64 {
    assert!(!rows.is_empty());
    rows.iter()
        .map(|r| ce(&head.logits(r.feature), r.label))
        .sum::<f64>()
        / rows.len() as f64
}

/// The derivative head is an auxiliary unit-margin tensor, not a claim that
/// margin1 belongs to the trained model's positive-margin interval.
pub fn gradient(head: &Head, unit: &Head, rows: &[Record]) -> (f64, f64) {
    assert!(!rows.is_empty());
    let (mut loss, mut derivative) = (0., 0.);
    for r in rows {
        let z = head.logits(r.feature);
        let u = unit.logits(r.feature);
        let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<_> = z.iter().map(|x| (x - max).exp()).collect();
        let normalizer: f64 = exps.iter().sum();
        loss += max + normalizer.ln() - z[r.label];
        derivative += exps
            .iter()
            .zip(&u)
            .map(|(p, x)| p * x / normalizer)
            .sum::<f64>()
            - u[r.label];
    }
    (loss / rows.len() as f64, derivative / rows.len() as f64)
}

pub fn train(fit: Fit, rows: &[Record], policy: Policy) -> Result<Outcome> {
    let forecast = policy.forecast(rows.len(), fit.coordinates.len())?;
    fit.validate(rows)?;
    let upper = policy.radius / fit.cost_per_margin();
    if !upper.is_finite() || upper < policy.minimum_margin || policy.initial_margin > upper {
        return Err("empty or violated observed-anchor margin interval".into());
    }
    let unit = Head::new(fit.clone(), 1., fit.cost_per_margin())?;
    let mut head = Head::new(fit, policy.initial_margin, policy.radius)?;
    let initial_ce = score(&head, rows);
    let mut steps = vec![];
    let mut work = Work::default();
    for epoch in 0..policy.iterations {
        let size = if policy.mode == Mode::Mini {
            policy.batch
        } else {
            rows.len()
        };
        for batch in rows.chunks(size) {
            let (before_ce, derivative) = gradient(&head, &unit, batch);
            if !before_ce.is_finite() || !derivative.is_finite() {
                return Err("nonfinite full-vocabulary CE or derivative".into());
            }
            let margin = (head.margin - policy.learning_rate * derivative)
                .clamp(policy.minimum_margin, upper);
            work.effective_updates += usize::from(margin != head.margin);
            head = Head::new(head.fit, margin, policy.radius)?;
            let after_ce = score(&head, batch);
            if !after_ce.is_finite() {
                return Err("nonfinite post-update CE".into());
            }
            steps.push(Step {
                epoch,
                record_ids: batch.iter().map(|r| r.id).collect(),
                before_ce,
                after_ce,
                gradient: derivative,
                margin,
            });
            work.gradient_batches += 1;
            work.gradient_records += batch.len();
        }
    }
    let final_ce = score(&head, rows);
    work.complete_products = 8
        * head.fit.coordinates.len() as u64
        * (2 * rows.len() as u64 + 3 * work.gradient_records as u64);
    Ok(Outcome {
        policy,
        forecast,
        work,
        initial_ce,
        final_ce,
        steps,
        head,
    })
}

pub fn replay(saved: &Outcome, rows: &[Record]) -> Result<()> {
    saved.head.validate()?;
    if *saved != train(saved.head.fit.clone(), rows, saved.policy.clone())? {
        return Err("saved coordinate CE trajectory differs from complete replay".into());
    }
    Ok(())
}
