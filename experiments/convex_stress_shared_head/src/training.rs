//! Ordinary CE differentiates every complete real/imaginary head coordinate.
//! Features and source routes are fixed. Final iterates and CE increases remain.
use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    Full,
    Mini,
    Forward,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub mode: Mode,
    pub iterations: usize,
    pub batch: usize,
    pub learning_rate: f64,
}
impl Policy {
    pub fn forward() -> Self {
        Self {
            mode: Mode::Forward,
            iterations: 0,
            batch: 127,
            learning_rate: 1.,
        }
    }
    pub fn validate(&self) -> Result<()> {
        if !(1..=1024).contains(&self.batch)
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.
            || self.learning_rate > 16.
            || match self.mode {
                Mode::Full => !(1..=128).contains(&self.iterations),
                Mode::Mini => !(1..=8).contains(&self.iterations),
                Mode::Forward => self.iterations != 0,
            }
        {
            return Err("invalid shared-head CE policy".into());
        }
        Ok(())
    }
}
pub fn policies() -> Result<Vec<Policy>> {
    Ok(serde_json::from_value(
        serde_json::from_str::<serde_json::Value>(PROTOCOL)?["training"].clone(),
    )?)
}
pub struct Encoded {
    pub block: usize,
    pub phi: [C; STATES],
    pub label: usize,
}
pub fn encode(input: &Input) -> Result<Vec<Encoded>> {
    input.validate()?;
    Ok(input
        .records
        .iter()
        .map(|r| Encoded {
            block: input.route[r.cell],
            phi: features::encode(input.n, r.cell),
            label: r.label,
        })
        .collect())
}
pub fn objective(model: &Model, rows: &[Encoded], batch: &[usize]) -> Result<f64> {
    if batch.is_empty() || batch.iter().any(|&i| i >= rows.len()) {
        return Err("invalid CE batch".into());
    }
    let mut loss = 0.;
    for &i in batch {
        let r = &rows[i];
        loss += cross_entropy(&model.logits_encoded(r.block, &r.phi), r.label);
    }
    Ok(loss / batch.len() as f64)
}
pub fn gradient(model: &Model, rows: &[Encoded], batch: &[usize]) -> Result<(f64, Vec<f64>)> {
    if batch.is_empty() || batch.iter().any(|&i| i >= rows.len()) {
        return Err("invalid derivative batch".into());
    }
    let v = model.input.vocabulary;
    let mut grad = vec![0.; model.coefficients.len()];
    let mut residual = vec![0.; v];
    let mut loss = 0.;
    for &i in batch {
        let r = &rows[i];
        let z = model.logits_encoded(r.block, &r.phi);
        loss += ce_gradient(&z, r.label, &mut residual);
        for (f, p) in r.phi.iter().enumerate() {
            let at = (r.block * WIDTH + 2 * f) * v;
            for (c, &value) in residual.iter().enumerate() {
                grad[at + c] += p.re * value / batch.len() as f64;
                grad[at + v + c] -= p.im * value / batch.len() as f64;
            }
        }
    }
    if grad.iter().any(|x| !x.is_finite()) || !loss.is_finite() {
        return Err("nonfinite ordinary CE gradient".into());
    }
    Ok((loss / batch.len() as f64, grad))
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub gradient_batches: usize,
    pub gradient_examples: usize,
    pub updates: usize,
    pub effective_updates: usize,
    pub ce_evaluations: usize,
    pub ce_examples: usize,
    pub projected_coefficients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub epoch: usize,
    pub record_ids: Vec<usize>,
    pub batch_ce_before: f64,
    pub gradient_l2: f64,
    pub change_l2: f64,
    pub maximum_class_l1: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    pub model: Model,
    pub input_sha256: String,
    pub initial_model_sha256: String,
    pub policy: Policy,
    pub initialization: initialize::Report,
    pub bounds: limits::Bounds,
    pub initial_ce: f64,
    pub final_ce: f64,
    pub epoch_ce: Vec<f64>,
    pub steps: Vec<Step>,
    pub work: Work,
}
pub fn run(input: Input, policy: Policy) -> Result<Outcome> {
    let bounds = limits::forecast(&input, &policy)?;
    let input_sha256 = digest(&input)?;
    let rows = encode(&input)?;
    let all: Vec<_> = (0..rows.len()).collect();
    let (mut model, initialization) = initialize::run(input)?;
    let initial_model_sha256 = digest(&model)?;
    let initial_ce = objective(&model, &rows, &all)?;
    let mut epoch_ce = vec![initial_ce];
    let mut steps = vec![];
    let mut work = Work {
        ce_evaluations: 1,
        ce_examples: rows.len(),
        ..Work::default()
    };
    for epoch in 0..policy.iterations {
        let width = if policy.mode == Mode::Full {
            all.len()
        } else {
            policy.batch
        };
        for batch in all.chunks(width) {
            let (loss, g) = gradient(&model, &rows, batch)?;
            let before = model.coefficients.clone();
            for (x, g) in model.coefficients.iter_mut().zip(&g) {
                *x -= policy.learning_rate * g;
            }
            model.project();
            let mass = model.validate()?;
            let change = model
                .coefficients
                .iter()
                .zip(&before)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt();
            steps.push(Step {
                epoch,
                record_ids: batch.iter().map(|&i| model.input.records[i].id).collect(),
                batch_ce_before: loss,
                gradient_l2: g.iter().map(|x| x * x).sum::<f64>().sqrt(),
                change_l2: change,
                maximum_class_l1: mass,
            });
            work.gradient_batches += 1;
            work.gradient_examples += batch.len();
            work.updates += 1;
            work.effective_updates += usize::from(change > 0.);
            work.projected_coefficients += model.coefficients.len();
        }
        epoch_ce.push(objective(&model, &rows, &all)?);
        work.ce_evaluations += 1;
        work.ce_examples += rows.len();
    }
    if work.gradient_batches != bounds.gradient_batches {
        return Err("gradient work differs from forecast".into());
    }
    Ok(Outcome {
        model,
        input_sha256,
        initial_model_sha256,
        policy,
        initialization,
        bounds,
        initial_ce,
        final_ce: *epoch_ce.last().unwrap(),
        epoch_ce,
        steps,
        work,
    })
}
pub fn replay(input: &Input, outcome: &Outcome) -> Result<()> {
    if input != &outcome.model.input || &run(input.clone(), outcome.policy.clone())? != outcome {
        return Err("shared head, records, CE history or work changed".into());
    }
    Ok(())
}
