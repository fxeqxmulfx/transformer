//! CM103 at6383026a: complete conditional CE gap on already learned states.
//! Floating-point diagnostics, without a rounding certificate or training update.
use bilinear::{
    model::cross_entropy,
    train::{Mode, Record},
};
pub use rotary::{C, Result};
use rotary::{
    atom::{State, row},
    model::Model,
};
use serde::Serialize;
pub mod experiment;
pub mod probes;
pub mod source;
#[cfg(test)]
mod tests;

/// Explicit CM103 bestRow: both complex coefficients participate, ties first.
pub fn best_row(a: State) -> Result<State> {
    let r = [a[0].norm(), a[1].norm()];
    if r.iter().any(|x| !x.is_finite()) || (0..2).any(|k| r[k] == 0. && a[k] != C::default()) {
        return Err("nonfinite or underflowed complete coefficient".into());
    }
    let k = usize::from(r[1] > r[0]);
    let mut u = [C::default(); 2];
    if r[k] != 0. {
        u[k] = -a[k].conj() / r[k];
    }
    Ok(u)
}

#[derive(Debug, Serialize)]
pub struct Summary {
    pub records: usize,
    pub classes: usize,
    pub components: usize,
    pub real_coordinates: usize,
    pub mean_ce: f64,
    pub correct: usize,
    pub linear_cost: f64,
    pub support_size: f64,
    pub approximate_gap: f64,
    pub approximate_ce_lower_bound: f64,
    pub predicted_change: f64,
    pub support_identity_error: f64,
    pub change_identity_error: f64,
    pub maximum_actual_logit_error: f64,
    pub maximum_operator_norm_excess: f64,
    pub ce_evaluations: usize,
}
pub struct Analysis {
    pub coefficient: Vec<Vec<State>>,
    pub best: Model,
    pub summary: Summary,
}
/// An immutable model reference prevents a change of centre during accumulation.
pub struct Scan<'a> {
    model: &'a Model,
    coefficient: Vec<Vec<State>>,
    records: usize,
    ce: f64,
    correct: usize,
    logit_error: f64,
}
impl<'a> Scan<'a> {
    pub fn new(model: &'a Model) -> Result<Self> {
        model.validate()?;
        Ok(Self {
            model,
            coefficient: vec![vec![[C::default(); 2]; model.objects]; model.atoms.len()],
            records: 0,
            ce: 0.,
            correct: 0,
            logit_error: 0.,
        })
    }
    /// Analytic full-class CE residuals are a read-only diagnostic in BOTH modes.
    /// This is not a forward-only training update. Native weights already include B.
    pub fn observe(&mut self, records: &[Record<'_>]) -> Result<()> {
        if records.is_empty() {
            return Err("empty diagnostic minibatch".into());
        }
        for record in records {
            let states: Vec<_> = self
                .model
                .atoms
                .iter()
                .map(|a| a.state(record.prefix))
                .collect::<Result<_>>()?;
            let logits = readouts::update::logits(self.model, &states)?;
            let actual = self.model.logits(record.prefix)?;
            for (&a, &b) in logits.iter().zip(&actual) {
                if !a.is_finite() || !b.is_finite() {
                    return Err("nonfinite actual logit".into());
                }
                self.logit_error = self.logit_error.max((a - b).abs());
            }
            if self.logit_error > 1e-12 {
                return Err("complete logits disagree".into());
            }
            self.ce += cross_entropy(&actual, record.target)?;
            let prediction = (1..actual.len())
                .fold(0, |best, c| if actual[c] > actual[best] { c } else { best });
            self.correct += usize::from(prediction == record.target);
            // scalars(Mini) independently validates ordinary CE: two CE evaluations/record.
            let signal = rotary::train::scalars(&actual, record.target, Mode::Mini, 1e-4)?;
            for ((rows, &weight), state) in self
                .coefficient
                .iter_mut()
                .zip(&self.model.weights)
                .zip(&states)
            {
                for (a, &s) in rows.iter_mut().zip(&signal) {
                    for k in 0..2 {
                        a[k] += state[k] * (weight * s);
                    }
                }
            }
            self.records = self.records.checked_add(1).ok_or("record overflow")?;
        }
        Ok(())
    }
    pub fn finish(mut self) -> Result<Analysis> {
        if self.records == 0 {
            return Err("no diagnostic records".into());
        }
        let mut best = self.model.clone();
        let (mut linear, mut support, mut minimum, mut change) = (0., 0., 0., 0.);
        for (j, rows) in self.coefficient.iter_mut().enumerate() {
            for (c, a) in rows.iter_mut().enumerate() {
                for z in a.iter_mut() {
                    *z /= self.records as f64;
                }
                let u = self.model.atoms[j].output[c];
                let v = best_row(*a)?;
                best.atoms[j].output[c] = v;
                linear += row(u, *a).re;
                support += a[0].norm().max(a[1].norm());
                minimum += row(v, *a).re;
                change += row([v[0] - u[0], v[1] - u[1]], *a).re;
            }
        }
        let excess = best.validate()?.max(self.model.validate()?);
        let gap = linear + support;
        if [linear, support, minimum, change, gap, self.ce]
            .iter()
            .any(|x| !x.is_finite())
            || gap < -1e-10
            || (minimum + support).abs() > 1e-10
            || (change + gap).abs() > 1e-10
        {
            return Err("invalid complete floating support diagnostic".into());
        }
        Ok(Analysis {
            coefficient: self.coefficient,
            best,
            summary: Summary {
                records: self.records,
                classes: self.model.objects,
                components: self.model.atoms.len(),
                real_coordinates: 4 * self.model.objects * self.model.atoms.len(),
                mean_ce: self.ce / self.records as f64,
                correct: self.correct,
                linear_cost: linear,
                support_size: support,
                approximate_gap: gap,
                approximate_ce_lower_bound: self.ce / self.records as f64 - gap,
                predicted_change: change,
                support_identity_error: (minimum + support).abs(),
                change_identity_error: (change + gap).abs(),
                maximum_actual_logit_error: self.logit_error,
                maximum_operator_norm_excess: excess,
                ce_evaluations: 2 * self.records,
            },
        })
    }
}
