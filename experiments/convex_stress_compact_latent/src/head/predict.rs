//! Interval separation or exact dyadic products of the actual rounded head.
//! Ported from reduced_latent at 6797e8ab, preserving every logical product.
//! CM72 (Lean 1279ab89) shares each pass's equal ordered columns and expands
//! every interval/secondary value before the unchanged certificate readout.
use super::{interval::Interval, *};
pub(super) fn argmax(z: &[f64]) -> usize {
    let mut best = 0;
    for c in 1..z.len() {
        if z[c] > z[best] {
            best = c;
        }
    }
    best
}
impl Head {
    pub fn predict(
        &self,
        states: &[usize],
        budget: &mut PredictionBudget,
        work: &mut Work,
    ) -> Result<Prediction> {
        let columns = self.query(states)?;
        let logits = self.logits(states, work)?;
        work.charge(100 * columns as u128 + 16 * self.shape.classes as u128)?;
        let bounds = self.intervals(states);
        let second = self.reverse_products(states);
        self.certify(states, budget, work, logits, bounds, second)
    }
    pub(super) fn intervals(&self, states: &[usize]) -> Vec<Interval> {
        let mut bounds = vec![Interval::point(0.); self.shape.classes];
        for (p, &s) in states.iter().enumerate() {
            let b = &self.blocks[(p * self.shape.states + s) / 4];
            for f in 0..4 {
                let (re, im) = source(s, s / 4 * 4 + f);
                for (c, value) in self.computation_columns(b) {
                    bounds[c] = bounds[c]
                        .add(Interval::point(value[f]).scale(re))
                        .add(Interval::point(value[4 + f]).scale(-im));
                }
            }
        }
        if let Some(sharing) = &self.sharing {
            sharing.expand(&mut bounds);
        }
        bounds
    }
    pub(super) fn reverse_products(&self, states: &[usize]) -> Vec<f64> {
        let mut second = vec![0.; self.shape.classes];
        for (p, &s) in states.iter().enumerate().rev() {
            let b = &self.blocks[(p * self.shape.states + s) / 4];
            for f in (0..4).rev() {
                let (re, im) = reference_source(s, s / 4 * 4 + f);
                for (c, value) in self.computation_columns(b) {
                    second[c] += (re as f64 * value[f] - im as f64 * value[4 + f]) * 0.5;
                }
            }
        }
        if let Some(sharing) = &self.sharing {
            sharing.expand(&mut second);
        }
        second
    }
    fn certify(
        &self,
        states: &[usize],
        budget: &mut PredictionBudget,
        work: &mut Work,
        logits: Vec<f64>,
        bounds: Vec<Interval>,
        second: Vec<f64>,
    ) -> Result<Prediction> {
        let residual = logits
            .iter()
            .zip(&second)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        let best = argmax(&logits);
        let separated = bounds.iter().all(|b| b.finite())
            && (0..bounds.len()).all(|c| c == best || bounds[best].lo > bounds[c].hi);
        if separated && best == argmax(&second) && residual <= 1e-7 {
            Ok(Prediction {
                logits,
                winner: best,
                maximizers: vec![best],
                certificate: Certificate::OutwardIntervals,
                maximum_independent_residual: residual,
                maximum_enclosure_width: bounds
                    .iter()
                    .map(|b| (b.hi - b.lo).next_up())
                    .fold(0., f64::max),
                target_realizability_certified: false,
            })
        } else {
            if budget.exact_queries >= budget.maximum_exact_queries {
                return Err("exact query budget exhausted".into());
            }
            budget.exact_queries += 1;
            exact::predict(self, states, work)
        }
    }
}
