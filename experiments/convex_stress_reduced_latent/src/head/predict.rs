//! Interval separation or exact dyadic products of the actual rounded head.
//! Adapted from bounded_latent at a3959a46; only exact zero columns are skipped.
use super::{interval::Interval, *};
pub use dense::head::{Certificate, Prediction};
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictionBudget {
    pub exact_queries: usize,
    pub maximum_exact_queries: usize,
}
impl PredictionBudget {
    pub fn new(maximum: usize) -> Result<Self> {
        if maximum > 65536 {
            return Err("exact query admission exceeds 65536".into());
        }
        Ok(Self {
            exact_queries: 0,
            maximum_exact_queries: maximum,
        })
    }
    fn exact(&mut self) -> Result<()> {
        if self.exact_queries >= self.maximum_exact_queries {
            return Err("exact query budget exhausted".into());
        }
        self.exact_queries += 1;
        Ok(())
    }
}
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
        let mut bounds = vec![Interval::point(0.); self.shape.classes];
        let mut second = vec![0.; self.shape.classes];
        for (p, &s) in states.iter().enumerate() {
            let b = &self.blocks[(p * self.shape.states + s) / 4];
            for f in 0..4 {
                let (re, im) = source(s, s / 4 * 4 + f);
                for (i, &c) in b.classes.iter().enumerate() {
                    bounds[c] = bounds[c]
                        .add(Interval::point(b.real[i][f]).scale(re))
                        .add(Interval::point(b.imaginary[i][f]).scale(-im));
                }
            }
        }
        for (p, &s) in states.iter().enumerate().rev() {
            let b = &self.blocks[(p * self.shape.states + s) / 4];
            for f in (0..4).rev() {
                let (re, im) = reference_source(s, s / 4 * 4 + f);
                for (i, &c) in b.classes.iter().enumerate() {
                    second[c] += (re as f64 * b.real[i][f] - im as f64 * b.imaginary[i][f]) * 0.5;
                }
            }
        }
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
            budget.exact()?;
            exact::predict(self, states, work)
        }
    }
}
