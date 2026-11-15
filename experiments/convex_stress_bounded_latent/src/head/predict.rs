//! Separated interval winners or independent exact dyadic products from the
//! complete saved head. A close unequal score is never declared a tie.
use super::{interval::Interval, *};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Certificate {
    OutwardIntervals,
    ExactDyadics,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub logits: Vec<f64>,
    pub winner: usize,
    pub maximizers: Vec<usize>,
    pub certificate: Certificate,
    pub maximum_independent_residual: f64,
    pub maximum_enclosure_width: f64,
    pub target_realizability_certified: bool,
}
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
    pub(super) fn exact(&mut self) -> Result<()> {
        if self.exact_queries >= self.maximum_exact_queries {
            return Err("exact ambiguity query budget exhausted".into());
        }
        self.exact_queries += 1;
        Ok(())
    }
}
pub fn argmax(z: &[f64]) -> usize {
    let mut best = 0;
    for c in 1..z.len() {
        if z[c] > z[best] {
            best = c;
        }
    }
    best
}
impl Head {
    /// Validate a deserialized tensor once before calling this repeatedly.
    pub fn predict(
        &self,
        states: &[usize],
        budget: &mut PredictionBudget,
        work: &mut Work,
    ) -> Result<Prediction> {
        self.query(states)?;
        let logits = self.logits(states, work)?;
        work.charge(100 * self.shape.programs as u128 * self.shape.classes as u128)?;
        let mut bounds = vec![Interval::point(0.); self.shape.classes];
        for (p, &s) in states.iter().enumerate() {
            for f in s / 4 * 4..s / 4 * 4 + 4 {
                let (re, im) = source(s, f);
                for (c, b) in bounds.iter_mut().enumerate() {
                    let i = self.at(p, f, c);
                    *b = b
                        .add(Interval::point(self.real[i]).scale(re))
                        .add(Interval::point(self.imaginary[i]).scale(-im));
                }
            }
        }
        let mut second = vec![0.; self.shape.classes];
        for (c, z) in second.iter_mut().enumerate() {
            for (p, &s) in states.iter().enumerate().rev() {
                for f in (s / 4 * 4..s / 4 * 4 + 4).rev() {
                    let (re, im) = reference_source(s, f);
                    let i = self.at(p, f, c);
                    *z += (re as f64 * self.real[i] - im as f64 * self.imaginary[i]) * 0.5;
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
            let maximum_enclosure_width = bounds
                .iter()
                .map(|b| (b.hi - b.lo).next_up())
                .fold(0., f64::max);
            Ok(Prediction {
                logits,
                winner: best,
                maximizers: vec![best],
                certificate: Certificate::OutwardIntervals,
                maximum_independent_residual: residual,
                maximum_enclosure_width,
                target_realizability_certified: false,
            })
        } else {
            budget.exact()?;
            exact::predict(self, states, work)
        }
    }
}
