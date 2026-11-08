//! Training-affine reconstruction with signed residuals and executed-logit allowances.
use crate::{
    algebra::linear,
    domain::model::W,
    geometry::{self, Basis, Group},
    gradient::energy,
    measure::Snapshot,
};
use serde_json::{Value, json};
pub const CLASSES: usize = 96;

pub struct View {
    pub delta: Vec<f64>,
    pub residual: Vec<f64>,
    pub projected_weights: Vec<f64>,
    pub interaction: Vec<f64>,
    reconstructed: Vec<f64>,
    defect: Vec<f64>,
    execution: Vec<f64>,
    actual: Vec<f64>,
    identity_error: f64,
    execution_error: f64,
}
impl View {
    pub fn new(s: &Snapshot, e: &[f64], g: &Group, b: &Basis, indices: &[usize]) -> Self {
        let w = geometry::contrasts(e, g.label, W);
        let x = geometry::selected(&s.features, indices, W);
        let delta = geometry::displacement(&s.features, indices, &g.center);
        let residual = geometry::project(&delta, &b.q, W);
        let projected_weights = geometry::project(&w, &b.q, W);
        let interaction = linear(&residual, indices.len(), W, &projected_weights, CLASSES);
        let direct_residual = linear(&residual, indices.len(), W, &w, CLASSES);
        let represented: Vec<_> = x.iter().zip(&residual).map(|(x, r)| x - r).collect();
        let reconstructed = linear(&represented, indices.len(), W, &w, CLASSES);
        let scores = linear(&x, indices.len(), W, &w, CLASSES);
        let actual: Vec<_> = indices
            .iter()
            .flat_map(|&q| {
                (0..97)
                    .filter(move |&c| c != g.label)
                    .map(move |c| s.logits[q * 97 + c] - s.logits[q * 97 + g.label])
            })
            .collect();
        let defect: Vec<_> = direct_residual
            .iter()
            .zip(&interaction)
            .map(|(r, j)| r - j)
            .collect();
        let execution: Vec<_> = actual.iter().zip(&scores).map(|(a, s)| a - s).collect();
        let mut identity_error: f64 = 0.;
        let mut execution_error: f64 = 0.;
        for k in 0..scores.len() {
            identity_error = identity_error.max(
                (reconstructed[k] + interaction[k] + defect[k] - scores[k]).abs()
                    / (1. + scores[k].abs() + reconstructed[k].abs() + direct_residual[k].abs()),
            );
            execution_error =
                execution_error.max(execution[k].abs() / (1. + actual[k].abs() + scores[k].abs()));
        }
        Self {
            delta,
            residual,
            projected_weights,
            interaction,
            reconstructed,
            defect,
            execution,
            actual,
            identity_error,
            execution_error,
        }
    }
}

#[derive(Default)]
pub struct Stats {
    n: usize,
    margins: [f64; 2],
    reconstructed: [f64; 2],
    budget: [f64; 2],
    positive_margins: [usize; 2],
    positive_ratios: [f64; 2],
    certified: [usize; 2],
    correct: [usize; 2],
    gain: usize,
    loss: usize,
    improved: usize,
    regressed: usize,
    residual_grows: usize,
    margin_improves_residual_grows: usize,
    invalid_certificates: usize,
    identity_error: f64,
    execution_error: f64,
    max_execution: f64,
    max_defect: f64,
}
impl Stats {
    pub fn add(&mut self, old: &View, new: &View) {
        assert_eq!(old.actual.len(), new.actual.len());
        for (j, v) in [old, new].iter().enumerate() {
            self.identity_error = self.identity_error.max(v.identity_error);
            self.execution_error = self.execution_error.max(v.execution_error);
            self.max_execution = self
                .max_execution
                .max(v.execution.iter().map(|x| x.abs()).fold(0., f64::max));
            self.max_defect = self
                .max_defect
                .max(v.defect.iter().map(|x| x.abs()).fold(0., f64::max));
            for row in 0..v.actual.len() / CLASSES {
                let at = row * CLASSES;
                let m = -v.actual[at..at + CLASSES]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let reconstructed = -v.reconstructed[at..at + CLASSES]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let budget = (at..at + CLASSES)
                    .map(|k| v.interaction[k].abs() + v.defect[k].abs() + v.execution[k].abs())
                    .fold(0., f64::max);
                let certified = (at..at + CLASSES).all(|k| {
                    v.reconstructed[k]
                        + v.interaction[k].abs()
                        + v.defect[k].abs()
                        + v.execution[k].abs()
                        < 0.
                });
                self.margins[j] += m;
                self.reconstructed[j] += reconstructed;
                self.budget[j] += budget;
                self.certified[j] += usize::from(certified);
                self.correct[j] += usize::from(m > 0.);
                self.invalid_certificates += usize::from(certified && m <= 0.);
                if reconstructed > 0. {
                    self.positive_margins[j] += 1;
                    self.positive_ratios[j] += budget / reconstructed;
                }
            }
        }
        for row in 0..old.actual.len() / CLASSES {
            self.n += 1;
            let at = row * CLASSES;
            let margin = |v: &View| {
                -v.actual[at..at + CLASSES]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            let budget = |v: &View| {
                (at..at + CLASSES)
                    .map(|k| v.interaction[k].abs() + v.defect[k].abs() + v.execution[k].abs())
                    .fold(0., f64::max)
            };
            let (m0, m1) = (margin(old), margin(new));
            self.gain += usize::from(m0 <= 0. && m1 > 0.);
            self.loss += usize::from(m0 > 0. && m1 <= 0.);
            self.improved += usize::from(m1 > m0);
            self.regressed += usize::from(m1 < m0);
            self.residual_grows += usize::from(budget(new) > budget(old));
            self.margin_improves_residual_grows +=
                usize::from(m1 > m0 && budget(new) > budget(old));
        }
    }
    pub fn valid(&self) -> bool {
        self.identity_error <= 1e-8
            && self.execution_error <= 5e-5
            && self.invalid_certificates == 0
    }
    pub fn json(&self) -> Value {
        let n = self.n as f64;
        let ratios = [0, 1].map(|j| {
            if self.positive_margins[j] > 0 {
                Some(self.positive_ratios[j] / self.positive_margins[j] as f64)
            } else {
                None
            }
        });
        json!({"queries":self.n,"old_new_mean_actual_margin":self.margins.map(|x|x/n),
            "old_new_mean_reconstructed_margin":self.reconstructed.map(|x|x/n),
            "old_new_mean_max_residual_budget":self.budget.map(|x|x/n),
            "old_new_positive_reconstructed_margins":self.positive_margins,
            "old_new_mean_ratio_on_positive_margins":ratios,
            "old_new_certified":self.certified,"old_new_strict_correct":self.correct,
            "correct_gained":self.gain,"correct_lost":self.loss,"margin_improved":self.improved,"margin_regressed":self.regressed,
            "residual_budget_grew":self.residual_grows,"margin_improved_while_residual_grew":self.margin_improves_residual_grows,
            "invalid_certificates":self.invalid_certificates,"identity_scaled_error":self.identity_error,
            "execution_scaled_error":self.execution_error,"max_execution_allowance":self.max_execution,
            "max_projection_defect":self.max_defect,"passed":self.valid()})
    }
}

pub fn residual_energy(v: &View) -> f64 {
    energy(&v.projected_weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_residual_can_grow_while_margin_improves() {
        let make = |s: f64, j: f64| View {
            delta: vec![],
            residual: vec![],
            projected_weights: vec![],
            interaction: vec![j; CLASSES],
            reconstructed: vec![s; CLASSES],
            defect: vec![0.; CLASSES],
            execution: vec![0.; CLASSES],
            actual: vec![s + j; CLASSES],
            identity_error: 0.,
            execution_error: 0.,
        };
        let mut stats = Stats::default();
        stats.add(&make(-2., 0.5), &make(-4., 1.));
        assert!(stats.valid());
        assert_eq!(stats.margin_improves_residual_grows, 1);
        assert_eq!(stats.certified, [1, 1]);
        stats.add(&make(1., -2.), &make(2., -4.));
        assert_eq!(stats.positive_margins, [1, 1]);
        assert_eq!(stats.correct, [2, 2]);
        assert_eq!(stats.certified, [1, 1]);
    }
}
