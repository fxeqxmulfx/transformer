//! Exact per-minibatch source AdamW accounting in fixed class projections.
use crate::{
    application::State,
    classes::{Partition, dot, energy},
    domain::{Result, model::E},
    inputs::require,
};
use serde_json::{Value, json};
pub const NAMES: [&str; 6] = [
    "decay",
    "isotropic_readout",
    "adaptive_readout_correction",
    "tied_encoder",
    "moment_memory",
    "rounding",
];
pub struct Summary {
    pub start: [f64; 3],
    pub end: [f64; 3],
    pub radial: [[f64; 6]; 3],
    pub quadratic: [f64; 3],
    pub updates: usize,
    pub increases: [usize; 3],
    pub damping_failures: usize,
    pub max_closure: f64,
    pub max_parameter_closure: f64,
    pub max_damping_excess: f64,
}

impl Summary {
    pub fn new(start: [f64; 3]) -> Self {
        Self {
            start,
            end: start,
            radial: [[0.; 6]; 3],
            quadratic: [0.; 3],
            updates: 0,
            increases: [0; 3],
            damping_failures: 0,
            max_closure: 0.,
            max_parameter_closure: 0.,
            max_damping_excess: 0.,
        }
    }
    pub fn json(&self) -> Value {
        json!({"updates":self.updates,"start":self.start,"end":self.end,"radial_terms":NAMES,"cumulative_radial":self.radial,"cumulative_quadratic":self.quadratic,
        "energy_increase_updates":self.increases,"damping_failures":self.damping_failures,"max_closure_scaled":self.max_closure,"max_parameter_closure":self.max_parameter_closure,"max_damping_excess":self.max_damping_excess})
    }
    pub fn record(
        &mut self,
        part: &Partition,
        old: &State,
        new: &State,
        readout: &[f64],
        full: &[f64],
        step: usize,
    ) -> Result<Value> {
        let old_e: Vec<_> = old.parameters[..E].iter().map(|&x| x as f64).collect();
        let new_e: Vec<_> = new.parameters[..E].iter().map(|&x| x as f64).collect();
        let delta: Vec<_> = new_e.iter().zip(&old_e).map(|(n, o)| n - o).collect();
        let b1 = 1. - 0.9f64.powf(step as f64);
        let b2 = (1. - 0.98f64.powf(step as f64)).sqrt();
        let d: Vec<_> = new.second[..E]
            .iter()
            .map(|&v| 1. / ((v as f64).sqrt() / b2 + 1e-8))
            .collect();
        let alpha = d.iter().copied().fold(f64::INFINITY, f64::min);
        let eta = 0.001 * (step as f64 / 10.).min(1.);
        let mut terms = vec![vec![0.; E]; 6];
        for k in 0..E {
            terms[0][k] = -eta * old_e[k];
            terms[1][k] = -eta * alpha * readout[k];
            terms[2][k] = -eta * (d[k] - alpha) * readout[k];
            terms[3][k] = -eta * d[k] * (full[k] - readout[k]);
            terms[4][k] = -eta * d[k] * (new.first[k] as f64 / b1 - full[k]);
            terms[5][k] = delta[k] - terms[..5].iter().map(|t| t[k]).sum::<f64>();
        }
        let parameter_error = (0..E)
            .map(|k| (delta[k] - terms.iter().map(|t| t[k]).sum::<f64>()).abs())
            .fold(0., f64::max);
        let rounding = terms[5].iter().map(|x| x.abs()).fold(0., f64::max);
        let arithmetic_scale = 1.
            + old_e.iter().map(|x| x.abs()).fold(0., f64::max)
            + (0..E)
                .map(|k| (eta * d[k] * new.first[k] as f64 / b1).abs())
                .fold(0., f64::max);
        let rounding_allowance = 8. * f32::EPSILON as f64 * arithmetic_scale;
        require(
            rounding <= rounding_allowance,
            "optimizer reconstruction exceeds fixed rounding allowance",
        )?;
        self.max_parameter_closure = self.max_parameter_closure.max(parameter_error);
        require(parameter_error < 1e-10, "source component sum failed")?;
        let oldp = part.project(&old_e);
        let newp = part.project(&new_e);
        let dp = part.project(&delta);
        let projected: Vec<_> = terms.iter().map(|v| part.project(v)).collect();
        let gp = part.project(readout);
        let damping = dot(&oldp[1], &gp[1]);
        let tolerance = 1e-8 * (1. + (energy(&oldp[1]) * energy(&gp[1])).sqrt());
        self.damping_failures += usize::from(damping < -tolerance);
        self.max_damping_excess = self.max_damping_excess.max((-damping - tolerance).max(0.));
        let before = oldp.each_ref().map(|v| energy(v));
        let after = newp.each_ref().map(|v| energy(v));
        let radial = std::array::from_fn::<_, 3, _>(|i| {
            std::array::from_fn::<_, 6, _>(|j| 2. * dot(&oldp[i], &projected[j][i]))
        });
        let quadratic = dp.each_ref().map(|v| energy(v));
        for i in 0..3 {
            let change = radial[i].iter().sum::<f64>() + quadratic[i];
            self.max_closure = self
                .max_closure
                .max((after[i] - before[i] - change).abs() / (1. + before[i] + after[i]));
            self.increases[i] +=
                usize::from(after[i] > before[i] + 1e-8 * (1. + after[i].max(before[i])));
            for (sum, term) in self.radial[i].iter_mut().zip(radial[i]) {
                *sum += term;
            }
            self.quadratic[i] += quadratic[i];
            let cumulative = self.radial[i].iter().sum::<f64>() + self.quadratic[i];
            self.max_closure = self.max_closure.max(
                (after[i] - self.start[i] - cumulative).abs()
                    / (1. + after[i] + self.start[i] + cumulative.abs()),
            );
        }
        require(
            self.max_closure < 1e-8,
            "finite/cumulative energy identity failed",
        )?;
        self.end = after;
        self.updates += 1;
        Ok(
            json!({"step":step,"before":before,"after":after,"radial":radial,"quadratic":quadratic,"unused_ce_damping":damping,"damping_tolerance":tolerance,"alpha":alpha,"eta":eta,"parameter_closure":parameter_error,"rounding_max":rounding,"rounding_allowance":rounding_allowance}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mixed_optimizer_components_close_over_every_prefix() {
        let part = Partition::new((0..97).map(|c| c < 33).collect(), 128);
        let mut old = State {
            parameters: (0..E)
                .map(|k| ((k as f64 * 0.37).sin() * 0.1) as f32)
                .collect(),
            first: vec![0.; E],
            second: vec![0.; E],
        };
        let initial: Vec<_> = old.parameters.iter().map(|&v| v as f64).collect();
        let mut summary = Summary::new(part.energies(&initial));
        let features: Vec<_> = (0..256).map(|k| (k as f64 * 0.23).cos()).collect();
        for step in 1..=20 {
            let e: Vec<_> = old.parameters.iter().map(|&v| v as f64).collect();
            let (_, readout) = crate::gradient::loss_gradient(&features, &[0, 1], &e, 128, 97);
            let full: Vec<_> = readout
                .iter()
                .enumerate()
                .map(|(k, g)| g + 0.01 * (k as f64 * 0.17).sin())
                .collect();
            let first: Vec<_> = (0..E)
                .map(|k| (0.9 * old.first[k] as f64 + 0.1 * full[k]) as f32)
                .collect();
            let second: Vec<_> = (0..E)
                .map(|k| (0.98 * old.second[k] as f64 + 0.02 * full[k] * full[k]) as f32)
                .collect();
            let eta = 0.001 * (step as f64 / 10.).min(1.);
            let parameters = (0..E)
                .map(|k| {
                    ((1. - eta) * e[k]
                        - eta * first[k] as f64
                            / (1. - 0.9f64.powi(step))
                            / ((second[k] as f64 / (1. - 0.98f64.powi(step))).sqrt() + 1e-8))
                        as f32
                })
                .collect();
            let new = State {
                parameters,
                first,
                second,
            };
            summary
                .record(&part, &old, &new, &readout, &full, step as usize)
                .unwrap();
            old = new;
        }
        assert_eq!(summary.updates, 20);
        assert_eq!(summary.damping_failures, 0);
        assert!(summary.quadratic.iter().sum::<f64>() > 0.001);
        assert!(summary.radial[0][3].abs() > 1e-4);
        assert!(summary.max_closure < 1e-12);
    }
}
