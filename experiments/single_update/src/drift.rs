//! Exact finite signed-interaction movement: optimizer, rounding, features and bases.
use crate::{
    algebra::linear,
    domain::model::W,
    forcing::{self, Direction, Parts},
    geometry,
    margins::{CLASSES, View},
};
use serde_json::{Value, json};
pub const NAMES: [&str; 10] = [
    "decay",
    "readout",
    "commutator",
    "encoder",
    "minibatch",
    "moment_memory",
    "update_rounding",
    "readout_basis",
    "centered_feature",
    "feature_basis",
];

pub struct Change {
    pub terms: [Vec<f64>; 10],
}
pub fn terms(
    old: &View,
    new: &View,
    d: &Direction,
    p: &Parts,
    q: &[f64],
    label: usize,
    new_e: &[f64],
) -> Change {
    let n = old.interaction.len() / CLASSES;
    let score = |r: &[f64], w: &[f64]| linear(r, n, W, w, CLASSES);
    let mut terms: [Vec<f64>; 10] = std::array::from_fn(|_| Vec::new());
    terms[0] = old.interaction.iter().map(|x| (d.decay - 1.) * x).collect();
    for i in 0..5 {
        terms[i + 1] = score(&old.residual, &geometry::contrasts(&p.terms[i], label, W))
            .iter()
            .map(|x| -d.eta * x)
            .collect();
    }
    terms[6] = score(
        &old.residual,
        &geometry::project(&geometry::contrasts(&d.rounding, label, W), q, W),
    );
    let old_q_new_w = geometry::project(&geometry::contrasts(new_e, label, W), q, W);
    terms[7] = score(
        &old.residual,
        &forcing::subtract(&new.projected_weights, &old_q_new_w),
    );
    let feature_movement = geometry::project(&forcing::subtract(&new.delta, &old.delta), q, W);
    terms[8] = score(&feature_movement, &new.projected_weights);
    let new_delta_old_q = geometry::project(&new.delta, q, W);
    terms[9] = score(
        &forcing::subtract(&new.residual, &new_delta_old_q),
        &new.projected_weights,
    );
    Change { terms }
}

#[derive(Default)]
pub struct Stats {
    n: usize,
    old_abs: f64,
    new_abs: f64,
    old_square: f64,
    new_square: f64,
    signed: [f64; 10],
    absolute: [f64; 10],
    change_abs: f64,
    closure: f64,
    scaled_closure: f64,
}
impl Stats {
    pub fn add(&mut self, old: &View, new: &View, c: &Change) {
        for k in 0..old.interaction.len() {
            self.n += 1;
            let (a, b) = (old.interaction[k], new.interaction[k]);
            self.old_abs += a.abs();
            self.new_abs += b.abs();
            self.old_square += a * a;
            self.new_square += b * b;
            let mut sum = 0.;
            let mut scale = 1. + a.abs() + b.abs();
            for i in 0..10 {
                let v = c.terms[i][k];
                sum += v;
                scale += v.abs();
                self.signed[i] += v;
                self.absolute[i] += v.abs();
            }
            self.change_abs += (b - a).abs();
            self.closure = self.closure.max((b - a - sum).abs());
            self.scaled_closure = self.scaled_closure.max((b - a - sum).abs() / scale);
        }
    }
    pub fn valid(&self) -> bool {
        self.scaled_closure <= 1e-8
    }
    pub fn json(&self) -> Value {
        let n = self.n as f64;
        json!({"pairs":self.n,"old_new_mean_absolute_interaction":[self.old_abs/n,self.new_abs/n],
            "old_new_mean_squared_interaction":[self.old_square/n,self.new_square/n],
            "term_names":NAMES,"mean_signed_terms":self.signed.map(|x|x/n),
            "mean_absolute_terms":self.absolute.map(|x|x/n),"mean_absolute_change":self.change_abs/n,
            "closure_max_error":self.closure,"closure_scaled_error":self.scaled_closure,"passed":self.valid()})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        application::State,
        domain::model::E,
        geometry::{Basis, Group},
        measure::Snapshot,
    };
    #[test]
    fn moving_basis_feature_and_fp32_update_close_together() {
        let state = |shift: f32| State {
            parameters: (0..E).map(|k| (k % 7) as f32 * 0.01 + shift).collect(),
            first: vec![0.003; E],
            second: vec![0.02; E],
        };
        let (before, after) = (state(0.), state(0.001));
        let d = Direction::new(
            &before,
            &after,
            &vec![0.01; E],
            &vec![-0.02; E],
            &vec![0.04; E],
            11,
        );
        let snapshot = |a: f64, b: f64| {
            let mut features = vec![0.; 2 * W];
            features[0] = 1.;
            features[W] = a;
            features[W + 1] = b;
            Snapshot {
                features,
                logits: vec![0.; 2 * 97],
                readout: vec![],
                full: vec![],
                metrics: json!({}),
                rms_error: 0.,
            }
        };
        let (s0, s1) = (snapshot(2., 3.), snapshot(2.5, 2.));
        let mut q0 = vec![0.; W * W];
        for k in 1..W {
            q0[k * W + k] = 1.;
        }
        let mut q1 = q0.clone();
        q1[0] = 0.5;
        q1[1] = -0.5;
        q1[W] = -0.5;
        q1[W + 1] = 0.5;
        let b0 = Basis {
            requested: 1,
            achieved: 1,
            q: q0,
        };
        let b1 = Basis {
            requested: 1,
            achieved: 1,
            q: q1,
        };
        let g = Group {
            key: 0,
            label: 0,
            train: vec![0],
            validation: vec![1],
            center: s0.features[..W].to_vec(),
            bases: vec![],
        };
        let e0: Vec<_> = before.parameters.iter().map(|&v| v as f64).collect();
        let e1: Vec<_> = after.parameters.iter().map(|&v| v as f64).collect();
        let old = View::new(&s0, &e0, &g, &b0, &[1]);
        let new = View::new(&s1, &e1, &g, &b1, &[1]);
        let p = d.project(&b0.q, W);
        let change = terms(&old, &new, &d, &p, &b0.q, 0, &e1);
        let mut stats = Stats::default();
        stats.add(&old, &new, &change);
        assert!(stats.valid());
        assert!(stats.change_abs > 0.);
        assert!(stats.absolute[7] > 0. && stats.absolute[8] > 0. && stats.absolute[9] > 0.);
    }
}
