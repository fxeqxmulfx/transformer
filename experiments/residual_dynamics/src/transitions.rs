//! RD4: finite checkpoint differences, not causal attribution to individual updates.
use crate::{
    diagnostic::{Snapshot, mean_norm},
    geometry::{self, RANKS},
    gradient::energy,
    model::W,
};
use serde_json::{Value, json};

struct Change {
    old: f64,
    frozen_new: f64,
    new: f64,
    old_norm: f64,
    new_norm: f64,
    vector_error: f64,
}
fn change(x0: &[f64], x1: &[f64], q0: &[f64], q1: &[f64], width: usize) -> Change {
    assert_eq!(x0.len(), x1.len());
    let n = (x0.len() / width) as f64;
    let old = geometry::project(x0, q0, width);
    let frozen = geometry::project(x1, q0, width);
    let new = geometry::project(x1, q1, width);
    let dx: Vec<_> = x1.iter().zip(x0).map(|(a, b)| a - b).collect();
    let dq: Vec<_> = q1.iter().zip(q0).map(|(a, b)| a - b).collect();
    let moved = geometry::project(&dx, q0, width);
    let rotated = geometry::project(x1, &dq, width);
    let vector_error = (0..new.len())
        .map(|i| (new[i] - old[i] - moved[i] - rotated[i]).abs())
        .fold(0., f64::max);
    Change {
        old: energy(&old) / n,
        frozen_new: energy(&frozen) / n,
        new: energy(&new) / n,
        old_norm: mean_norm(&old, width),
        new_norm: mean_norm(&new, width),
        vector_error,
    }
}
#[derive(Default)]
struct Accum {
    count: usize,
    state: [f64; 3],
    readout: [f64; 3],
    product: [f64; 2],
    max_vector_error: f64,
}
impl Accum {
    fn add(&mut self, state: &Change, readout: &Change, n: usize) {
        self.count += n;
        for (dest, c) in [(&mut self.state, state), (&mut self.readout, readout)] {
            for (j, v) in [c.old, c.frozen_new, c.new].iter().enumerate() {
                dest[j] += n as f64 * v;
            }
            self.max_vector_error = self.max_vector_error.max(c.vector_error);
        }
        self.product[0] += n as f64 * state.old_norm * readout.old_norm;
        self.product[1] += n as f64 * state.new_norm * readout.new_norm;
    }
    fn json(&self) -> Value {
        let n = self.count as f64;
        let describe = |e: [f64; 3]| {
            json!({"old":e[0]/n,"new_with_old_basis":e[1]/n,"new":e[2]/n,
            "movement":(e[1]-e[0])/n,"basis":(e[2]-e[1])/n,"total":(e[2]-e[0])/n,
            "increased":e[2]>e[0]+1e-8*(n+e[0]),
            "energy_identity_error":((e[2]-e[0])-((e[1]-e[0])+(e[2]-e[1]))).abs()/n})
        };
        json!({"count":self.count,"state":describe(self.state),"readout":describe(self.readout),
            "norm_product":{"old":self.product[0]/n,"new":self.product[1]/n,
                "increased":self.product[1]>self.product[0]+1e-8*(n+self.product[0])},
            "max_vector_identity_error":self.max_vector_error})
    }
}
pub fn compare(old: &Snapshot, new: &Snapshot) -> Value {
    let mut result = Vec::new();
    for kind in 0..2 {
        assert_eq!(old.groups[kind].len(), new.groups[kind].len());
        let mut totals: Vec<[Accum; 2]> = RANKS.iter().map(|_| Default::default()).collect();
        let mut groups = Vec::new();
        for (g0, g1) in old.groups[kind].iter().zip(&new.groups[kind]) {
            assert_eq!(
                (g0.key, g0.label, &g0.train, &g0.validation),
                (g1.key, g1.label, &g1.train, &g1.validation)
            );
            let w0 = geometry::contrasts(&old.embedding, g0.label, W);
            let w1 = geometry::contrasts(&new.embedding, g1.label, W);
            let mut methods = Vec::new();
            for (r, (b0, b1)) in g0.bases.iter().zip(&g1.bases).enumerate() {
                let weights = change(&w0, &w1, &b0.q, &b1.q, W);
                let mut parts = Vec::new();
                for (s, indices) in [&g0.train, &g0.validation].iter().enumerate() {
                    let x0 = geometry::displacement(&old.states, indices, &g0.center);
                    let x1 = geometry::displacement(&new.states, indices, &g1.center);
                    let features = change(&x0, &x1, &b0.q, &b1.q, W);
                    let mut local = Accum::default();
                    local.add(&features, &weights, indices.len());
                    totals[r][s].add(&features, &weights, indices.len());
                    parts.push(local.json());
                }
                methods.push(json!({"rank":RANKS[r],"train":parts[0],"validation":parts[1]}));
            }
            groups.push(json!({"key":g0.key,"label":g0.label,"methods":methods}));
        }
        let aggregate: Vec<_> = RANKS
            .iter()
            .enumerate()
            .map(|(r, rank)| {
                json!({"rank":rank,
            "train":totals[r][0].json(),"validation":totals[r][1].json()})
            })
            .collect();
        result.push(json!({"grouping":if kind==0 {"sum"}else{"orbit"},"aggregate":aggregate,"groups":groups}));
    }
    json!({"training_ce_old":old.training_ce,"training_ce_new":new.training_ce,
        "training_ce_decreased":new.training_ce<old.training_ce,"groups":result})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn moving_basis_separates_motion_from_rotation_and_retains_growth() {
        let c = change(
            &[1., 2.],
            &[3., 4.],
            &[0., 0., 0., 1.],
            &[1., 0., 0., 0.],
            2,
        );
        assert_eq!((c.old, c.frozen_new, c.new), (4., 16., 9.));
        assert_eq!((c.frozen_new - c.old, c.new - c.frozen_new), (12., -7.));
        assert!(c.vector_error < 1e-15);
    }
}
