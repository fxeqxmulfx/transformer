//! AI4: signed interactions and exact finite changes, including the mixed term.
use crate::{
    algebra::linear,
    diagnostic::Snapshot,
    geometry::{self, RANKS},
    gradient::energy,
    model::W,
};
use serde_json::{Value, json};

#[derive(Default)]
pub struct Stats {
    n: usize,
    signed: f64,
    absolute: f64,
    square: f64,
    product: f64,
    defect: f64,
    cauchy_excess: f64,
}
impl Stats {
    pub fn add(&mut self, u: &[f64], r: &[f64], w: &[f64], width: usize) {
        let nc = u.len() / width;
        let nq = r.len() / width;
        let j = linear(r, nq, width, u, nc);
        let direct = linear(r, nq, width, w, nc);
        let un: Vec<_> = u.chunks_exact(width).map(|x| energy(x).sqrt()).collect();
        for (q, x) in r.chunks_exact(width).enumerate() {
            let rn = energy(x).sqrt();
            for (c, unorm) in un.iter().enumerate() {
                let at = q * nc + c;
                self.n += 1;
                self.signed += j[at];
                self.absolute += j[at].abs();
                self.square += j[at] * j[at];
                self.product += unorm * rn;
                self.defect = self.defect.max((direct[at] - j[at]).abs());
                self.cauchy_excess = self.cauchy_excess.max(j[at].abs() - unorm * rn);
            }
        }
    }
    pub fn json(&self) -> Value {
        let n = self.n as f64;
        json!({"pairs":self.n,"mean_signed":self.signed/n,"mean_absolute":self.absolute/n,
            "mean_square":self.square/n,"mean_norm_product":self.product/n,
            "absolute_to_norm_product":if self.product>0. {Some(self.absolute/self.product)}else{None},
            "max_projection_defect":self.defect,"max_cauchy_excess":self.cauchy_excess})
    }
}

#[derive(Default)]
struct Drift {
    n: usize,
    old_abs: f64,
    new_abs: f64,
    old_square: f64,
    new_square: f64,
    signed: [f64; 3],
    absolute: [f64; 3],
    absolute_change: f64,
    bound: f64,
    max_error: f64,
    max_bound_excess: f64,
}
impl Drift {
    fn add(&mut self, u0: &[f64], r0: &[f64], u1: &[f64], r1: &[f64], width: usize) {
        let (nc, nq) = (u0.len() / width, r0.len() / width);
        let du: Vec<_> = u1.iter().zip(u0).map(|(a, b)| a - b).collect();
        let dr: Vec<_> = r1.iter().zip(r0).map(|(a, b)| a - b).collect();
        let old = linear(r0, nq, width, u0, nc);
        let new = linear(r1, nq, width, u1, nc);
        let terms = [
            linear(r0, nq, width, &du, nc),
            linear(&dr, nq, width, u0, nc),
            linear(&dr, nq, width, &du, nc),
        ];
        let un: Vec<_> = u0.chunks_exact(width).map(|x| energy(x).sqrt()).collect();
        let dun: Vec<_> = du.chunks_exact(width).map(|x| energy(x).sqrt()).collect();
        for q in 0..nq {
            let rn = energy(&r0[q * width..(q + 1) * width]).sqrt();
            let drn = energy(&dr[q * width..(q + 1) * width]).sqrt();
            for c in 0..nc {
                let at = q * nc + c;
                let change = new[at] - old[at];
                let sum = terms.iter().map(|t| t[at]).sum::<f64>();
                let bound = dun[c] * rn + un[c] * drn + dun[c] * drn;
                self.n += 1;
                self.old_abs += old[at].abs();
                self.new_abs += new[at].abs();
                self.old_square += old[at].powi(2);
                self.new_square += new[at].powi(2);
                self.absolute_change += change.abs();
                self.bound += bound;
                self.max_error = self.max_error.max((change - sum).abs());
                self.max_bound_excess = self.max_bound_excess.max(change.abs() - bound);
                for (j, t) in terms.iter().enumerate() {
                    self.signed[j] += t[at];
                    self.absolute[j] += t[at].abs();
                }
            }
        }
    }
    fn json(&self) -> Value {
        let n = self.n as f64;
        json!({"pairs":self.n,"old_mean_absolute":self.old_abs/n,"new_mean_absolute":self.new_abs/n,
            "old_mean_square":self.old_square/n,"new_mean_square":self.new_square/n,
            "absolute_increased":self.new_abs>self.old_abs+1e-8*(n+self.old_abs),
            "square_increased":self.new_square>self.old_square+1e-8*(n+self.old_square),
            "signed_readout_change":self.signed[0]/n,"signed_state_change":self.signed[1]/n,
            "signed_mixed_change":self.signed[2]/n,
            "absolute_readout_change":self.absolute[0]/n,"absolute_state_change":self.absolute[1]/n,
            "absolute_mixed_change":self.absolute[2]/n,"mean_absolute_total_change":self.absolute_change/n,
            "mean_change_bound":self.bound/n,"max_identity_error":self.max_error,
            "max_change_bound_excess":self.max_bound_excess})
    }
}
pub fn compare(old: &Snapshot, new: &Snapshot) -> Value {
    let mut result = Vec::new();
    for kind in 0..2 {
        assert_eq!(old.groups[kind].len(), new.groups[kind].len());
        let mut totals: Vec<[Drift; 2]> = RANKS.iter().map(|_| Default::default()).collect();
        let mut records = Vec::new();
        for (g0, g1) in old.groups[kind].iter().zip(&new.groups[kind]) {
            assert_eq!(
                (g0.key, g0.label, &g0.train, &g0.validation),
                (g1.key, g1.label, &g1.train, &g1.validation)
            );
            let w0 = geometry::contrasts(&old.embedding, g0.label, W);
            let w1 = geometry::contrasts(&new.embedding, g1.label, W);
            let mut methods = Vec::new();
            for (rank, (b0, b1)) in g0.bases.iter().zip(&g1.bases).enumerate() {
                let u0 = geometry::project(&w0, &b0.q, W);
                let u1 = geometry::project(&w1, &b1.q, W);
                let mut parts = Vec::new();
                for (s, indices) in [&g0.train, &g0.validation].iter().enumerate() {
                    let r0 = geometry::project(
                        &geometry::displacement(&old.states, indices, &g0.center),
                        &b0.q,
                        W,
                    );
                    let r1 = geometry::project(
                        &geometry::displacement(&new.states, indices, &g1.center),
                        &b1.q,
                        W,
                    );
                    let mut local = Drift::default();
                    local.add(&u0, &r0, &u1, &r1, W);
                    totals[rank][s].add(&u0, &r0, &u1, &r1, W);
                    parts.push(local.json());
                }
                methods.push(json!({"rank":RANKS[rank],"train":parts[0],"validation":parts[1]}));
            }
            records.push(json!({"key":g0.key,"label":g0.label,"methods":methods}));
        }
        let aggregate: Vec<_> = RANKS
            .iter()
            .enumerate()
            .map(|(r, rank)| {
                json!({"rank":rank,
            "train":totals[r][0].json(),"validation":totals[r][1].json()})
            })
            .collect();
        result.push(json!({"grouping":if kind==0 {"sum"}else{"orbit"},"aggregate":aggregate,"groups":records}));
    }
    json!({"training_ce_old":old.training_ce,"training_ce_new":new.training_ce,
        "training_ce_decreased":new.training_ce<old.training_ce,"groups":result})
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_mixed_change_preserves_cancellation_when_norm_product_grows() {
        let (u, r, un, rn) = ([1., 0.], [1., 0.], [0., 2.], [3., 0.]);
        let mut d = Drift::default();
        d.add(&u, &r, &un, &rn, 2);
        assert_eq!(d.signed, [-1., 2., -2.]);
        assert_eq!((d.old_abs, d.new_abs), (1., 0.));
        assert_eq!(d.max_error, 0.);
        assert_eq!(energy(&un) * energy(&rn), 36.);
        let mut stats = Stats::default();
        stats.add(&un, &rn, &un, 2);
        assert_eq!(stats.json()["mean_norm_product"], 6.);
        assert_eq!(stats.json()["mean_absolute"], 0.);
    }
}
