//! Checkpoint geometry and RD2 forcing audit. Bases see only observed training labels.
use crate::{
    data::Split,
    geometry::{self, FeatureMoment, Group, RANKS},
    gradient::{self, energy},
    model::{E, Forward, P, W},
};
use serde_json::{Value, json};

pub struct Snapshot {
    pub states: Vec<f64>,
    pub embedding: Vec<f64>,
    pub groups: [Vec<Group>; 2],
    pub training_ce: f64,
}
pub fn target(q: usize) -> usize {
    (q / P + q % P).pow(3) % P
}

fn source_metrics(out: &Forward, indices: &[usize]) -> Value {
    let (mut loss, mut correct) = (0., 0);
    for &q in indices {
        let row = &out.logits[q * P..(q + 1) * P];
        let best = (1..P).fold(0, |b, c| if row[c] > row[b] { c } else { b });
        correct += usize::from(best == target(q));
        let max = row[best];
        loss += max - row[target(q)] + row.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
    }
    json!({"count":indices.len(),"source_correct":correct,"mean_ce":loss/indices.len() as f64})
}
pub fn mean_norm(x: &[f64], width: usize) -> f64 {
    x.chunks_exact(width).map(|r| energy(r).sqrt()).sum::<f64>() / (x.len() / width) as f64
}
#[derive(Default)]
struct Metrics {
    count: usize,
    state: f64,
    state_scale: f64,
    readout: f64,
    readout_scale: f64,
    product: f64,
}
impl Metrics {
    fn add(&mut self, x: &[f64], residual: &[f64], w: &[f64], u: &[f64]) {
        let n = x.len() / W;
        self.count += n;
        self.state += energy(residual);
        self.state_scale += energy(x);
        self.readout += n as f64 * energy(u) / (P - 1) as f64;
        self.readout_scale += n as f64 * energy(w) / (P - 1) as f64;
        self.product += n as f64 * mean_norm(residual, W) * mean_norm(u, W);
    }
    fn json(&self) -> Value {
        let n = self.count as f64;
        assert!(n > 0.);
        json!({"count":self.count,"state_residual_energy":self.state/n,
            "state_displacement_energy":self.state_scale/n,"state_relative_energy":self.state/self.state_scale,
            "readout_residual_energy":self.readout/n,"readout_contrast_energy":self.readout_scale/n,
            "readout_relative_energy":self.readout/self.readout_scale,"mean_norm_product":self.product/n})
    }
}

pub fn analyze(out: &Forward, split: &Split, p: &[f64]) -> (Snapshot, Value) {
    let states: Vec<_> = out
        .states
        .iter()
        .enumerate()
        .map(|(i, x)| x / out.denominators[i / W])
        .collect();
    let embedding = p[..E].to_vec();
    let train_states = geometry::selected(&states, &split.train, W);
    let labels: Vec<_> = split.train.iter().map(|&i| target(i)).collect();
    let (training_ce, gradient) = gradient::loss_gradient(&train_states, &labels, &embedding, W, P);
    let train = source_metrics(out, &split.train);
    let validation = source_metrics(out, &split.validation);
    assert!((training_ce - train["mean_ce"].as_f64().unwrap()).abs() < 1e-10);
    let training: Vec<_> = split.train.iter().copied().zip(labels).collect();
    let groups = [
        geometry::groups(&states, &training, W, false),
        geometry::groups(&states, &training, W, true),
    ];
    let moment = FeatureMoment::new(&train_states, W);
    let sample_indices: Vec<_> = (0..32)
        .map(|j| split.train[j * (split.train.len() - 1) / 31])
        .collect();
    let sample = geometry::selected(&states, &sample_indices, W);
    let sample_moment = FeatureMoment::new(&sample, W);
    let mut all = Vec::new();
    let (mut max_sample_error, mut max_projector_error, mut max_bound_excess) = (0f64, 0f64, 0f64);
    for (kind, grouped) in groups.iter().enumerate() {
        let mut totals: Vec<[Metrics; 2]> = RANKS.iter().map(|_| Default::default()).collect();
        let mut records = Vec::new();
        for group in grouped {
            let w = geometry::contrasts(&embedding, group.label, W);
            let x = [
                geometry::displacement(&states, &group.train, &group.center),
                geometry::displacement(&states, &group.validation, &group.center),
            ];
            let mut methods = Vec::new();
            for (r, basis) in group.bases.iter().enumerate() {
                let projected_gradient = geometry::project(&gradient, &basis.q, W);
                let lhs = energy(&projected_gradient);
                let feature_energy = moment.residual_energy(&basis.q);
                let bound = 2. * training_ce * feature_energy;
                let direct_sample = energy(&geometry::project(&sample, &basis.q, W)) / 32.;
                let sample_error = (direct_sample - sample_moment.residual_energy(&basis.q)).abs();
                let qq = geometry::project(&basis.q, &basis.q, W);
                let projector_error = qq
                    .iter()
                    .zip(&basis.q)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0., f64::max);
                max_sample_error = max_sample_error.max(sample_error);
                max_projector_error = max_projector_error.max(projector_error);
                max_bound_excess = max_bound_excess.max(lhs - bound);
                let u = geometry::project(&w, &basis.q, W);
                let mut parts = Vec::new();
                for (s, displacement) in x.iter().enumerate() {
                    let residual = geometry::project(displacement, &basis.q, W);
                    let mut metrics = Metrics::default();
                    metrics.add(displacement, &residual, &w, &u);
                    totals[r][s].add(displacement, &residual, &w, &u);
                    parts.push(metrics.json());
                }
                methods.push(json!({"rank":basis.requested,"achieved":basis.achieved,
                    "projected_gradient_energy":lhs,"all_training_projected_feature_energy":feature_energy,
                    "loss_forcing_bound":bound,"gradient_to_bound_ratio":if bound>0. {Some(lhs/bound)}else{None},
                    "bound_holds":lhs<=bound+1e-8*(1.+bound),"sample_moment_error":sample_error,
                    "sample_moment_holds":sample_error<=1e-8*(1.+direct_sample),
                    "projector_max_defect":projector_error,"train":parts[0],"validation":parts[1]}));
            }
            records.push(json!({"key":group.key,"label":group.label,"training_count":group.train.len(),
                "validation_count":group.validation.len(),"routes_verified":true,"training_labels_consistent":true,
                "training_center_energy":energy(&group.center),"methods":methods}));
        }
        let aggregate: Vec<_> = RANKS
            .iter()
            .enumerate()
            .map(|(r, rank)| {
                json!({"rank":rank,
            "train":totals[r][0].json(),"validation":totals[r][1].json()})
            })
            .collect();
        all.push(json!({"grouping":if kind==0 {"sum"}else{"orbit"},"aggregate":aggregate,"groups":records}));
    }
    let report = json!({"train":train,"validation":validation,
        "gradient_scope":"partial final-readout CE derivative with normalized states frozen",
        "training_state_mean_energy":energy(&train_states)/split.train.len() as f64,
        "readout_gradient_energy":energy(&gradient),"groups":all,
        "max_sample_moment_error":max_sample_error,"max_projector_defect":max_projector_error,
        "max_gradient_bound_excess":max_bound_excess,
        "source_residual_logit_energy":energy(&out.residual)/(P*P) as f64});
    (
        Snapshot {
            states,
            embedding,
            groups,
            training_ce,
        },
        report,
    )
}
