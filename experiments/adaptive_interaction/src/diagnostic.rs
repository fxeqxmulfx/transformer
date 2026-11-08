//! AI1-AI4: saved adaptive forcing and signed residual interactions.
use crate::{
    Saved,
    data::Split,
    geometry::{self, FeatureMoment, Group, RANKS},
    gradient::{self, energy},
    interaction::Stats,
    model::{E, Forward, P, W},
    moments,
};
use serde_json::{Value, json};

pub struct Snapshot {
    pub states: Vec<f64>,
    pub embedding: Vec<f64>,
    pub groups: [Vec<Group>; 2],
    pub training_ce: f64,
}
fn target(q: usize) -> usize {
    (q / P + q % P).pow(3) % P
}
fn source_metrics(out: &Forward, indices: &[usize]) -> Value {
    let (mut loss, mut correct) = (0., 0);
    for &q in indices {
        let row = &out.logits[q * P..(q + 1) * P];
        let best = (1..P).fold(0, |b, c| if row[c] > row[b] { c } else { b });
        correct += usize::from(best == target(q));
        loss += row[best] - row[target(q)]
            + row.iter().map(|v| (v - row[best]).exp()).sum::<f64>().ln();
    }
    json!({"count":indices.len(),"source_correct":correct,"mean_ce":loss/indices.len() as f64})
}
pub fn analyze(out: &Forward, split: &Split, saved: &Saved, step: usize) -> (Snapshot, Value) {
    let states: Vec<_> = out
        .states
        .iter()
        .enumerate()
        .map(|(i, x)| x / out.denominators[i / W])
        .collect();
    let embedding = saved.parameters[..E].to_vec();
    let train_states = geometry::selected(&states, &split.train, W);
    let labels: Vec<_> = split.train.iter().map(|&i| target(i)).collect();
    let (training_ce, g) = gradient::loss_gradient(&train_states, &labels, &embedding, W, P);
    let train = source_metrics(out, &split.train);
    let validation = source_metrics(out, &split.validation);
    assert!((training_ce - train["mean_ce"].as_f64().unwrap()).abs() < 1e-10);
    let training: Vec<_> = split.train.iter().copied().zip(labels).collect();
    let groups = [
        geometry::groups(&states, &training, W, false),
        geometry::groups(&states, &training, W, true),
    ];
    let moment = FeatureMoment::new(&train_states, W);
    let direction = if step > 0 {
        Some(moments::corrected(&saved.first, &saved.second, step))
    } else {
        None
    };
    let mut all = Vec::new();
    for (kind, grouped) in groups.iter().enumerate() {
        let mut totals: Vec<[Stats; 2]> = RANKS.iter().map(|_| Default::default()).collect();
        let mut records = Vec::new();
        for group in grouped {
            let w = geometry::contrasts(&embedding, group.label, W);
            let x = [
                geometry::displacement(&states, &group.train, &group.center),
                geometry::displacement(&states, &group.validation, &group.center),
            ];
            let mut methods = Vec::new();
            for (r, basis) in group.bases.iter().enumerate() {
                let u = geometry::project(&w, &basis.q, W);
                let mut parts = Vec::new();
                for (s, x) in x.iter().enumerate() {
                    let residual = geometry::project(x, &basis.q, W);
                    let mut local = Stats::default();
                    local.add(&u, &residual, &w, W);
                    totals[r][s].add(&u, &residual, &w, W);
                    parts.push(local.json());
                }
                let adaptive = direction.as_ref().map(|d| {
                    moments::audit(
                        d,
                        &g,
                        &basis.q,
                        W,
                        training_ce,
                        moment.residual_energy(&basis.q),
                    )
                });
                methods.push(json!({"rank":basis.requested,"achieved":basis.achieved,
                    "adaptive":adaptive,"train":parts[0],"validation":parts[1]}));
            }
            records.push(json!({"key":group.key,"label":group.label,"training_count":group.train.len(),
                "validation_count":group.validation.len(),"routes_verified":true,"training_labels_consistent":true,
                "methods":methods}));
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
    let report = json!({"train":train,"validation":validation,"groups":all,
        "adaptive_status":if step>0 {"saved_historical_moments"}else{"not_applicable_initial_step"},
        "gradient_scope":"current full-training partial readout derivative; historical moment discrepancy includes other effects",
        "readout_gradient_energy":energy(&g),"training_state_mean_energy":energy(&train_states)/split.train.len() as f64,
        "first_moment_energy":energy(&saved.first),"second_moment_min":saved.second.iter().copied().fold(f64::INFINITY,f64::min),
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
