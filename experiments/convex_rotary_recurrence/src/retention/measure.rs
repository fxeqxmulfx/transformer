//! Complete state statistics and exact empirical collisions, without feature pruning.
use super::trace;
use crate::{
    Result,
    atom::{State, row},
    model::Model,
};
use bilinear::{model::cross_entropy, train::Record};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub(super) fn run(
    model: &Model,
    records: &[Record<'_>],
    mut guard: impl FnMut() -> Result<()>,
) -> Result<Value> {
    if records.is_empty() {
        return Err("empty state inspection".into());
    }
    model.validate()?;
    let mut features = vec![vec![]; records.len()];
    let mut logits = vec![vec![0.; model.objects]; records.len()];
    let mut weighted_norm = vec![0.; records.len()];
    let mut components = vec![];
    let mut real_error = 0f64;
    let mut constant_components = 0;
    for (index, (atom, &weight)) in model.atoms.iter().zip(&model.weights).enumerate() {
        guard()?;
        let mut positions: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        let mut lengths: BTreeMap<usize, Vec<State>> = BTreeMap::new();
        for (i, record) in records.iter().enumerate() {
            let states = trace::states(atom, record.prefix)?;
            for (position, &state) in states.iter().enumerate() {
                positions
                    .entry(position)
                    .or_default()
                    .push(trace::norm(state));
            }
            let end = *states.last().ok_or("missing end state")?;
            let real = trace::independent_real(atom, record.prefix)?;
            for (a, b) in trace::values(end).iter().zip(real) {
                real_error = real_error.max((a - b).abs());
            }
            lengths.entry(record.prefix.len()).or_default().push(end);
            weighted_norm[i] += weight * trace::norm(end);
            features[i].extend(trace::values(end).map(|x| trace::bits(weight * x)));
            for (value, &out) in logits[i].iter_mut().zip(&atom.output) {
                *value += weight * row(out, end).re;
            }
        }
        let by_length: BTreeMap<_, _> = lengths
            .iter()
            .map(|(&len, xs)| (len, trace::variation(xs)))
            .collect();
        let constant = by_length.values().all(|v| v["exactly_constant"] == true);
        constant_components += usize::from(constant);
        let pooled_variance = lengths
            .iter()
            .map(|(len, xs)| {
                by_length[len]["mean_squared_distance_from_mean"]
                    .as_f64()
                    .unwrap()
                    * xs.len() as f64
            })
            .sum::<f64>()
            / records.len() as f64;
        components.push(json!({"index":index,"weight":weight,"frequency":atom.frequency,
            "initial_max_norm":trace::norm(atom.initial),"within_length_variance":pooled_variance,
            "constant_within_every_length":constant,
            "norm_by_position":positions.iter().map(|(&t,xs)| (t, trace::statistics(xs))).collect::<BTreeMap<_,_>>(),
            "end_by_length":by_length}));
    }
    if real_error > 2e-12 {
        return Err("independent real state check failed".into());
    }
    let mut collision_groups: BTreeMap<Vec<u64>, BTreeMap<usize, usize>> = BTreeMap::new();
    let mut ce = 0.;
    let mut correct = 0;
    let mut logit_error = 0f64;
    for (i, r) in records.iter().enumerate() {
        if i % 64 == 0 {
            guard()?;
        }
        let actual = model.logits(r.prefix)?;
        for (x, y) in logits[i].iter().zip(actual) {
            logit_error = logit_error.max((x - y).abs());
        }
        ce += cross_entropy(&logits[i], r.target)?;
        let mut best = 0;
        for c in 1..model.objects {
            if logits[i][c] > logits[i][best] {
                best = c;
            }
        }
        correct += usize::from(best == r.target);
        *collision_groups
            .entry(std::mem::take(&mut features[i]))
            .or_default()
            .entry(r.target)
            .or_default() += 1;
    }
    if logit_error > 2e-12 {
        return Err("traced complete logits differ from model".into());
    }
    let conflicting: Vec<_> = collision_groups.values().filter(|g| g.len() > 1).collect();
    Ok(json!({"summary":{
        "records":records.len(),"components":model.atoms.len(),"classes":model.objects,
        "state_coordinate_values_compared":records.len()*model.atoms.len()*4,
        "maximum_independent_real_error":real_error,"maximum_complete_logit_error":logit_error,
        "exactly_constant_components_within_every_length":constant_components,
        "complete_feature_groups":collision_groups.len(),"mixed_label_feature_groups":conflicting.len(),
        "records_in_mixed_label_feature_groups":conflicting.iter().map(|g|g.values().sum::<usize>()).sum::<usize>(),
        "mass_weighted_final_max_norm":trace::statistics(&weighted_norm),
        "training_correct":correct,"training_accuracy":correct as f64/records.len() as f64,
        "training_ce":ce/records.len() as f64
    },"components":components,"mixed_label_groups":conflicting}))
}
