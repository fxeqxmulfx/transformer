//! No file access: this constructor receives only a saved anchor and known inputs.
use crate::{
    Result, batch, checkpoints,
    classes::Partition,
    domain::data::{Data, Plan},
    evidence::require,
    gpu, guarded, state,
};
use probability_tube::{
    ceiling, contrast_radius,
    encoder::{Shape, norm},
    stability_number,
};
use serde_json::{Value, json};
pub const ENERGY: f64 = 128. * (1. + 5e-6);
pub fn readout(s: &crate::application::State) -> state::State {
    let cv = |x: &[f32]| x[..97 * 128].iter().map(|x| *x as f64).collect();
    state::State {
        p: cv(&s.parameters),
        m: cv(&s.first),
        v: cv(&s.second),
    }
}
pub fn probabilities(logits: &[f64], active: &[bool]) -> Vec<f64> {
    logits
        .as_chunks::<97>()
        .0
        .iter()
        .map(|row| {
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let den = row.iter().map(|x| (x - max).exp()).sum::<f64>();
            row.iter()
                .enumerate()
                .filter(|(c, _)| !active[*c])
                .map(|(_, x)| (x - max).exp() / den)
                .fold(0., f64::max)
        })
        .collect()
}
pub fn execution(p: &[f64], trace: &batch::Trace, actual: &gpu::Snapshot) -> Result<Value> {
    let feature = trace
        .features
        .as_chunks::<128>()
        .0
        .iter()
        .zip(actual.features.as_chunks::<128>().0)
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0., f64::max);
    let dots = crate::algebra::linear(
        &actual.features,
        actual.features.len() / 128,
        128,
        &p[..97 * 128],
        97,
    );
    let readout = dots
        .iter()
        .zip(&actual.logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let energy = actual
        .features
        .as_chunks::<128>()
        .0
        .iter()
        .map(|x| x.iter().map(|v| v * v).sum::<f64>())
        .fold(0., f64::max);
    require(
        feature <= 1e-3 && readout <= 1e-3 && energy <= ENERGY,
        "execution allowance",
    )?;
    let graph_error = trace
        .logits
        .iter()
        .zip(&actual.logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    Ok(json!({"feature_error":feature,"readout_error":readout,
        "graph_logit_error":graph_error,"maximum_feature_energy":energy}))
}
fn contrast_norm(p: &[f64], active: &[bool]) -> f64 {
    (0..97)
        .filter(|u| !active[*u])
        .flat_map(|u| {
            (0..97).filter(move |c| *c != u).map(move |c| {
                (0..128)
                    .map(|k| (p[u * 128 + k] - p[c * 128 + k]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
        })
        .fold(0., f64::max)
}
pub fn make(
    cohort: &str,
    seed: u64,
    anchor: usize,
    saved: &crate::application::State,
    data: &Data,
    plan: &Plan,
) -> Result<Value> {
    let active = crate::construct::support(data);
    let cv = |v: &[f32]| v.iter().map(|x| *x as f64).collect::<Vec<_>>();
    let (p, m, v) = (cv(&saved.parameters), cv(&saved.first), cv(&saved.second));
    let shape = Shape::ORIGINAL;
    let mut tube = guarded::Tube::new(shape, &p, &m, &v, anchor)?;
    let trace = batch::forward(&p, &data.train);
    let e = gpu::engine(saved, data, plan)?;
    let actual = gpu::snapshot(&e, data)?;
    let execution = execution(&p, &trace, &actual)?;
    let old_q = probabilities(&actual.logits, &active);
    let q0 = old_q.iter().copied().fold(0., f64::max);
    let contrast = contrast_norm(&p, &active);
    let old = readout(saved);
    let (alpha, r) = state::metric(&old, &active, anchor);
    let eta = state::rate(cohort, anchor);
    let s = eta * alpha / (1. - 0.9f64.powf((anchor + 1) as f64));
    let part = Partition::new(active.clone(), 128);
    let w = part.project(&old.p)[1].clone();
    let m = part.project(&old.m)[1].clone();
    let mut metrics = vec![];
    for (name, weights) in [("isotropic", vec![1.; 128]), ("feature", r.clone())] {
        let rmax = weights.iter().copied().fold(0., f64::max);
        let potential = crate::metric_analysis::potential(&w, &m, &weights, s);
        metrics.push(json!({"metric":name,"rmax":rmax,"before":potential,
            "ordinary_energy_bound":1.9*rmax*potential,
            "ordinary_energy_bound_ratio":1.9*rmax*potential/norm(&w).powi(2),
            "universal_coefficient":stability_number(s,1.,rmax,ENERGY),
            "old_q_coefficient_invalid_as_forecast":stability_number(s,q0,rmax,ENERGY)}));
    }
    let horizons: Vec<_> = checkpoints::HORIZONS
        .into_iter()
        .filter(|h| anchor + h <= 10000)
        .collect();
    let mut previous = 0;
    let mut maxima = vec![0f64; 100];
    let mut forecasts = vec![];
    for horizon in horizons {
        for h in previous + 1..=horizon {
            tube.advance(h, state::rate(cohort, anchor + h - 1));
            for (max, r) in maxima.iter_mut().zip(tube.displacement()) {
                *max = max.max(r);
            }
        }
        previous = horizon;
        let caps = guarded::caps(shape, &maxima)?;
        let row_motion = (0..97)
            .filter(|u| !active[*u])
            .flat_map(|u| {
                let rows = &caps.rows;
                (0..97)
                    .filter(move |c| *c != u)
                    .map(move |c| rows[u] + rows[c])
            })
            .fold(0., f64::max);
        let features = trace.feature_radii(&data.train, &caps);
        let execution_change = if horizon == 0 {
            0.
        } else {
            (2. * contrast + row_motion) * 1e-3 + 4e-3
        };
        let budgets: Vec<_> = features
            .iter()
            .zip(&trace.norms)
            .map(|(rf, n)| contrast_radius(contrast, n[8], row_motion, *rf, execution_change))
            .collect();
        let probabilities: Vec<_> = old_q
            .iter()
            .zip(&budgets)
            .map(|(q, b)| ceiling(*q, *b))
            .collect();
        let q = probabilities.iter().copied().fold(0., f64::max);
        forecasts.push(json!({"horizon":horizon,"parameter_caps":maxima,"feature_radii":features,
            "contrast_budgets":budgets,"probability_ceilings":probabilities,"q":q,
            "coefficients":metrics.iter().map(|m|stability_number(s,q,m["rmax"].as_f64().unwrap(),ENERGY)).collect::<Vec<_>>() }));
    }
    Ok(
        json!({"cohort":cohort,"seed":seed,"anchor":anchor,"active":active,"alpha":alpha,"r":r,"eta":eta,"s":s,
        "old_q":q0,"old_query_probabilities":old_q,"training_inputs":4704,"old_contrast_norm":contrast,
        "old_execution_checks":execution,"metrics":metrics,"forecasts":forecasts}),
    )
}
