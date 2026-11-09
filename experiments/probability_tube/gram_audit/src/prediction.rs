//! No file access. The only state argument is the named old checkpoint.
use crate::{
    CAP, Result,
    application::State,
    domain::data::{Data, Plan},
    evidence::require,
    gpu, observe, sources, state,
};
use probability_gram::{Covariance, batch_indices, batch_window, transport};
use serde_json::{Value, json};
use std::time::Instant;

pub fn make(
    old: &State,
    data: &Data,
    plan: &Plan,
    frozen: [&Value; 2],
    started: Instant,
    charged: f64,
) -> Result<Value> {
    let f = frozen[0];
    for key in [
        "cohort",
        "seed",
        "anchor",
        "active",
        "alpha",
        "r",
        "eta",
        "s",
        "old_q",
        "old_query_probabilities",
        "training_inputs",
        "metrics",
    ] {
        require(f[key] == frozen[1][key], "changed old input or metric")?;
    }
    let anchor = f["anchor"].as_u64().ok_or("anchor")? as usize;
    let active = sources::support(data);
    require(f["active"] == json!(active), "active classes")?;
    let current = observe::readout(old);
    let (alpha, r) = state::metric(&current, &active, anchor);
    let eta = state::rate(f["cohort"].as_str().ok_or("cohort")?, anchor);
    let s = eta * alpha / (1. - 0.9f64.powf((anchor + 1) as f64));
    require(
        f["r"] == json!(r) && f["alpha"] == alpha && f["eta"] == eta && f["s"] == s,
        "fixed metric reproduction",
    )?;
    let e = gpu::engine(old, data, plan)?;
    let actual = gpu::snapshot(&e, data)?;
    require(
        f["old_query_probabilities"] == json!(observe::probabilities(&actual.logits, &active)),
        "old GPU probability reproduction",
    )?;
    let p: Vec<_> = old.parameters.iter().map(|v| *v as f64).collect();
    let execution = observe::execution(&p, &actual, &data.train)?;
    let mut forecasts = vec![];
    for family in 0..2 {
        let rows = frozen[family]["forecasts"].as_array().ok_or("forecasts")?;
        let expected: Vec<_> = crate::checkpoints::HORIZONS
            .into_iter()
            .filter(|h| anchor + h <= 10000)
            .collect();
        require(rows.len() == expected.len(), "horizon inventory")?;
        for (forecast, horizon) in rows.iter().zip(expected) {
            require(forecast["horizon"] == horizon, "source horizon")?;
            require(
                charged + started.elapsed().as_secs_f64() < CAP,
                "construction budget",
            )?;
            let (q, radii) = observe::converted(forecast)?;
            let maximum = q.iter().copied().fold(0., f64::max);
            let mut maxima = [[f64::NEG_INFINITY; 3]; 2];
            let mut worst = [[anchor; 3]; 2];
            let mut evaluated = [0usize; 2];
            let mut dominated = [0usize; 2];
            let mut maximum_entry_error = 0f64;
            let metrics = [vec![1.; 128], r.clone()];
            let rmax = metrics
                .each_ref()
                .map(|r| r.iter().copied().fold(0., f64::max));
            let window = batch_window(anchor, horizon, 10000)?;
            let batch_count = window.len();
            let mut tail_batches = 0;
            for step in window {
                let indices = batch_indices(&plan.order, &plan.boundaries, step)?;
                require(
                    indices.len() == 512 || indices.len() == 96,
                    "source batch size",
                )?;
                tail_batches += usize::from(indices.len() == 96);
                let qb: Vec<_> = indices.iter().map(|i| q[*i as usize]).collect();
                let sb: Vec<_> = indices.iter().map(|i| radii[*i as usize]).collect();
                // Zero here is only a lower comparison for branch dominance.
                // It is never used as an upper bound on the old operator.
                let floors = [
                    transport(0., &qb, &sb, rmax[0], observe::ENERGY, maximum)?,
                    transport(0., &qb, &sb, rmax[1], observe::ENERGY, maximum)?,
                ];
                let needs = floors.map(|v| v.transported_gram < v.row_trace);
                let covariance = if needs.iter().any(|b| *b) {
                    let features: Vec<_> = indices
                        .iter()
                        .flat_map(|i| {
                            actual.features[*i as usize * 128..(*i as usize + 1) * 128]
                                .iter()
                                .copied()
                        })
                        .collect();
                    Some(Covariance::new(&features, &qb, 128)?)
                } else {
                    None
                };
                if let Some(c) = &covariance {
                    maximum_entry_error = maximum_entry_error.max(c.maximum_entry_error);
                }
                for metric in 0..2 {
                    let bound = if needs[metric] {
                        evaluated[metric] += 1;
                        let old = covariance
                            .as_ref()
                            .ok_or("missing covariance")?
                            .bound(&metrics[metric])?;
                        transport(
                            old.selected,
                            &qb,
                            &sb,
                            rmax[metric],
                            observe::ENERGY,
                            maximum,
                        )?
                        .selected
                    } else {
                        dominated[metric] += 1;
                        floors[metric].row_trace
                    };
                    for (method, value) in [floors[metric].uniform, floors[metric].row_trace, bound]
                        .into_iter()
                        .enumerate()
                    {
                        if value > maxima[metric][method] {
                            maxima[metric][method] = value;
                            worst[metric][method] = step;
                        }
                    }
                }
            }
            let metrics:Vec<_>=(0..2).map(|metric|json!({"metric":(["isotropic","feature"][metric]),
                "L":maxima[metric],"stability":maxima[metric].map(|l|((19.*s).next_up()*l).next_up()),
                "worst_batch":worst[metric],"gram_batches":evaluated[metric],"trace_dominated_batches":dominated[metric]})).collect();
            forecasts.push(json!({"family":(["original","refined"][family]),"horizon":horizon,
                "batch_count":batch_count,"tail_batches":tail_batches,"maximum_entry_error":maximum_entry_error,"metrics":metrics}));
        }
    }
    Ok(
        json!({"cohort":f["cohort"],"seed":f["seed"],"anchor":anchor,"active":active,"r":r,
        "alpha":alpha,"eta":eta,"s":s,"metrics":f["metrics"],"old_execution":execution,
        "methods":["uniform","row_trace","gram"],"forecasts":forecasts}),
    )
}
