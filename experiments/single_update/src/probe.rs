//! One independent restored-state update, then complete paired geometry diagnostics.
use crate::{
    application::Backend,
    domain::{
        Result,
        model::{E, W},
    },
    drift,
    forcing::{self, Direction},
    infrastructure::{engine::Engine, storage},
    inputs::{Case, Saved, require},
    margins::{self, View},
    measure,
};
use serde_json::{Value, json};

pub fn run(case: &Case<'_>, saved: &Saved) -> Result<Value> {
    let mut engine = Engine::new(case.arm, &saved.state.parameters, case.data, case.plan)?;
    engine.restore(&saved.state)?;
    let validation = measure::validate(&engine, case.arm, &saved.state.parameters, case.data)?;
    let old = measure::snapshot(&engine, case.data, true)?;
    let difficult = crate::query_summary::freeze_difficult(&old, case.data);
    let training = measure::training(case.data);
    let indices: Vec<_> = training.iter().map(|&(i, _)| i).collect();
    let labels: Vec<_> = training.iter().map(|&(_, y)| y).collect();
    let old_e: Vec<_> = saved.state.parameters[..E]
        .iter()
        .map(|&v| v as f64)
        .collect();
    let (_, independent) = crate::gradient::loss_gradient(
        &crate::geometry::selected(&old.features, &indices, W),
        &labels,
        &old_e,
        W,
        97,
    );
    let train_readout_error = old
        .readout
        .iter()
        .zip(&independent)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let train_readout_scale = independent.iter().map(|x| x.abs()).fold(0., f64::max);
    require(
        train_readout_error <= 2e-6 + 2e-4 * train_readout_scale,
        "weighted full-training readout gradient mismatch",
    )?;
    for name in ["train", "validation"] {
        require(
            old.metrics[name]["correct"] == saved.last_report[name]["correct"]
                && (old.metrics[name]["loss"].as_f64().unwrap()
                    - saved.last_report[name]["loss"].as_f64().unwrap())
                .abs()
                    < 1e-12,
            "saved source metrics did not replay",
        )?;
    }
    let offset = case.plan.boundaries[case.step];
    let end = case.plan.boundaries[case.step + 1];
    let batch = end - offset;
    engine.forward(false, offset, batch, true)?;
    let (batch_readout, batch_full) = engine.audit_gradient(batch)?;
    engine.update(case.step + 1)?;
    let next = engine.state()?;
    let next_bytes = storage::state_bytes(case.step + 1, &next);
    let forecast =
        crate::forecast_report::prepare(case, &saved.state.parameters, &next.parameters)?;
    engine.restore(&saved.state)?;
    engine.train_step(offset, batch, case.step + 1)?;
    require(
        next_bytes == storage::state_bytes(case.step + 1, &engine.state()?),
        "instrumentation changed source train_step state",
    )?;
    let new = measure::snapshot(&engine, case.data, false)?;
    let direction = Direction::new(
        &saved.state,
        &next,
        &old.readout,
        &old.full,
        &batch_full,
        case.step + 1,
    );
    let new_e: Vec<_> = next.parameters[..E].iter().map(|&v| v as f64).collect();
    let (query_control, query_rows) = crate::query_summary::audit(
        &old, &new, &old_e, &new_e, &direction, case.data, &difficult,
    )?;
    let feature_forecast =
        forecast.audit(case, &old, &new, &next.parameters, &query_rows, &difficult)?;
    let old_groups = measure::groups(&old, case.data);
    let new_groups = measure::groups(&new, case.data);
    let mut results = Vec::new();
    for kind in 0..2 {
        let mut aggregate_energy: [forcing::Energy; 3] = Default::default();
        let mut aggregate_margin: [[margins::Stats; 2]; 3] = Default::default();
        let mut aggregate_drift: [[drift::Stats; 2]; 3] = Default::default();
        let mut residuals = [[0.; 2]; 3];
        let mut records = Vec::new();
        for (g0, g1) in old_groups[kind].iter().zip(&new_groups[kind]) {
            require(
                (g0.key, g0.label, &g0.train, &g0.validation)
                    == (g1.key, g1.label, &g1.train, &g1.validation),
                "group route/label mismatch",
            )?;
            let mut methods = Vec::new();
            for (rank, (b0, b1)) in g0.bases.iter().zip(&g1.bases).enumerate() {
                let parts = direction.project(&b0.q, W);
                let mut local_energy = forcing::Energy::default();
                local_energy.add(&parts);
                aggregate_energy[rank].add(&parts);
                require(
                    local_energy.valid(),
                    "projected gradient decomposition did not close",
                )?;
                let mut splits = Vec::new();
                for (split, indices) in [&g0.train, &g0.validation].iter().enumerate() {
                    let v0 = View::new(&old, &old_e, g0, b0, indices);
                    let v1 = View::new(&new, &new_e, g1, b1, indices);
                    if split == 0 {
                        residuals[rank][0] += margins::residual_energy(&v0);
                        residuals[rank][1] += margins::residual_energy(&v1);
                    }
                    let change =
                        drift::terms(&v0, &v1, &direction, &parts, &b0.q, g0.label, &new_e);
                    let mut local_margin = margins::Stats::default();
                    local_margin.add(&v0, &v1);
                    let mut local_drift = drift::Stats::default();
                    local_drift.add(&v0, &v1, &change);
                    require(
                        local_margin.valid(),
                        "margin reconstruction or certificate failed",
                    )?;
                    require(
                        local_drift.valid(),
                        "finite interaction decomposition did not close",
                    )?;
                    aggregate_margin[rank][split].add(&v0, &v1);
                    aggregate_drift[rank][split].add(&v0, &v1, &change);
                    splits.push(
                        json!({"margins":local_margin.json(),"interaction":local_drift.json()}),
                    );
                }
                methods.push(
                    json!({"rank":b0.requested,"old_new_achieved":[b0.achieved,b1.achieved],
                    "forcing":local_energy.json(),"train":splits[0],"validation":splits[1]}),
                );
            }
            records.push(
                json!({"key":g0.key,"label":g0.label,"training_count":g0.train.len(),
                "validation_count":g0.validation.len(),"methods":methods}),
            );
        }
        let aggregate:Vec<_>=(0..3).map(|r|json!({"rank":crate::geometry::RANKS[r],
            "forcing":aggregate_energy[r].json(),"old_new_contrast_residual_energy":residuals[r],
            "train":{"margins":aggregate_margin[r][0].json(),"interaction":aggregate_drift[r][0].json()},
            "validation":{"margins":aggregate_margin[r][1].json(),"interaction":aggregate_drift[r][1].json()}})).collect();
        results.push(json!({"grouping":if kind==0 {"sum"}else{"orbit"},"aggregate":aggregate,"groups":records}));
    }
    Ok(
        json!({"metadata":saved.metadata,"source_metrics_old":old.metrics,"source_metrics_new":new.metrics,
        "gradient_validation":validation,"full_training_readout_max_error":train_readout_error,
        "bitwise_train_step_match":true,"new_state_sha256":storage::hash(&next_bytes),
        "next_batch":{"start":offset,"size":batch,"rows_sha256":storage::hash(&storage::words(&case.plan.order[offset..end]))},
        "direction":direction.summary(),"old_new_rms_max_error":[old.rms_error,new.rms_error],
        "next_batch_readout_energy":crate::gradient::energy(&batch_readout),"next_batch_full_energy":crate::gradient::energy(&batch_full),
        "query_control":query_control,"feature_forecast":feature_forecast,"geometry":results}),
    )
}
