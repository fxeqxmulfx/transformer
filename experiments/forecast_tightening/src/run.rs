//! Restored source updates, complete pre-evaluation freeze, then all-query audit.
use crate::{
    application::Backend,
    audit,
    domain::Result,
    forecast_network,
    infrastructure::{engine::Engine, storage},
    inputs::{Case, Saved, require},
    intervals::Frozen,
    measure,
    network::{self, Factors},
};
use kernite_cuda::Cuda;
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
pub fn run(case: &Case<'_>, saved: &Saved, directory: &Path, expected: &str) -> Result<Value> {
    let cuda = Cuda::new()?;
    let start = Instant::now();
    let mut engine = Engine::new(case.arm, &saved.state.parameters, case.data, case.plan)?;
    engine.restore(&saved.state)?;
    cuda.sync()?;
    let setup_seconds = start.elapsed().as_secs_f64();
    let diagnostic = Instant::now();
    let independent = measure::validate(&engine, case.arm, &saved.state.parameters, case.data)?;
    let old = measure::snapshot(&engine, case.data, false)?;
    for name in ["train", "validation"] {
        require(
            old.metrics[name]["correct"] == saved.last_report[name]["correct"]
                && (old.metrics[name]["loss"].as_f64().unwrap()
                    - saved.last_report[name]["loss"].as_f64().unwrap())
                .abs()
                    < 1e-12,
            "old source metrics differ",
        )?;
    }
    let difficult = audit::difficult(case.data, &old);
    let old_audit_seconds = diagnostic.elapsed().as_secs_f64();
    let update = Instant::now();
    let offset = case.plan.boundaries[case.step];
    let batch = case.plan.boundaries[case.step + 1] - offset;
    engine.train_step(offset, batch, case.step + 1)?;
    cuda.sync()?;
    let source_update_seconds = update.elapsed().as_secs_f64();
    let next = engine.state()?;
    let bytes = storage::state_bytes(case.step + 1, &next);
    let state_hash = storage::hash(&bytes);
    require(state_hash == expected, "next state differs from task #188")?;
    let representation = Instant::now();
    let prior: Vec<_> = saved.state.parameters.iter().map(|&v| v as f64).collect();
    let next: Vec<_> = next.parameters.iter().map(|&v| v as f64).collect();
    let delta: Vec<_> = next.iter().zip(&prior).map(|(n, o)| n - o).collect();
    let representation_seconds = representation.elapsed().as_secs_f64();
    let tangent = Instant::now();
    let tangent_error =
        forecast_network::tangent_check(case.arm, &prior, &delta, &case.data.train[..24])?;
    let tangent_check_seconds = tangent.elapsed().as_secs_f64();
    let prepare = Instant::now();
    let matrices = Instant::now();
    let factors = Factors::new(case.arm, &next);
    let gram_seconds = matrices.elapsed().as_secs_f64();
    let rows = forecast_network::canonical_rows(case.data);
    let propagate = Instant::now();
    let forecasts = network::features(case.arm, &prior, &delta, &rows, &factors);
    let propagation_seconds = propagate.elapsed().as_secs_f64();
    let readout = Instant::now();
    let frozen = Frozen::new(forecasts, &next, &rows)?;
    let readout_seconds = readout.elapsed().as_secs_f64();
    let preparation_seconds = representation_seconds + prepare.elapsed().as_secs_f64();
    let persist = Instant::now();
    let binary = frozen.persist(&directory.join("forecast.bin"))?;
    storage::json_write(
        &directory.join("frozen.json"),
        &json!({"binary":binary,"difficult_queries":difficult,"state_sha256":state_hash,
        "matrix_bounds":[factors.first.json(),factors.second.json()]}),
    )?;
    let persistence_seconds = persist.elapsed().as_secs_f64();
    // No updated-network evaluation has occurred above this line.
    let audit_start = Instant::now();
    let new = measure::snapshot(&engine, case.data, false)?;
    let actual = network::features(case.arm, &next, &vec![0.; next.len()], &rows, &factors);
    let checks = audit::run(audit::Problem {
        frozen: &frozen,
        actual: &actual,
        old: &old,
        new: &new,
        next: &next,
        rows: &rows,
        data: case.data,
        difficult: &difficult,
        directory,
    })?;
    let audit_seconds = audit_start.elapsed().as_secs_f64();
    let (forecast_payload_bytes, forecast_capacity_bytes) = frozen.numeric_storage();
    Ok(
        json!({"passed":checks["passed"],"state_sha256":state_hash,"source_state_match":true,
        "old_metrics":old.metrics,"new_metrics":new.metrics,"independent_source_checks":independent,
        "tangent_scaled_error":tangent_error,"binary":binary,"matrix_bounds":[factors.first.json(),factors.second.json()],
        "forecast_payload_bytes":forecast_payload_bytes,"forecast_capacity_bytes":forecast_capacity_bytes,
        "seconds":{"setup":setup_seconds,"old_audit":old_audit_seconds,"source_update":source_update_seconds,
        "tangent_check":tangent_check_seconds,"prepare":preparation_seconds,"representation":representation_seconds,"gram":gram_seconds,
        "propagation":propagation_seconds,"readout":readout_seconds,"persistence":persistence_seconds,"audit":audit_seconds},"checks":checks}),
    )
}
