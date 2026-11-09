//! One actual source step: old geometry, forecast persistence, then future audit.
use crate::{
    algebra::linear,
    application::{Backend, State},
    audit,
    bridge::Bridge,
    domain::{
        Result,
        model::{Arm, E, W},
    },
    forecast_network, groups,
    infrastructure::{engine::Engine, storage},
    inputs::{Case, require},
    intervals::Frozen,
    measure,
    network::{self, Factors},
    tight_audit,
};
use kernite_cuda::Cuda;
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};
pub fn probe(
    engine: &mut Engine,
    case: &Case<'_>,
    prior_state: &State,
    directory: &Path,
    history: &Value,
    reference: Option<&str>,
) -> Result<Value> {
    let cuda = Cuda::new()?;
    let independent = measure::validate(engine, case.arm, &prior_state.parameters, case.data)?;
    let old = measure::snapshot(engine, case.data, false)?;
    let mut history_match = None;
    if case.step.is_multiple_of(100) {
        let saved = &history[case.step / 100];
        require(saved["step"] == case.step, "missing saved history row")?;
        for name in ["train", "validation"] {
            require(
                old.metrics[name]["correct"] == saved[name]["correct"]
                    && (old.metrics[name]["loss"].as_f64().unwrap()
                        - saved[name]["loss"].as_f64().unwrap())
                    .abs()
                        < 1e-12,
                "replayed history differs",
            )?;
        }
        history_match = Some(true);
    }
    let difficult = tight_audit::difficult(case.data, &old);
    let update = Instant::now();
    let offset = case.plan.boundaries[case.step];
    engine.train_step(
        offset,
        case.plan.boundaries[case.step + 1] - offset,
        case.step + 1,
    )?;
    cuda.sync()?;
    let update_seconds = update.elapsed().as_secs_f64();
    let next_state = engine.state()?;
    let bytes = storage::state_bytes(case.step + 1, &next_state);
    let next_hash = storage::hash(&bytes);
    if let Some(expected) = reference {
        require(next_hash == expected, "next state differs from task #196")?;
    }
    fs::write(directory.join("next-state.bin"), &bytes)?;
    let prepare = Instant::now();
    let prior: Vec<_> = prior_state.parameters.iter().map(|&v| v as f64).collect();
    let next: Vec<_> = next_state.parameters.iter().map(|&v| v as f64).collect();
    let delta: Vec<_> = next.iter().zip(&prior).map(|(a, b)| a - b).collect();
    let tangent_error =
        forecast_network::tangent_check(Arm::Plain, &prior, &delta, &case.data.train[..24])?;
    let factors = Factors::new(case.arm, &next);
    // Third coordinates are placeholders: feature computation does not read labels.
    let mut rows: Vec<u32> = (0..97)
        .flat_map(|a| (0..97).flat_map(move |b| [a, b, 0]))
        .collect();
    let forecasts = network::features(case.arm, &prior, &delta, &rows, &factors);
    let forecast_seconds = prepare.elapsed().as_secs_f64();
    let geometry_start = Instant::now();
    let inputs: Vec<_> = rows
        .as_chunks::<3>()
        .0
        .iter()
        .map(|r| (r[0] as usize, r[1] as usize))
        .collect();
    let geometry = groups::construct(
        &inputs,
        &forecasts.output.base.a,
        &prior[..E],
        &measure::training(case.data),
        W,
        97,
    );
    // No validation target is read to construct a query's candidate label.
    for (q, t) in geometry.labels.iter().enumerate() {
        rows[q * 3 + 2] = t.unwrap_or(0) as u32;
    }
    let geometry_seconds = geometry_start.elapsed().as_secs_f64();
    let readout = Instant::now();
    let frozen = Frozen::new(forecasts, &next, &rows)?;
    let bridge = Bridge::new(geometry, &frozen, &prior, &next)?;
    let readout_seconds = readout.elapsed().as_secs_f64();
    let persist = Instant::now();
    let bridge_binary = bridge.persist(&directory.join("bridge.bin"))?;
    let forecast_binary = frozen.persist(&directory.join("forecast.bin"))?;
    storage::json_write(
        &directory.join("frozen.json"),
        &json!({"bridge":bridge_binary,"forecast":forecast_binary,"next_state_sha256":next_hash,"groups":bridge.geometry}),
    )?;
    let persistence_seconds = persist.elapsed().as_secs_f64();
    // The first updated-network forward evaluation occurs after both freezes.
    let check = Instant::now();
    let new = measure::snapshot(engine, case.data, false)?;
    let actual = network::features(case.arm, &next, &vec![0.; next.len()], &rows, &factors);
    let source = tight_audit::run(tight_audit::Problem {
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
    let mut result = audit::run(&bridge, &old, &new, case.data)?;
    storage::json_write(&directory.join("bridge-queries.json"), &result["queries"])?;
    result.as_object_mut().unwrap().remove("queries");
    // Check the geometry endpoints against independent future FP64 logits too.
    let exact = linear(&actual.output.base.a, 9409, W, &next[..E], 97);
    let mut real_violations = 0;
    for (q, label) in bridge.labels.iter().enumerate() {
        let Some(t) = *label else {
            continue;
        };
        for (m, u) in bridge.upper.iter().enumerate().filter(|(m, _)| m % 3 != 0) {
            let _ = m;
            for c in 0..97 {
                let gap = exact[q * 97 + c] - exact[q * 97 + t];
                real_violations += usize::from(
                    gap > u[q * 97 + c] + crate::query_step::tolerance(&[gap, u[q * 97 + c]]),
                );
            }
        }
    }
    let passed = result["passed"] == true && source["passed"] == true && real_violations == 0;
    Ok(
        json!({"passed":passed,"history_match":history_match,"next_state_sha256":next_hash,
        "task196_state_match":reference.map(|_|true),"old_metrics":old.metrics,"new_metrics":new.metrics,
        "independent_source_checks":independent,"tangent_scaled_error":tangent_error,
        "bridge_binary":bridge_binary,"forecast_binary":forecast_binary,"old_geometry_max_violation":bridge.old_geometry_violation,
        "future_real_violations":real_violations,"bridge":result,"source":source,
        "seconds":{"update":update_seconds,"forecast":forecast_seconds,"geometry":geometry_seconds,"readout":readout_seconds,"persistence":persistence_seconds,"audit":check.elapsed().as_secs_f64()}}),
    )
}
