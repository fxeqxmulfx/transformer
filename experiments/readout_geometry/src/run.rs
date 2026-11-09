//! One actual source step: old geometry, forecast persistence, then future audit.
use crate::{
    algebra::linear,
    application::{Backend, State},
    attribution, audit,
    bridge::Bridge,
    domain::{
        Result,
        model::{Arm, E, W},
    },
    forcing::Direction,
    forecast_network, groups,
    infrastructure::{engine::Engine, storage},
    inputs::{Case, require},
    intervals::Frozen,
    measure,
    network::{self, Factors},
    reconstruction::Family,
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
    let old = measure::snapshot(engine, case.data, true)?;
    let training = measure::training(case.data);
    let indices: Vec<_> = training.iter().map(|&(q, _)| q).collect();
    let labels: Vec<_> = training.iter().map(|&(_, t)| t).collect();
    let prior: Vec<_> = prior_state.parameters.iter().map(|&v| v as f64).collect();
    let (_, partial) = crate::gradient::loss_gradient(
        &crate::geometry::selected(&old.features, &indices, W),
        &labels,
        &prior[..E],
        W,
        97,
    );
    let gradient_error = partial
        .iter()
        .zip(&old.readout)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    let gradient_scale = partial.iter().map(|x| x.abs()).fold(0., f64::max);
    require(
        gradient_error <= 2e-6 + 2e-4 * gradient_scale,
        "full-training partial readout gradient mismatch",
    )?;
    let old_bytes = storage::state_bytes(case.step, prior_state);
    fs::write(directory.join("old-state.bin"), &old_bytes)?;
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
    let batch = case.plan.boundaries[case.step + 1] - offset;
    engine.forward(false, offset, batch, true)?;
    let (_, batch_full) = engine.audit_gradient(batch)?;
    engine.update(case.step + 1)?;
    let instrumented = storage::state_bytes(case.step + 1, &engine.state()?);
    engine.restore(prior_state)?;
    engine.train_step(
        offset,
        case.plan.boundaries[case.step + 1] - offset,
        case.step + 1,
    )?;
    cuda.sync()?;
    let update_seconds = update.elapsed().as_secs_f64();
    let next_state = engine.state()?;
    let bytes = storage::state_bytes(case.step + 1, &next_state);
    require(
        bytes == instrumented,
        "instrumentation changed ordinary source update",
    )?;
    let direction = Direction::new(
        prior_state,
        &next_state,
        &old.readout,
        &old.full,
        &batch_full,
        case.step + 1,
    );
    let next_hash = storage::hash(&bytes);
    if let Some(expected) = reference {
        require(next_hash == expected, "next state differs from task #196")?;
    }
    fs::write(directory.join("next-state.bin"), &bytes)?;
    let prepare = Instant::now();
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
        Family::Hidden,
    );
    let readout_geometry = groups::construct(
        &inputs,
        &forecasts.output.base.a,
        &prior[..E],
        &training,
        W,
        97,
        Family::Readout,
    );
    require(
        geometry.labels == readout_geometry.labels,
        "basis choice changed transferred labels",
    )?;
    // No validation target is read to construct a query's candidate label.
    for (q, t) in geometry.labels.iter().enumerate() {
        rows[q * 3 + 2] = t.unwrap_or(0) as u32;
    }
    let geometry_seconds = geometry_start.elapsed().as_secs_f64();
    let attribute = Instant::now();
    let splits = tight_audit::split_indices(case.data);
    let mut attributed = attribution::prepare(attribution::Problem {
        prior: &prior,
        next: &next,
        direction: &direction,
        forecast: &forecasts,
        rows: &rows,
        labels: &geometry.labels,
        splits: &splits,
    })?;
    storage::json_write(&directory.join("attribution.json"), &attributed)?;
    let attribution_hash = storage::hash(&fs::read(directory.join("attribution.json"))?);
    attributed.as_object_mut().unwrap().remove("queries");
    let attribution_seconds = attribute.elapsed().as_secs_f64();
    let readout = Instant::now();
    let frozen = Frozen::new(forecasts, &next, &rows)?;
    let bridges = [
        Bridge::new(geometry, &frozen, &prior, &next)?,
        Bridge::new(readout_geometry, &frozen, &prior, &next)?,
    ];
    let readout_seconds = readout.elapsed().as_secs_f64();
    let persist = Instant::now();
    let mut metadata = json!({});
    for (family, bridge) in Family::ALL.into_iter().zip(&bridges) {
        let binary = bridge.persist(&directory.join(format!("{}-bridge.bin", family.name())))?;
        metadata[family.name()] = json!({"binary":binary,"groups":bridge.geometry,"old_geometry_max_violation":bridge.old_geometry_violation});
    }
    let forecast_binary = frozen.persist(&directory.join("forecast.bin"))?;
    storage::json_write(
        &directory.join("frozen.json"),
        &json!({"families":metadata,"forecast":forecast_binary,"next_state_sha256":next_hash,"attribution_sha256":attribution_hash}),
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
    let mut results = json!({});
    for (family, bridge) in Family::ALL.into_iter().zip(&bridges) {
        let mut result = audit::run(bridge, &old, &new, case.data)?;
        storage::json_write(
            &directory.join(format!("{}-bridge-queries.json", family.name())),
            &result["queries"],
        )?;
        result.as_object_mut().unwrap().remove("queries");
        results[family.name()] = result;
    }
    // Check the geometry endpoints against independent future FP64 logits too.
    let exact = linear(&actual.output.base.a, 9409, W, &next[..E], 97);
    let mut real_violations = 0;
    for bridge in &bridges {
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
    }
    let passed = Family::ALL
        .iter()
        .all(|f| results[f.name()]["passed"] == true)
        && source["passed"] == true
        && real_violations == 0;
    Ok(
        json!({"passed":passed,"history_match":history_match,"next_state_sha256":next_hash,
        "task196_state_match":reference.map(|_|true),"old_metrics":old.metrics,"new_metrics":new.metrics,
        "independent_source_checks":independent,"tangent_scaled_error":tangent_error,
        "bridge_metadata":metadata,"forecast_binary":forecast_binary,"old_state_sha256":storage::hash(&old_bytes),
        "instrumented_step_matches":true,"full_train_readout_gradient_error":gradient_error,"attribution":attributed,"attribution_sha256":attribution_hash,
        "future_real_violations":real_violations,"bridges":results,"source":source,
        "seconds":{"update_and_replay":update_seconds,"forecast":forecast_seconds,"geometry":geometry_seconds,"attribution":attribution_seconds,"readout":readout_seconds,"persistence":persistence_seconds,"audit":check.elapsed().as_secs_f64()}}),
    )
}
