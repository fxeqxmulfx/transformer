//! Preserve task347's scoring at1371ae4f, adding certified readback and separate
//! all-heldout raw99/exact99 decisions. No target enters numerical inference.
use super::*;

pub(super) fn goals(m: &Value) -> Result<Value> {
    let n = m["targets"].as_u64().ok_or("missing raw denominator")? as u128;
    let c = m["correct"].as_u64().ok_or("missing raw count")? as u128;
    let rows = m["examples"]
        .as_u64()
        .ok_or("missing sequence denominator")? as u128;
    let exact = m["exact"].as_u64().ok_or("missing exact count")? as u128;
    if n == 0 || rows == 0 || c > n || exact > rows {
        return Err("invalid scoring counts".into());
    }
    Ok(
        json!({"original_success":100*exact>=95*rows,"raw99":100*c>=99*n,
        "exact99":100*exact>=99*rows,"both99":100*c>=99*n && 100*exact>=99*rows}),
    )
}

pub(super) fn audit_training(t: &model::Trained) -> Result<Value> {
    t.model.head.validate()?;
    t.training.domain.validate()?;
    if t.training.head != t.model.head {
        return Err("model and training heads differ".into());
    }
    let mut maximum = 0.0f64;
    for p in std::iter::once(&t.training.initial_projection)
        .chain(t.training.steps.iter().map(|s| &s.projection))
    {
        let v = t.training.domain.violation(&p.theta);
        maximum = maximum.max(v);
        if !v.is_finite()
            || v > t.training.policy.projection_tolerance * 1.01
            || p.normalized_step > t.training.policy.projection_tolerance
            || p.normalized_violation > t.training.policy.projection_tolerance
        {
            return Err("saved projection fails original-anchor/budget recheck".into());
        }
    }
    let charge = t.model.head.max_column_charge();
    if charge > t.training.domain.radius * (1. + t.training.policy.projection_tolerance) {
        return Err("complete parameter column exceeds declared budget".into());
    }
    let mut previous = &t.training.initial_projection.theta;
    let mut largest_step = 0.0f64;
    for step in &t.training.steps {
        largest_step = largest_step.max(
            previous
                .iter()
                .flatten()
                .zip(step.projection.theta.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max),
        );
        previous = &step.projection.theta;
    }
    Ok(
        json!({"initial_ce":t.training.initial_ce,"final_ce":t.training.final_ce,
        "work":t.training.work,"forecast":t.training.forecast,"programs":t.model.programs.len(),
        "complete_real_coefficients":t.model.head.coefficients.len(),"complete_charge":joint::charge(&t.model.head.theta),
        "actual_max_column_charge":charge,"radius":t.training.domain.radius,
        "maximum_rechecked_projection_violation":maximum,"minimum_total_mass":t.training.domain.minimum_mass,
        "largest_actual_parameter_step":largest_step,"effective_update_definition":"bitwise parameter change; may include projection roundoff when CE is already numerically zero",
        "final_masses":t.model.head.theta.iter().map(|p|p[5]).collect::<Vec<_>>()}),
    )
}
pub(super) fn measured(
    data: &dataset::dataset::Dataset,
    prepared: &Prepared,
    dir: &Path,
) -> Result<Value> {
    let empty: Vec<_> = [model::Prior::Coordinates, model::Prior::Categorical]
        .into_iter()
        .flat_map(|prior| {
            fit::settings()
                .unwrap()
                .policies
                .into_iter()
                .map(move |policy| fit::Endpoint {
                    prior,
                    policy,
                    trained: None,
                    error: prepared
                        .error
                        .clone()
                        .or(Some("fit construction failed".into())),
                })
        })
        .collect();
    let endpoints = prepared
        .training
        .as_ref()
        .map_or(empty.as_slice(), |s| s.endpoints.as_slice());
    if endpoints.len() != 6 {
        return Err("six endpoint denominator differs".into());
    }
    let mut out = vec![];
    for (index, e) in endpoints.iter().enumerate() {
        let mut error = e.error.clone();
        let (mut measurement, mut training) = (None, None);
        if let Some(t) = &e.trained {
            let result = (|| -> Result<(Value, Value)> {
                let audit = audit_training(t)?;
                let mut result = evaluation::evaluate(
                    &t.model,
                    &prepared.training.as_ref().ok_or("missing input")?.input,
                    &data.task.splits,
                    &dir.join(format!("raw-{index}.jsonl")),
                )?;
                let readback = raw::verify(
                    &t.model,
                    &prepared.training.as_ref().ok_or("missing input")?.input,
                    &data.task.splits,
                    &dir.join(format!("raw-{index}.jsonl")),
                    &result,
                )?;
                result["raw_readback"] = readback;
                let bytes =
                    disk::endpoint(data.task.vocab, t.model.programs.len(), &data.task.splits)?;
                if fs::metadata(dir.join(format!("raw-{index}.jsonl")))?.len() > bytes {
                    return Err("raw evaluation exceeded exact-schema disk bound".into());
                }
                Ok((audit, result))
            })();
            match result {
                Ok((audit, result)) => {
                    training = Some(audit);
                    measurement = Some(result);
                }
                Err(e) => error = Some(e.to_string()),
            }
        }
        let primary = measurement.as_ref().map_or(Value::Null, |m| {
            m["splits"][&data.task.selection]["goals"].clone()
        });
        let test = measurement
            .as_ref()
            .map_or(Value::Null, |m| m["splits"]["test"]["goals"].clone());
        let mut heldout = 0;
        let mut all_heldout99 = error.is_none() && measurement.is_some();
        let mut all_heldout_raw99 = all_heldout99;
        let mut all_heldout_exact99 = all_heldout99;
        if let Some(m) = &measurement {
            let splits = m["splits"]
                .as_object()
                .ok_or("missing all original splits")?;
            if splits.keys().collect::<Vec<_>>() != data.task.splits.keys().collect::<Vec<_>>() {
                return Err("changed original split set".into());
            }
            for (name, s) in splits {
                let computed = goals(&s["metrics"])?;
                if s["goals"] != computed {
                    return Err("goals differ from actual integer counts".into());
                }
                if !name.starts_with("train") {
                    heldout += 1;
                    all_heldout99 &= computed["both99"] == true;
                    all_heldout_raw99 &= computed["raw99"] == true;
                    all_heldout_exact99 &= computed["exact99"] == true;
                }
            }
        }
        all_heldout99 &= heldout > 0;
        all_heldout_raw99 &= heldout > 0;
        all_heldout_exact99 &= heldout > 0;
        out.push(
            json!({"prior":e.prior,"mode":e.policy.mode,"policy":e.policy,"error":error,
            "primary":primary,"test":test,"all_heldout99":all_heldout99,
            "all_heldout_raw99":all_heldout_raw99,"all_heldout_exact99":all_heldout_exact99,
            "measurement":measurement,"training":training}),
        );
    }
    Ok(
        json!({"case":data.name,"selection":data.task.selection,"endpoints":out,
        "complete_case_forecast":prepared.complete_forecast,"target_realizability_certified":false}),
    )
}
pub fn score(root: &Path, case: &str) -> Result<()> {
    let inv = inventory(root)?;
    all_sealed(root)?;
    let (data, prepared, seal) = load(root, &inv, case)?;
    let dir = root.join(case);
    let results = measured(&data, &prepared, &dir)?;
    write(
        &dir.join("metrics.json"),
        &json!({"seal":seal,"results":results}),
    )
}
pub fn replay(root: &Path, case: &str) -> Result<()> {
    let inv = inventory(root)?;
    all_sealed(root)?;
    let (data, prepared, seal) = load(root, &inv, case)?;
    let dir = root.join(case);
    let repeat = dir.join("replay");
    fs::create_dir(&repeat)?;
    let results = measured(&data, &prepared, &repeat)?;
    let repeated = json!({"seal":seal,"results":results});
    write(&repeat.join("metrics.json"), &repeated)?;
    let saved: Value = read(&dir.join("metrics.json"))?;
    if saved != repeated {
        return Err("complete fresh score replay differs".into());
    }
    for i in 0..6 {
        let file = format!("raw-{i}.jsonl");
        let (a, b) = (dir.join(&file), repeat.join(&file));
        if a.exists() != b.exists() || (a.exists() && digest(&a)? != digest(&b)?) {
            return Err("complete raw query vectors differ on replay".into());
        }
    }
    write(
        &dir.join("replay.json"),
        &json!({"case":case,"exact_replay":true,
        "state_sha256":seal["state_sha256"],"metrics_sha256":digest(&dir.join("metrics.json"))?}),
    )
}
