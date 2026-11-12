use convex_stress_hull::{
    Expert, Result, all_metrics, counts, data, encoded, features, hash, ideal_ce, meta,
    pair_eligibility, scale, spectral, teacher_ce,
};
use serde_json::{Value, json};
use std::{fs, path::Path};

pub fn identity(report: &Value) -> Result<(Value, data::Task)> {
    let p = meta::protocol()?;
    let row = meta::selected(report["task"].as_str().ok_or("missing task")?)?;
    let task = data::build(&row)?;
    if report["source_fingerprint"] != meta::fingerprint()
        || report["protocol"] != p
        || report["data_fingerprints"] != row["data_fingerprints"]
        || report["seed"] != p["seed"]
        || report["vocab"] != task.vocab
        || report["scale"] != scale(&task)
        || report["selection"] != task.selection
        || report["pair_ineligible_reason"]
            != json!(pair_eligibility(task.vocab, &task.splits["train"]))
        || !meta::allowed(&p, "arms", report["arm"].as_str().ok_or("missing arm")?)
        || !meta::allowed(
            &p,
            "candidates",
            report["candidate"].as_str().ok_or("missing candidate")?,
        )
        || (meta::number(report, "ideal_ce")? - ideal_ce(task.vocab, scale(&task))).abs() > 1e-12
    {
        return Err("source/protocol/data/configuration mismatch".into());
    }
    Ok((row, task))
}
pub fn load(out: &Path, report: &Value) -> Result<Expert> {
    let bytes = fs::read(out.join("model.json"))?;
    if report["model_sha256"] != hash(&bytes) {
        return Err("model hash mismatch".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}
pub fn close(a: &Value, b: &Value, tolerance: f64) -> Result<()> {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (
                a.as_f64().ok_or("invalid number")?,
                b.as_f64().ok_or("invalid number")?,
            );
            if !a.is_finite() || !b.is_finite() || (a - b).abs() > tolerance {
                return Err(format!("numeric replay mismatch {a} != {b}").into());
            }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            for (a, b) in a.iter().zip(b) {
                close(a, b, tolerance)?;
            }
        }
        (Value::Object(a), Value::Object(b)) if a.len() == b.len() => {
            for (k, a) in a {
                close(a, b.get(k).ok_or("missing replay key")?, tolerance)?;
            }
        }
        _ if a == b => {}
        _ => return Err("replay value mismatch".into()),
    }
    Ok(())
}
pub fn replay(out: &Path) -> Result<Value> {
    let report: Value = serde_json::from_slice(&fs::read(out.join("result.json"))?)?;
    let (row, task) = identity(&report)?;
    let p = meta::protocol()?;
    let arm = report["arm"].as_str().unwrap();
    let candidate = report["candidate"].as_str().unwrap();
    let status = report["status"].as_str().ok_or("missing status")?;
    let workers = p["frequency_workers"].as_u64().unwrap() as usize;
    if status == "ineligible" {
        if candidate == "counts"
            || pair_eligibility(task.vocab, &task.splits["train"]).is_none()
            || !report["model_sha256"].is_null()
            || report["updates"] != 0
            || !report["metrics"].is_null()
        {
            return Err("invalid ineligible record".into());
        }
    } else {
        let batch = row["case"]["batch"].as_u64().unwrap() as usize;
        let count = task.splits["train"].len();
        let per_epoch = count.div_ceil(batch);
        let updates = report["updates"].as_u64().ok_or("missing updates")? as usize;
        let expected_examples = if arm == "forward_only" {
            count
        } else if arm == "full_batch" {
            count * updates
        } else {
            updates / per_epoch * count + ((updates % per_epoch) * batch).min(count)
        };
        if report["examples_presented"] != expected_examples
            || report["batch_size"] != batch
            || report["updates_per_epoch"] != per_epoch
            || updates == 0
        {
            return Err("invalid exposure metadata".into());
        }
        if status == "fit_failure" {
            if candidate != "counts" || arm != "forward_only" || !report["model_sha256"].is_null() {
                return Err("invalid fit failure".into());
            }
            let (records, _) = encoded(features(&task), &task.splits["train"]);
            let mut learner = counts::Forward::new(features(&task), scale(&task));
            learner.update_batch(&records.iter().collect::<Vec<_>>());
            let failed = learner
                .snapshot()
                .err()
                .ok_or("reported failed fit actually succeeds")?;
            if json!(failed) != report["failed_queries"] {
                return Err("failed queries differ".into());
            }
        } else if status == "complete" {
            let model = load(out, &report)?;
            let feasibility = model.validate(&task, candidate, workers)?;
            close(&feasibility, &report["feasibility"], 1e-8)?;
            if all_metrics(&model, &task)? != report["metrics"] {
                return Err("split metrics differ".into());
            }
            let ce = teacher_ce(&model, &task.splits["train"])?;
            let excess = ce - ideal_ce(task.vocab, scale(&task));
            close(&json!(ce), &report["training_ce"], 1e-12)?;
            close(&json!(excess), &report["ideal_ce_excess"], 1e-12)?;
            match &model {
                Expert::Moments(m) => {
                    let bytes = fs::read(out.join("diagnostic_state.json"))?;
                    if hash(&bytes) != report["diagnostic_state_sha256"] {
                        return Err("state hash mismatch".into());
                    }
                    let mut solver = spectral::solver::Solver::new(
                        m,
                        spectral::samples(m, &task.splits["train"])?,
                        spectral::solver::Mode::parse(arm)?,
                        workers,
                    )?;
                    solver.restore_diagnostic_state(serde_json::from_slice(&bytes)?)?;
                    let (reconstructed, alpha, norms) = solver.snapshot()?;
                    if reconstructed.probabilities != m.probabilities {
                        return Err("state does not reconstruct model".into());
                    }
                    close(&json!(alpha), &report["uniform_mixture_alpha"], 1e-12)?;
                    close(&json!(norms), &report["shadow_moment_norms"], 1e-8)?;
                    close(
                        &json!(solver.full_residual()?),
                        &report["full_operator_residual"],
                        1e-8,
                    )?;
                    close(&json!(solver.fit_defect(m)), &report["fit_defect"], 1e-12)?;
                    if report["train_prefixes"] != count
                        || report["prefixes_presented"] != expected_examples
                    {
                        return Err("spectral prefix exposure mismatch".into());
                    }
                }
                Expert::Counts(m) => {
                    if arm == "forward_only" {
                        let (records, _) = encoded(features(&task), &task.splits["train"]);
                        let mut learner = counts::Forward::new(features(&task), scale(&task));
                        learner.update_batch(&records.iter().collect::<Vec<_>>());
                        let reference = learner
                            .snapshot()
                            .map_err(|_| "complete-data forward unexpectedly failed")?;
                        if serde_json::to_vec(m)? != serde_json::to_vec(&reference)? {
                            return Err("forward batch equivalence failed".into());
                        }
                    } else {
                        close(&json!(ce), &report["fresh_training_ce"], 1e-8)?;
                    }
                }
            }
            let reason = report["stop_reason"]
                .as_str()
                .ok_or("missing stop reason")?;
            if (candidate != "counts" || arm != "forward_only")
                && updates > meta::limit(&p, arm, candidate)
            {
                return Err("update cap exceeded".into());
            }
            let valid = match reason {
                "ideal_ce_excess" => excess <= meta::number(&p, "ideal_ce_tolerance")?,
                "full_fw_gap" => {
                    candidate == "counts"
                        && arm == "full_batch"
                        && meta::number(&report, "fresh_fw_gap")?
                            <= meta::number(&p, "full_gap_tolerance")?
                }
                "forward_complete" => {
                    candidate == "counts" && arm == "forward_only" && updates == per_epoch
                }
                "budget" => updates == meta::limit(&p, arm, candidate),
                _ => false,
            };
            if !valid {
                return Err("invalid stopping condition".into());
            }
        } else {
            return Err("unknown terminal status".into());
        }
    }
    Ok(
        json!({"replayed":true,"task":report["task"],"arm":arm,"candidate":candidate,"status":status}),
    )
}

pub fn compare_original(old: &Path, new: &Path) -> Result<()> {
    let old_report: Value = serde_json::from_slice(&fs::read(old.join("result.json"))?)?;
    let report: Value = serde_json::from_slice(&fs::read(new.join("result.json"))?)?;
    identity(&report)?;
    let bytes = fs::read(old.join("model.json"))?;
    if old_report["source_fingerprint"]
        != "913357a9c11a6f700148dd73bb60d23e8e3f00e601bfb6b24aea5825c62ce742"
        || old_report["model_sha256"] != hash(&bytes)
        || old_report["task"] != report["task"]
        || old_report["arm"] != report["arm"]
        || old_report["budget_name"] != report["candidate"]
    {
        return Err("invalid original control identity".into());
    }
    let old_model: spectral::Model = serde_json::from_slice(&bytes)?;
    let Expert::Moments(model) = load(new, &report)? else {
        return Err("expected moment control".into());
    };
    if model.probabilities != old_model.probabilities {
        return Err("original coordinates changed".into());
    }
    println!(
        "{}",
        json!({"original_coordinates_equal":true,"task":report["task"],"arm":report["arm"],"candidate":report["candidate"]})
    );
    Ok(())
}
