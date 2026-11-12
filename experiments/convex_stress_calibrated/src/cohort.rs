//! Frozen original data and endpoint-only evaluation for the loss ablation.
use convex_stress_calibrated::{
    AdaptiveModel, AdaptiveTrainer, Aligned, Readout, Result, data, features::Adapter, hash,
    input::Observation, metrics, random::Random,
};
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path, time::Instant};

pub const SUITE: &str = include_str!("../../convex_stress/suite.json");
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub fn fingerprint() -> String {
    let sources: &[&[u8]] = &[
        include_bytes!("lib.rs"),
        include_bytes!("loss.rs"),
        include_bytes!("cohort.rs"),
        include_bytes!("main.rs"),
        include_bytes!("summary.rs"),
        include_bytes!("tests.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        include_bytes!("../run_suite.sh"),
        include_bytes!("../../convex_stress_counts/src/lib.rs"),
        include_bytes!("../../convex_stress_counts/src/counts.rs"),
        include_bytes!("../../convex_stress_counts/src/persistent.rs"),
        include_bytes!("../../convex_stress_repair/src/lib.rs"),
        include_bytes!("../../convex_stress_minibatch/src/lib.rs"),
        include_bytes!("../../convex_stress_minibatch/src/input.rs"),
        include_bytes!("../../convex_stress_minibatch/src/metrics.rs"),
        include_bytes!("../../convex_stress/src/features.rs"),
        include_bytes!("../../convex_stress/src/channel.rs"),
        include_bytes!("../../convex_stress/src/vendor/data.rs"),
        include_bytes!("../../convex_stress/src/vendor/arithmetic_data.rs"),
        include_bytes!("../../convex_stress/src/vendor/random.rs"),
        include_bytes!("../../convex_stress/src/vendor/pcg.rs"),
        include_bytes!("../../convex_stress/src/vendor/blake.rs"),
        SUITE.as_bytes(),
        PROTOCOL.as_bytes(),
    ];
    let mut bytes = vec![];
    for source in sources {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    hash(&bytes)
}
pub fn rows() -> Result<Vec<Value>> {
    Ok(serde_json::from_str::<Value>(SUITE)?["tasks"]
        .as_array()
        .ok_or("no tasks")?
        .clone())
}
fn selected(name: &str) -> Result<Value> {
    rows()?
        .into_iter()
        .find(|r| r["name"] == name)
        .ok_or_else(|| "unknown task".into())
}
pub fn readout(name: &str) -> Result<Readout> {
    let p: Value = serde_json::from_str(PROTOCOL)?;
    match name {
        "scaled" => Ok(Readout::Scaled(p["scale"].as_f64().unwrap())),
        "calibrated" => Ok(Readout::Calibrated(p["epsilon"].as_f64().unwrap())),
        _ => Err("unknown readout".into()),
    }
}
fn features(task: &data::Task) -> Aligned {
    Aligned {
        base: Adapter {
            vocab: task.vocab,
            context: task
                .splits
                .values()
                .flatten()
                .map(|e| {
                    e.tokens
                        .len()
                        .max(e.prompt.len() + e.limit.saturating_sub(1))
                })
                .max()
                .unwrap(),
        },
    }
}
fn encoded(a: Aligned, examples: &[data::Example]) -> (Vec<Observation>, Vec<Vec<usize>>) {
    let mut records = vec![];
    let mut indices = vec![];
    for e in examples {
        let mut ids = vec![];
        for (pos, &y) in e.targets.iter().enumerate() {
            if y < 0 {
                continue;
            }
            ids.push(records.len());
            records.push(a.observation(&e.tokens[..=pos], y as usize));
        }
        indices.push(ids);
    }
    (records, indices)
}
fn all_metrics(model: &AdaptiveModel, task: &data::Task) -> Result<Value> {
    let mut splits = serde_json::Map::new();
    for (name, examples) in &task.splits {
        splits.insert(name.clone(), metrics::evaluate(model, examples)?);
    }
    Ok(splits.into())
}
pub fn passes(v: &Value) -> bool {
    v["exact"]
        .as_u64()
        .zip(v["examples"].as_u64())
        .is_some_and(|(hits, n)| n > 0 && hits * 100 >= n * 99)
}
pub fn run(arm: &str, name: &str, out: &Path) -> Result<()> {
    let readout = readout(arm)?;
    let row = selected(name)?;
    fs::create_dir(out)?;
    let started = Instant::now();
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    let mut result = json!({"task":name,"tier":row["tier"],"arm":arm,
        "source_fingerprint":fingerprint(),"protocol":protocol,
        "data_fingerprints":row["data_fingerprints"],"status":"running"});
    fs::write(
        out.join("metadata.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    // data::build checks the frozen source hashes before the optimizer sees labels.
    let task = data::build(&row)?;
    let a = features(&task);
    let (records, indices) = encoded(a, &task.splits["train"]);
    if records.is_empty() {
        return Err("no supervised prefixes".into());
    }
    let mut trainer = AdaptiveTrainer::new(a, readout, records.len());
    let mut rng = Random::new(protocol["seed"].as_u64().unwrap() as u32 + 10_000);
    let mut trace = fs::File::create(out.join("training.jsonl"))?;
    result["setup_seconds"] = started.elapsed().as_secs_f64().into();
    let (ce, gap, reason) = loop {
        // Preserve the count solver's complete-example shuffle, then all prefixes.
        let mut order: Vec<_> = (0..indices.len()).collect();
        rng.shuffle(&mut order);
        let batch: Vec<_> = order
            .iter()
            .flat_map(|&i| &indices[i])
            .map(|&i| (i, &records[i]))
            .collect();
        trainer.update_batch(&batch);
        let (ce, gap) = trainer.objective(&records);
        if !ce.is_finite()
            || !gap.is_finite()
            || gap < -1e-7
            || ce < readout.ideal_ce(task.vocab) - 1e-8
        {
            return Err("invalid objective or fresh oracle gap".into());
        }
        writeln!(
            trace,
            "{}",
            json!({"update":trainer.updates,"training_ce":ce,
            "fresh_fw_gap":gap,"seconds":started.elapsed().as_secs_f64()})
        )?;
        trace.flush()?;
        println!("update={} ce={ce:.8} gap={gap:.8}", trainer.updates);
        let reason = if ce - readout.ideal_ce(task.vocab)
            <= protocol["ideal_ce_tolerance"].as_f64().unwrap()
        {
            Some("ideal_ce_excess")
        } else if gap <= protocol["gap_tolerance"].as_f64().unwrap() {
            Some("fresh_fw_gap")
        } else if trainer.updates >= protocol["max_updates"].as_u64().unwrap() as usize {
            Some("update_cap")
        } else if started.elapsed().as_secs_f64() >= protocol["training_seconds"].as_f64().unwrap()
        {
            Some("time_cap")
        } else {
            None
        };
        if let Some(reason) = reason {
            break (ce, gap, reason);
        }
    };
    let model = trainer.snapshot();
    model.feasibility_error()?;
    result["selection"] = task.selection.clone().into();
    result["updates"] = trainer.updates.into();
    result["prefixes_presented"] = trainer.observations.into();
    result["train_prefixes"] = records.len().into();
    result["shared_updates"] = trainer.shared_updates.into();
    result["stop_reason"] = reason.into();
    result["training_ce"] = ce.into();
    result["fresh_fw_gap"] = gap.into();
    result["ideal_ce_excess"] = (ce - readout.ideal_ce(task.vocab)).into();
    result["training_seconds_including_setup"] = started.elapsed().as_secs_f64().into();
    drop(trainer);
    drop(records);
    let bytes = serde_json::to_vec(&model)?;
    fs::write(out.join("model.json"), &bytes)?;
    result["model_sha256"] = hash(&bytes).into();
    // Endpoint is immutable before any held-out metric is evaluated.
    result["splits"] = all_metrics(&model, &task)?;
    result["status"] = "complete".into();
    result["total_seconds"] = started.elapsed().as_secs_f64().into();
    fs::write(out.join("result.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
pub fn replay(out: &Path) -> Result<()> {
    let result: Value = serde_json::from_slice(&fs::read(out.join("result.json"))?)?;
    if result["source_fingerprint"] != fingerprint()
        || result["protocol"] != serde_json::from_str::<Value>(PROTOCOL)?
        || result["status"] != "complete"
    {
        return Err("result identity mismatch".into());
    }
    let row = selected(result["task"].as_str().ok_or("missing task")?)?;
    if result["data_fingerprints"] != row["data_fingerprints"] || result["tier"] != row["tier"] {
        return Err("data identity mismatch".into());
    }
    let task = data::build(&row)?;
    let bytes = fs::read(out.join("model.json"))?;
    if result["model_sha256"] != hash(&bytes) {
        return Err("model hash mismatch".into());
    }
    let model: AdaptiveModel = serde_json::from_slice(&bytes)?;
    let a = features(&task);
    if model.readout != readout(result["arm"].as_str().ok_or("missing arm")?)?
        || model.features.base.vocab != a.base.vocab
        || model.features.base.context != a.base.context
        || result["selection"] != task.selection
    {
        return Err("model configuration mismatch".into());
    }
    model.feasibility_error()?;
    let (records, _) = encoded(a, &task.splits["train"]);
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let updates = result["updates"].as_u64().ok_or("missing updates")?;
    if updates == 0
        || updates > p["max_updates"].as_u64().unwrap()
        || result["train_prefixes"] != records.len()
        || result["prefixes_presented"] != updates * records.len() as u64
    {
        return Err("training count mismatch".into());
    }
    let (ce, gap) = model.objective(&records);
    for (key, actual) in [("training_ce", ce), ("fresh_fw_gap", gap)] {
        if (result[key].as_f64().ok_or("missing objective")? - actual).abs() > 1e-7 {
            return Err(format!("replay mismatch: {key}").into());
        }
    }
    if (result["ideal_ce_excess"].as_f64().ok_or("missing excess")?
        - (ce - model.readout.ideal_ce(task.vocab)))
    .abs()
        > 1e-8
    {
        return Err("ideal CE excess mismatch".into());
    }
    let expected_reason =
        if ce - model.readout.ideal_ce(task.vocab) <= p["ideal_ce_tolerance"].as_f64().unwrap() {
            "ideal_ce_excess"
        } else if gap <= p["gap_tolerance"].as_f64().unwrap() {
            "fresh_fw_gap"
        } else if updates == p["max_updates"].as_u64().unwrap() {
            "update_cap"
        } else if result["training_seconds_including_setup"]
            .as_f64()
            .ok_or("missing time")?
            >= p["training_seconds"].as_f64().unwrap()
        {
            "time_cap"
        } else {
            return Err("unjustified stop".into());
        };
    if result["stop_reason"] != expected_reason {
        return Err("stop reason mismatch".into());
    }
    let trace: Vec<Value> = fs::read_to_string(out.join("training.jsonl"))?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    if trace.len() != updates as usize
        || trace.iter().enumerate().any(|(i, r)| r["update"] != i + 1)
    {
        return Err("training trace inventory mismatch".into());
    }
    for key in ["training_ce", "fresh_fw_gap"] {
        if trace.last().unwrap()[key] != result[key] {
            return Err("trace endpoint mismatch".into());
        }
    }
    let actual = all_metrics(&model, &task)?;
    if actual != result["splits"] {
        return Err("split metrics mismatch".into());
    }
    let teacher = actual["train"]
        .get("teacher_forced")
        .unwrap_or(&actual["train"]);
    if (teacher["loss"].as_f64().ok_or("missing teacher CE")? - ce).abs() > 1e-8 {
        return Err("ordinary softmax CE disagrees with optimizer".into());
    }
    println!("replay verified {} {}", result["arm"], result["task"]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn endpoint_replay_rejects_corrupt_objectives_and_configuration() -> Result<()> {
        let base =
            std::env::temp_dir().join(format!("convex-calibrated-test-{}", std::process::id()));
        fs::create_dir(&base)?;
        let cell = base.join("cell");
        run("calibrated", "suite-boolean-and-random", &cell)?;
        replay(&cell)?;
        let result_path = cell.join("result.json");
        let original = fs::read(&result_path)?;
        let mut result: Value = serde_json::from_slice(&original)?;
        result["training_ce"] = (result["training_ce"].as_f64().unwrap() + 0.1).into();
        fs::write(&result_path, serde_json::to_vec(&result)?)?;
        assert!(replay(&cell).is_err());
        fs::write(&result_path, &original)?;
        let mut result: Value = serde_json::from_slice(&original)?;
        let model_path = cell.join("model.json");
        let mut model: AdaptiveModel = serde_json::from_slice(&fs::read(&model_path)?)?;
        model.readout = Readout::Calibrated(0.02);
        let bytes = serde_json::to_vec(&model)?;
        result["model_sha256"] = hash(&bytes).into();
        fs::write(&model_path, &bytes)?;
        fs::write(&result_path, serde_json::to_vec(&result)?)?;
        assert!(replay(&cell).is_err());
        fs::write(base.join("fingerprint.txt"), fingerprint())?;
        assert!(crate::summary::summarize(&base).is_err());
        fs::remove_dir_all(base)?;
        Ok(())
    }
}
