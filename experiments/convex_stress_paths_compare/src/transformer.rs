//! Retain all35 original transformer endpoints and every observed checkpoint.
use crate::*;
use std::io::{BufRead, BufReader};
pub(crate) fn load(
    root: &Path,
    inv: &Value,
    summary: &Value,
    suite: &Value,
    loaded: &mut Loaded,
) -> Result<()> {
    check(
        summary["producer_commit"] == inv["producer_commit"]
            && summary["cells"] == 35
            && summary["all35_conventional_transformer_evaluated"] == true
            && rows(summary, "results")?.len() == 35
            && rows(inv, "cases")?.len() == 35
            && rows(inv, "specifications")?.len() == 35,
        "incomplete original transformer parent",
    )?;
    let (mut primary_count, mut all_count, mut caps, mut seconds, mut updates) =
        (0, 0, 0, 0., 0u64);
    for task in rows(suite, "tasks")? {
        let name = text(task, "name")?;
        let case = json!({"task":name,"seed":1});
        let id = format!("transformer--{name}--seed1");
        check(
            rows(inv, "cases")?.iter().filter(|c| **c == case).count() == 1,
            "missing or duplicate original transformer case",
        )?;
        let specs: Vec<_> = rows(inv, "specifications")?
            .iter()
            .filter(|r| r["case"] == case && r["id"] == id)
            .collect();
        let saved: Vec<_> = rows(summary, "results")?
            .iter()
            .filter(|r| r["case"] == case)
            .collect();
        check(
            specs.len() == 1 && saved.len() == 1,
            "missing or duplicate transformer specification or endpoint",
        )?;
        let (spec, row) = (specs[0], saved[0]);
        data(suite, name, &spec["data_fingerprints"])?;
        let dir = root.join(&id);
        let result = loaded.evidence.load(&dir.join("result.json"))?;
        let replay = loaded
            .evidence
            .load(&root.join(format!("{id}.replay.json")))?;
        loaded.evidence.bind(
            &dir.join("result.json"),
            Some(text(&replay, "result_sha256")?),
        )?;
        check(
            result["status"] == "complete"
                && result["case"] == case
                && result["id"] == id
                && result["metadata"]["specification"] == *spec
                && result["metadata"]["producer_commit"] == inv["producer_commit"]
                && result["metadata"] == loaded.evidence.load(&dir.join("metadata.json"))?
                && replay["replayed"] == true
                && replay["id"] == id
                && replay["source_fingerprint"] == inv["source_fingerprint"]
                && replay["final_update_bit_exact"] == true
                && replay["original_metrics_replayed"] == true
                && replay["complete_training_rerun"] == false,
            "transformer replay or metadata changed",
        )?;
        for (file, key) in [
            ("updates.jsonl", "updates_sha256"),
            ("observations.jsonl", "observations_sha256"),
        ] {
            check(replay[key] == result[key], "transformer trace seal differs")?;
            loaded
                .evidence
                .bind(&dir.join(file), Some(text(&result, key)?))?;
        }
        let mut observations = 0usize;
        for line in BufReader::new(fs::File::open(dir.join("observations.jsonl"))?).lines() {
            let line = line?;
            check(
                line.len() <= 64 * 1024 * 1024,
                "transformer observation exceeds64MiB",
            )?;
            let record: Value = serde_json::from_str(&line)?;
            let step = record["step"].as_u64().ok_or("missing observation step")?;
            let mut snapshots = vec![("checkpoint", format!("at-{step:06}.bin"))];
            if step > 0 {
                snapshots.push(("before", format!("before-{step:06}.bin")));
            }
            for (key, file) in snapshots {
                check(
                    record[key]["file"] == file,
                    "transformer checkpoint filename changed",
                )?;
                loaded
                    .evidence
                    .bind(&dir.join(&file), Some(text(&record[key], "file_sha256")?))?;
            }
            observations += 1;
        }
        check(
            result["observations"] == observations
                && replay["observations_replayed"] == observations,
            "observed checkpoint coverage changed",
        )?;
        let endpoint = &result["endpoint"]["checkpoint"];
        loaded.evidence.bind(
            &dir.join(artifacts::local(text(endpoint, "file")?)?),
            Some(text(endpoint, "file_sha256")?),
        )?;
        check(
            replay["endpoint_body_sha256"] == endpoint["body_sha256"]
                && row["checkpoint_body_sha256"] == endpoint["body_sha256"],
            "transformer endpoint seal changed",
        )?;
        for key in [
            "metrics",
            "primary_split",
            "progress",
            "stop_reason",
            "total_seconds_including_data",
            "observations",
        ] {
            check(
                row[key] == result[key],
                "transformer summary differs from replayed endpoint",
            )?;
        }
        let (primary, all) = report::passes(&result["metrics"], text(&result, "primary_split")?)?;
        check(
            row["primary_pass"] == primary && row["all_splits_pass"] == all,
            "transformer original success flags changed",
        )?;
        primary_count += usize::from(primary);
        all_count += usize::from(all);
        caps += usize::from(result["stop_reason"] == "original_update_cap");
        seconds += number(&result, "total_seconds_including_data")?;
        updates += result["progress"]["completed_steps"]
            .as_u64()
            .ok_or("missing transformer updates")?;
        loaded.transformers.push(json!({"endpoint":row,"result":loaded.evidence.bind(&dir.join("result.json"),None)?,
            "replay":loaded.evidence.bind(&root.join(format!("{id}.replay.json")),None)?,"charged_once":true}));
    }
    check(
        summary["primary_pass"] == primary_count
            && summary["all_split_pass"] == all_count
            && summary["original_cap_failures"] == caps,
        "transformer accounting changed",
    )?;
    loaded.accounting.insert("transformer".into(),json!({"endpoints":35,"original_cap_failures":caps,
        "primary_passes":primary_count,"all_split_passes":all_count,"updates":updates,
        "total_run_seconds":seconds,"replay_cost":"separate preserved review logs","charged_once":true,
        "matched_compute":false}));
    Ok(())
}
