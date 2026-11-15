//! Authenticate saved inputs and independently recount one already scored case.
use crate::*;
use serde_json::{Value, json};
use std::{fs, path::Path};

fn bind(root: &Path, binary: &Path, producer: &str) -> Result<Value> {
    let inv: Value = artifacts::read(&root.join("inventory.json"))?;
    if inv["producer_commit"] != producer
        || inv["binary_sha256"] != artifacts::digest(binary)?
        || inv["protocol"] != serde_json::from_str::<Value>(PROTOCOL)?
        || inv["protocol_sha256"] != dataset::hash(PROTOCOL.as_bytes())
        || inv["cases"] != json!(dataset::CASES)
    {
        return Err("frozen producer, binary, protocol or case list differs".into());
    }
    for case in dataset::CASES {
        let dir = root.join(case);
        let seal: Value = artifacts::read(&dir.join("seal.json"))?;
        if seal["case"] != case
            || seal["endpoint_count"] != 6
            || seal["all_modes_and_priors_sealed_before_scoring"] != true
            || seal["state_sha256"] != artifacts::digest(&dir.join("state.json"))?
        {
            return Err("all thirteen complete training seals are required".into());
        }
    }
    Ok(inv)
}
pub fn case(root: &Path, binary: &Path, producer: &str, case: &str, output: &Path) -> Result<()> {
    if !dataset::CASES.contains(&case) || output.starts_with(root) {
        return Err("original case and external fresh audit output required".into());
    }
    let inv = bind(root, binary, producer)?;
    let dir = root.join(case);
    // A metric must already exist before regenerating its heldout source rows.
    let metric: Value = artifacts::read(&dir.join("metrics.json"))?;
    let seal: Value = artifacts::read(&dir.join("seal.json"))?;
    let state = dir.join("state.json");
    if 64 * fs::metadata(&state)?.len() as u128 + 4 * 1024u128.pow(3) > joint::MAX_BYTES {
        return Err("readback state plus data buffer refused".into());
    }
    let prepared: runner::Prepared = artifacts::read(&state)?;
    let data = from_data(dataset::dataset::build(case))?;
    let metadata = from_data(dataset::dataset::metadata(&data))?;
    let original: Vec<_> = inv["data_cases"]
        .as_array()
        .ok_or("missing original data metadata")?
        .iter()
        .filter(|x| x["case"] == case)
        .collect();
    if original.len() != 1
        || original[0]["data"] != metadata
        || prepared.metadata != metadata
        || metric["seal"] != seal
        || metric["results"]["case"] != case
        || metric["results"]["selection"] != data.task.selection
    {
        return Err("original source metadata, selection or saved state differs".into());
    }
    let endpoints = metric["results"]["endpoints"]
        .as_array()
        .ok_or("missing six endpoints")?;
    if endpoints.len() != 6 {
        return Err("six endpoint denominator differs".into());
    }
    let mut rows = vec![];
    for (i, endpoint) in endpoints.iter().enumerate() {
        let prior = if i < 3 { "Coordinates" } else { "Categorical" };
        let mode = ["Forward", "Full", "Mini"][i % 3];
        if endpoint["prior"] != prior || endpoint["mode"] != mode {
            return Err("endpoint identity differs".into());
        }
        let trained = prepared
            .training
            .as_ref()
            .and_then(|s| s.endpoints.get(i))
            .and_then(|e| e.trained.as_ref());
        if !endpoint["error"].is_null() {
            if endpoint["measurement"].is_object() || endpoint["all_heldout99"] != false {
                return Err("refused endpoint credited with accuracy".into());
            }
            rows.push(
                json!({"prior":prior,"mode":mode,"refusal":endpoint["error"],"measured":false}),
            );
            continue;
        }
        let trained = trained.ok_or("measured endpoint has no trained model")?;
        let input = &prepared
            .training
            .as_ref()
            .ok_or("missing training state")?
            .input;
        trained.model.head.validate()?;
        trained.training.domain.validate()?;
        if trained.training.head != trained.model.head
            || serde_json::to_value(trained.model.prior)? != prior
            || serde_json::to_value(trained.training.policy.mode)? != mode
        {
            return Err("complete trained head or endpoint policy differs".into());
        }
        let forecast = evaluation::forecast(
            input.vocabulary,
            trained.model.programs.len(),
            &data.task.splits,
        )?;
        let file = format!("raw-{i}.jsonl");
        let measurement = &endpoint["measurement"];
        if measurement["raw_file"] != file {
            return Err("raw filename differs".into());
        }
        let readback = raw::verify(
            &trained.model,
            input,
            &data.task.splits,
            &dir.join(&file),
            measurement,
        )?;
        let mut all_raw = true;
        let mut all_exact = true;
        let mut heldout = 0;
        for (name, split) in measurement["splits"]
            .as_object()
            .ok_or("missing split metrics")?
        {
            let g = goals(&split["metrics"])?;
            if split["goals"] != g {
                return Err("integer accuracy threshold differs".into());
            }
            if !name.starts_with("train") {
                heldout += 1;
                all_raw &= g["raw99"] == true;
                all_exact &= g["exact99"] == true;
            }
        }
        if heldout == 0
            || endpoint["all_heldout99"] != (all_raw && all_exact)
            || endpoint["primary"] != measurement["splits"][&data.task.selection]["goals"]
            || endpoint["test"] != measurement["splits"]["test"]["goals"]
        {
            return Err("complete heldout decision differs".into());
        }
        rows.push(
            json!({"prior":prior,"mode":mode,"measured":true,"raw_sha256":measurement["raw_sha256"],
            "readback":readback,"admitted_evaluation_forecast":forecast,
            "all_heldout_raw99":all_raw,"all_heldout_exact99":all_exact,"heldout_splits":heldout}),
        );
    }
    artifacts::write(
        output,
        &json!({"case":case,"producer_commit":producer,"binary_sha256":inv["binary_sha256"],
        "state_sha256":seal["state_sha256"],"metrics_sha256":artifacts::digest(&dir.join("metrics.json"))?,
        "endpoints":rows,"read_only":true,"independent_source_recount":true}),
    )
}
fn goals(m: &Value) -> Result<Value> {
    let n = m["targets"].as_u64().ok_or("missing raw denominator")? as u128;
    let c = m["correct"].as_u64().ok_or("missing raw count")? as u128;
    let r = m["examples"]
        .as_u64()
        .ok_or("missing sequence denominator")? as u128;
    let e = m["exact"].as_u64().ok_or("missing exact count")? as u128;
    if n == 0 || r == 0 || c > n || e > r {
        return Err("invalid raw score counts".into());
    }
    Ok(json!({"original_success":100*e>=95*r,"raw99":100*c>=99*n,
        "exact99":100*e>=99*r,"both99":100*c>=99*n && 100*e>=99*r}))
}
