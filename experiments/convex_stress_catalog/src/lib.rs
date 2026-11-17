//! Native reconciliation of the original165 catalog, plan, historical snapshot
//! and later35-task manifest. These are benchmark inputs, not model features.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fmt::Write;
use std::{collections::BTreeMap, path::Path};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const PROTOCOL: &str = include_str!("../protocol.json");

pub fn checked_text(name: &str, path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    let mut hash = String::with_capacity(64);
    for byte in Sha256::digest(&bytes) {
        write!(&mut hash, "{byte:02x}")?;
    }
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    if protocol["input_sha256"][name] != hash {
        return Err(format!("frozen {name} source hash differs: {hash}").into());
    }
    Ok(String::from_utf8(bytes)?)
}

fn keyed<'a>(rows: &'a Value, key: &str, expected: usize) -> Result<BTreeMap<String, &'a Value>> {
    let rows = rows.as_array().ok_or("missing case array")?;
    if rows.len() != expected {
        return Err("case count differs".into());
    }
    let mut result = BTreeMap::new();
    for row in rows {
        let name = row[key].as_str().ok_or("missing case identity")?;
        if result.insert(name.into(), row).is_some() {
            return Err(format!("duplicate case {name}").into());
        }
    }
    Ok(result)
}

/// Read all unique case IDs and historical fields from the original Markdown catalog.
pub fn catalog_rows(text: &str) -> Result<BTreeMap<String, Value>> {
    let mut section = "";
    let mut result = BTreeMap::new();
    for line in text.lines() {
        if let Some(heading) = line.strip_prefix("## ") {
            section = heading;
        }
        if !line.starts_with("| `") {
            continue;
        }
        let fields: Vec<_> = line.split('|').map(str::trim).collect();
        if fields.len() != 6 || !fields[1].ends_with('`') {
            return Err("unexpected catalog row format".into());
        }
        let id = fields[1].trim_matches('`').to_string();
        let row = json!({"description":fields[2],"history":fields[3],
            "tier":fields[4],"section":section});
        if result.insert(id, row).is_some() {
            return Err("duplicate catalog case".into());
        }
    }
    if result.len() != 165 {
        return Err("catalog does not contain165 cases".into());
    }
    Ok(result)
}

fn historical_status(status: &str) -> Result<&str> {
    match status {
        "budget_exhausted" => Ok("budget"),
        "skipped_at_user_request" => Ok("skipped"),
        "stopped_at_user_request" => Ok("stopped"),
        "passed" | "pending" => Ok(status),
        _ => Err("unknown historical status".into()),
    }
}

pub fn inventory(catalog: &str, plan: &Value, existing: &Value, history: &Value) -> Result<Value> {
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    let catalog = catalog_rows(catalog)?;
    let planned = keyed(&plan["cases"], "name", 165)?;
    let previous = keyed(&existing["tasks"], "name", 35)?;
    let recorded = keyed(&history["runs"], "case", 165)?;
    if planned.keys().ne(catalog.keys()) || planned.keys().ne(recorded.keys()) {
        return Err("catalog, plan and history identities differ".into());
    }
    if previous.keys().any(|id| !planned.contains_key(id)) {
        return Err("existing35 contains an unplanned case".into());
    }
    let mut cases = Vec::new();
    let mut statuses = BTreeMap::<String, usize>::new();
    let mut retained = 0;
    for (id, case) in &planned {
        let item = &catalog[id];
        let old = recorded[id];
        let status = old["status"].as_str().ok_or("missing historical status")?;
        let tier = previous
            .get(id)
            .map_or("—", |r| r["tier"].as_str().unwrap_or(""));
        if item["history"] != historical_status(status)?
            || item["tier"] != tier
            || case["metric"] != old["metric"]
            || case["target"] != old["target"]
            || case["kind"] != old["kind"]
            || old["seed"] != 1
            || case["data_seed"] != 1
        {
            return Err(format!("source criterion, status, tier or seed differs for {id}").into());
        }
        *statuses.entry(status.to_string()).or_default() += 1;
        retained += usize::from(old["final_target_met"] == true);
        let tests: Vec<_> = old["final_test"]
            .as_object()
            .map_or_else(Vec::new, |m| m.keys().cloned().collect());
        cases.push(json!({
            "id":id,"catalog":item,"original_case":case,"historical_reference":old,
            "recorded_test_splits":tests,"existing35":previous.get(id),
            "additional_case":!previous.contains_key(id),
            "control":protocol["controls"].as_array().unwrap().iter().any(|x| x == id),
            "first_cohort":protocol["first_cohort"].as_array().unwrap().iter().any(|x| x == id),
            "new_accuracy_targets":{"raw_token":protocol["raw_token_accuracy_target"],
                "exact_sequence":protocol["exact_sequence_accuracy_target"]},
            "expanded_model_result":Value::Null
        }));
    }
    if serde_json::to_value(&statuses)? != history["counts"]
        || retained != 40
        || statuses.get("passed") != Some(&48)
    {
        return Err("historical crossing/retention accounting differs".into());
    }
    for id in protocol["first_cohort"]
        .as_array()
        .ok_or("missing first cohort")?
    {
        let id = id.as_str().ok_or("invalid first cohort identity")?;
        if !planned.contains_key(id)
            || previous.contains_key(id)
            || planned[id]["metric"] != "sequence_accuracy"
            || planned[id]["target"] != 0.95
            || planned[id]["max_updates"] != 100000
            || planned[id]["batch"] != 32
        {
            return Err("first cohort is not an original additional sequence case".into());
        }
    }
    Ok(json!({"protocol":protocol,"cases":cases,"case_count":165,
        "existing_case_count":35,"additional_case_count":130,
        "historical_status_counts":statuses,"historical_retained_targets":retained,
        "new_model_evaluations":0,"new99_percent_passes":Value::Null}))
}

#[cfg(test)]
mod tests;
