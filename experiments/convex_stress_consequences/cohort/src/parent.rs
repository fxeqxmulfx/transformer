use super::*;
pub(super) fn validate(parent: &Value) -> Result<()> {
    let suite: Value = serde_json::from_str(SUITE)?;
    let rows = parent["choices"]
        .as_array()
        .ok_or("missing complete parent choices")?;
    if rows.len() != 105 || parent["choices_compared"] != 105 || parent["full35_result"] != true {
        return Err("complete105 parent choices required".into());
    }
    for task in suite["tasks"].as_array().ok_or("missing suite tasks")? {
        for mode in modes()? {
            let matching: Vec<_> = rows
                .iter()
                .filter(|r| r["task"] == task["name"] && r["training_mode"] == mode)
                .collect();
            if matching.len() != 1 {
                return Err("parent task/mode missing or duplicated".into());
            }
            let r = matching[0];
            if !matches!(r["status"].as_str(), Some("selected" | "no_model"))
                || (r["status"] == "selected") == r["selected"].is_null()
            {
                return Err("inconsistent parent terminal choice".into());
            }
        }
    }
    Ok(())
}
pub(super) fn record<'a>(parent: &'a Value, name: &str) -> Result<&'a Value> {
    let rows: Vec<_> = parent["records"]
        .as_array()
        .ok_or("missing original parent records")?
        .iter()
        .filter(|r| r["task"] == name)
        .collect();
    if rows.len() != 1 {
        return Err("missing or duplicate original parent record".into());
    }
    Ok(rows[0])
}
pub(super) fn original(parent: &Value) -> Result<()> {
    validate(parent)?;
    let p = protocol()?;
    if parent["source_fingerprint"] != p["parent_source_fingerprint"]
        || parent["producer_commit"] != p["parent_producer_commit"]
        || parent["tasks"] != 23
        || parent["process_failures"] != 0
        || parent["records"].as_array().map(Vec::len) != Some(23)
    {
        return Err("complete original task223 parent freeze differs".into());
    }
    for name in tasks()? {
        let r = record(parent, &name)?;
        if r["status"] != "complete"
            || r["seal"]["replayed"] != true
            || r["result"]["metadata"]["source_fingerprint"] != p["parent_source_fingerprint"]
            || r["result"]["metadata"]["producer_commit"] != p["parent_producer_commit"]
        {
            return Err("original parent lacks a terminal replayed case".into());
        }
    }
    Ok(())
}
pub(super) fn choices(parent: &Value, name: &str, input: &Value) -> Result<Vec<Value>> {
    let meta = &record(parent, name)?["result"]["metadata"]["input"];
    for key in ["task", "vocab", "selection", "data_fingerprints"] {
        if meta[key] != input[key] {
            return Err("parent original data/vocabulary/split differs".into());
        }
    }
    modes()?
        .iter()
        .map(|mode| {
            let rows: Vec<_> = parent["choices"]
                .as_array()
                .ok_or("missing parent choices")?
                .iter()
                .filter(|r| r["task"] == name && r["training_mode"] == *mode)
                .collect();
            if rows.len() != 1 || rows[0]["selection"] != input["selection"] {
                return Err("parent mode or primary split differs".into());
            }
            Ok(rows[0].clone())
        })
        .collect()
}
