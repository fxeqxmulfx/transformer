use super::*;
pub(super) fn validate(parent: &Value) -> Result<()> {
    let suite: Value = serde_json::from_str(SUITE)?;
    let rows = parent["choices"]
        .as_array()
        .ok_or("missing complete parent choices")?;
    if rows.len() != 105 || parent["choices_compared"] != 105 || parent["full35_result"] != true {
        return Err("full105 complete parent required".into());
    }
    for task in suite["tasks"].as_array().ok_or("missing full suite")? {
        for mode in modes()? {
            let matching: Vec<_> = rows
                .iter()
                .filter(|r| r["task"] == task["name"] && r["training_mode"] == mode)
                .collect();
            if matching.len() != 1 {
                return Err("missing/duplicate complete parent task/mode".into());
            }
            let r = matching[0];
            if !matches!(r["status"].as_str(), Some("selected" | "no_model"))
                || (r["status"] == "selected") == r["selected"].is_null()
            {
                return Err("nonterminal or inconsistent complete parent choice".into());
            }
        }
    }
    Ok(())
}
pub(super) fn original(parent: &Value) -> Result<()> {
    validate(parent)?;
    let p = protocol()?;
    if parent["source_fingerprint"] != p["parent_source_fingerprint"]
        || parent["producer_commit"] != p["parent_producer_commit"]
        || parent["tasks"] != 23
        || parent["process_failures"] != 0
        || parent["records"]
            .as_array()
            .ok_or("missing original records")?
            .len()
            != 23
    {
        return Err("complete original modular parent freeze differs".into());
    }
    for name in tasks()? {
        let row = record(parent, &name)?;
        if row["status"] != "complete"
            || row["result"]["metadata"]["source_fingerprint"] != p["parent_source_fingerprint"]
            || row["result"]["metadata"]["producer_commit"] != p["parent_producer_commit"]
            || row["seal"]["replayed"] != true
        {
            return Err("original modular parent is not completely replayed".into());
        }
    }
    Ok(())
}
pub(super) fn record<'a>(parent: &'a Value, name: &str) -> Result<&'a Value> {
    let rows: Vec<_> = parent["records"]
        .as_array()
        .ok_or("missing modular parent records")?
        .iter()
        .filter(|r| r["task"] == name)
        .collect();
    if rows.len() != 1 {
        return Err("missing/duplicate modular parent record".into());
    }
    Ok(rows[0])
}
pub(super) fn choices(parent: &Value, name: &str, record: &Value) -> Result<Vec<Value>> {
    let inventory = &parent["complete_parent"]["parents"]["complete_general_comparison"]["parents"]
        ["hull_inventory"]["tasks"];
    let meta = inventory
        .as_array()
        .ok_or("missing original parent inventory")?
        .iter()
        .find(|r| r["task"] == name)
        .ok_or("missing original task")?;
    if meta["vocab"] != record["vocab"]
        || meta["scale"] != 1.0
        || meta["data_fingerprints"] != record["data_fingerprints"]
    {
        return Err("parent/original inputs, vocabulary or CE scale differ".into());
    }
    modes()?
        .iter()
        .map(|mode| {
            let rows: Vec<_> = parent["choices"]
                .as_array()
                .ok_or("missing complete choices")?
                .iter()
                .filter(|r| r["task"] == name && r["training_mode"] == *mode)
                .collect();
            if rows.len() != 1 || rows[0]["selection"] != record["selection"] {
                return Err("parent mode or primary split differs".into());
            }
            Ok(rows[0].clone())
        })
        .collect()
}
pub(super) fn proposal(
    raw: &[u8],
    parent: &Value,
    name: &str,
    x: &Inputs,
) -> Result<search::Report> {
    let r = record(parent, name)?;
    if r["result"]["selection"]["report_sha256"] != hash(raw)
        || r["result"]["selection"]["input_sha256"] != x.record["training_input_sha256"]
    {
        return Err("immutable modular proposal/input binding changed".into());
    }
    let report: search::Report = serde_json::from_slice(raw)?;
    if report.input != x.input {
        return Err("original proposal observations or policy differ".into());
    }
    Ok(report)
}
