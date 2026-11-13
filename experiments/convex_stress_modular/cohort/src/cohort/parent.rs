use super::*;
pub(super) fn validate(parent: &Value) -> Result<()> {
    let suite: Value = serde_json::from_str(SUITE)?;
    let modes = modes()?;
    let rows = parent["choices"]
        .as_array()
        .ok_or("missing complete parent choices")?;
    if rows.len() != 105 || parent["choices_compared"] != 105 || parent["full35_result"] != true {
        return Err("full105 complete parent required".into());
    }
    for task in suite["tasks"].as_array().ok_or("missing full suite")? {
        for mode in &modes {
            let matches: Vec<_> = rows
                .iter()
                .filter(|r| r["task"] == task["name"] && r["training_mode"] == *mode)
                .collect();
            if matches.len() != 1 {
                return Err("missing/duplicate complete parent task/mode".into());
            }
            let r = matches[0];
            if !matches!(r["status"].as_str(), Some("selected" | "no_model"))
                || (r["status"] == "selected") == r["selected"].is_null()
            {
                return Err("nonterminal or inconsistent complete parent choice".into());
            }
        }
    }
    Ok(())
}
pub(super) fn choices(parent: &Value, name: &str, record: &Value) -> Result<Vec<Value>> {
    let rows = parent["choices"]
        .as_array()
        .ok_or("missing parent choices")?;
    let inventory =
        &parent["parents"]["complete_general_comparison"]["parents"]["hull_inventory"]["tasks"];
    let meta = inventory
        .as_array()
        .ok_or("missing original parent inventory")?
        .iter()
        .find(|r| r["task"] == name)
        .ok_or("missing original parent task")?;
    if meta["vocab"] != record["vocab"]
        || meta["scale"] != 1.0
        || meta["data_fingerprints"] != record["data_fingerprints"]
    {
        return Err("parent/new original inputs, vocabulary or CE scale differ".into());
    }
    modes()?
        .into_iter()
        .map(|mode| {
            let matches: Vec<_> = rows
                .iter()
                .filter(|r| r["task"] == name && r["training_mode"] == mode)
                .collect();
            if matches.len() != 1 || matches[0]["selection"] != record["selection"] {
                return Err("parent mode or primary split differs".into());
            }
            Ok(matches[0].clone())
        })
        .collect()
}
pub(super) fn select(report: &search::Report, parents: &[Value]) -> Result<Value> {
    let mut references = vec![];
    for old in parents {
        let mut candidates = vec![];
        if !old["selected"].is_null() {
            candidates.push(json!({"id":"parent","train_ce":old["selected"]["train_ce"],
                "model_sha256":old["selected"]["model_sha256"],"source":"complete_parent"}));
        }
        for (index, c) in report.candidates.iter().enumerate() {
            candidates.push(
                json!({"id":format!("modular-{index}"),"train_ce":c.audit["train_ce"],
                "model_sha256":hash(&serde_json::to_vec(&c.model)?),"source":"modular"}),
            );
        }
        let (mut selected, mut best) = (Value::Null, f64::INFINITY);
        for c in &candidates {
            let ce = c["train_ce"]
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or("missing finite observed CE")?;
            if ce + 1e-12 < best {
                selected = c["id"].clone();
                best = ce;
            }
        }
        references.push(
            json!({"mode":old["training_mode"],"selected":selected,"candidates":candidates,
            "standalone_constructor_reference":true,"gradient_training_run":false}),
        );
    }
    Ok(Value::Array(references))
}
