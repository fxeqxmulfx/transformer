//! Preserve complete reviewed328, including its earlier refusals. Repeated
//! controls must retain all12 trained states and every measured endpoint.
use crate::*;
use serde_json::Value;
use std::{collections::BTreeSet, path::Path};

pub fn check_report(report: &Value) -> Result<()> {
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let cases = report["cases"].as_array().ok_or("missing earlier cases")?;
    let names: BTreeSet<_> = cases
        .iter()
        .filter_map(|r| r["results"]["case"].as_str())
        .collect();
    let expected = parent::dataset::cases()?;
    if report["complete"] != true
        || cases.len() != 12
        || names.len() != 12
        || expected.iter().any(|c| !names.contains(c.as_str()))
        || report["inventory"]["producer_commit"] != p["previous_attempt_producer"]
    {
        return Err("earlier attempt scope or producer changed".into());
    }
    Ok(())
}
pub fn read(root: &Path) -> Result<Value> {
    let report = runner::artifacts::read(&root.join("previous-attempt.json"))?;
    check_report(&report)?;
    Ok(report)
}
pub fn case(root: &Path, name: &str) -> Result<Value> {
    if generative::CASES.contains(&name) {
        return Ok(Value::Null);
    }
    read(root)?["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["results"]["case"] == name)
        .cloned()
        .ok_or("earlier case missing".into())
}
pub fn check_case(old: &Value, now: &Value, state_hash: &str) -> Result<()> {
    let before = old["results"]["modes"]
        .as_array()
        .ok_or("earlier modes missing")?;
    let after = now["modes"].as_array().ok_or("new modes missing")?;
    if old["training_seal"]["state_sha256"] != state_hash
        || old["results"]["case"] != now["case"]
        || before.len() != 3
        || after.len() != 3
    {
        return Err("control changed a trained state or scope".into());
    }
    for (a, b) in before.iter().zip(after) {
        if a != b {
            return Err("control changed a previously measured endpoint".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn prior_controls_and_states_cannot_change_even_on_refusal() -> Result<()> {
        let modes: Vec<_> = ["Full", "Mini", "Forward"]
            .into_iter()
            .map(|m| json!({"mode":m,"policy":{"mode":m},"error":null,"raw":1.0}))
            .collect();
        let now = json!({"case":"x","modes":modes});
        let mut old = json!({"training_seal":{"state_sha256":"state"},"results":now});
        check_case(&old, &now, "state")?;
        let mut altered = now.clone();
        altered["modes"][0]["raw"] = json!(0.9);
        assert!(check_case(&old, &altered, "state").is_err());
        assert!(check_case(&old, &now, "different").is_err());
        old["results"]["modes"][0]["error"] = json!("128B refused");
        assert!(check_case(&old, &altered, "state").is_err());
        assert_eq!(old["results"]["modes"][0]["error"], "128B refused");
        altered["modes"][1]["mode"] = json!("Full");
        assert!(check_case(&old, &altered, "state").is_err());
        Ok(())
    }
}
