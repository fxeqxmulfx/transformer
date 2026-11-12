//! Training-only comparison with the selection of the decoded-phase cohort,
//! whose summary retains every earlier candidate. Nothing here trains or reads
//! a held-out label to choose.
use super::*;
use std::collections::BTreeSet;

/// For each earlier choice, order [paired candidate of the same task and mode,
/// earlier selection]: the first within the tolerance of the CE lower bound,
/// else the lower training CE. The earlier choice is retained whole.
pub(super) fn combine(own: &Value, parent: &Value, tolerance: f64) -> Result<Value> {
    let candidates = own["candidates"]
        .as_array()
        .ok_or("missing paired candidates")?;
    let mut choices = vec![];
    let mut seen = BTreeSet::new();
    for prior in parent["choices"]
        .as_array()
        .ok_or("missing parent choices")?
    {
        let task = prior["task"].as_str().ok_or("missing parent task")?;
        let mode = prior["prior_training_mode"]
            .as_str()
            .ok_or("missing parent mode")?;
        if !seen.insert((task, mode)) {
            return Err("repeated parent choice".into());
        }
        let mut matching = candidates
            .iter()
            .filter(|c| c["task"] == task && c["training_mode"] == mode);
        let (Some(paired), None) = (matching.next(), matching.next()) else {
            return Err("parent choice without exactly one paired candidate".into());
        };
        let pair = [paired.clone(), prior["selected"].clone()];
        let vocab = pair[0]["vocab"].as_u64().ok_or("missing vocabulary")? as usize;
        if pair[1]["vocab"] != vocab {
            return Err("candidate vocabularies differ".into());
        }
        let losses = pair
            .iter()
            .map(|x| number(x, "train_ce"))
            .collect::<Result<Vec<_>>>()?;
        let winner = convex_stress_relabel::choose_by_train(&losses, ideal_loss(vocab), tolerance)?;
        choices.push(
            json!({"task":task,"training_mode":mode,"selected":pair[winner],
            "paired_selected":winner==0,"candidates":pair,"baseline":prior}),
        );
    }
    if choices.len() != candidates.len() {
        return Err("paired comparison scope incomplete".into());
    }
    let tally = |key: &str| {
        choices
            .iter()
            .filter(|c| c["selected"][key] == true)
            .count()
    };
    let earlier = |key: &str| {
        choices
            .iter()
            .filter(|c| c["baseline"]["selected"][key] == true)
            .count()
    };
    Ok(json!({"choices_compared":choices.len(),
        "paired_selected_choices":choices.iter().filter(|c| c["paired_selected"]==true).count(),
        "primary_pass_choices":tally("primary_pass"),
        "all_split_pass_choices":tally("all_splits_pass"),
        "earlier_primary_pass_choices":earlier("primary_pass"),
        "earlier_all_split_pass_choices":earlier("all_splits_pass"),
        "choices":choices,"full35_result":false,"fresh_transformer_comparison":false}))
}
pub fn integrate(root: &Path, parent: &Path) -> Result<Value> {
    let p = protocol()?;
    let own = summarize(root)?;
    if own != read(&root.join("summary.json"))? {
        return Err("paired summary changed since it was written".into());
    }
    if convex_stress_decode::cohort::fingerprint() != p["parent_fingerprint"] {
        return Err("decoded parent fingerprint changed".into());
    }
    let saved_bytes = bytes(&parent.join("summary.json"), 64 * 1024 * 1024)?;
    let saved: Value = serde_json::from_slice(&saved_bytes)?;
    if saved != convex_stress_decode::cohort::summarize(parent)?
        || saved["source_fingerprint"] != p["parent_fingerprint"]
        || saved["cells"] != 8
        || saved["choices"].as_array().map(Vec::len) != Some(24)
    {
        return Err("complete unchanged decoded-phase parent required".into());
    }
    let mut result = combine(&own, &saved, number(&p, "ideal_ce_tolerance")?)?;
    result["source_fingerprint"] = fingerprint().into();
    result["protocol"] = p;
    result["paired_summary_sha256"] =
        hash(&bytes(&root.join("summary.json"), 64 * 1024 * 1024)?).into();
    result["parent"] = json!({"root":fs::canonicalize(parent)?,
        "summary_sha256":hash(&saved_bytes),"source_fingerprint":saved["source_fingerprint"]});
    Ok(result)
}
