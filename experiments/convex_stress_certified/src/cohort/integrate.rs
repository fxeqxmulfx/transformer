//! CE-only selection against the complete audited parent and its predecessors.
use super::*;
use std::collections::BTreeSet;

pub(super) fn combine(own: &Value, parent: &Value, tolerance: f64) -> Result<Value> {
    let candidates = own["candidates"]
        .as_array()
        .ok_or("missing certified candidates")?;
    let (mut choices, mut seen) = (vec![], BTreeSet::new());
    for prior in parent["choices"]
        .as_array()
        .ok_or("missing paired choices")?
    {
        let task = prior["task"].as_str().ok_or("missing task")?;
        let mode = prior["training_mode"]
            .as_str()
            .ok_or("missing training mode")?;
        if !seen.insert((task, mode)) {
            return Err("duplicate parent choice".into());
        }
        let matching: Vec<_> = candidates
            .iter()
            .filter(|c| c["task"] == task && c["training_mode"] == mode)
            .collect();
        if matching.len() > 1 {
            return Err("duplicate certified candidate".into());
        }
        let mut comparison: Vec<_> = matching.into_iter().cloned().collect();
        comparison.push(prior["selected"].clone());
        let vocab = count(&comparison[0], "vocab")?;
        if comparison.iter().any(|c| c["vocab"] != vocab) {
            return Err("vocabulary mismatch".into());
        }
        let losses = comparison
            .iter()
            .map(|c| number(c, "train_ce"))
            .collect::<Result<Vec<_>>>()?;
        let winner = convex_stress_relabel::choose_by_train(&losses, ideal_loss(vocab), tolerance)?;
        choices.push(json!({"task":task,"training_mode":mode,"selected":comparison[winner],
            "certified_selected":comparison.len()==2&&winner==0,"candidates":comparison,"baseline":prior}));
    }
    if candidates.iter().any(|c| {
        !seen.contains(&(
            c["task"].as_str().unwrap_or(""),
            c["training_mode"].as_str().unwrap_or(""),
        ))
    }) {
        return Err("certified candidate has no parent choice".into());
    }
    Ok(
        json!({"choices_compared":choices.len(),"certified_selected_choices":choices.iter().filter(|c|c["certified_selected"]==true).count(),
        "primary_pass_choices":choices.iter().filter(|c|c["selected"]["primary_pass"]==true).count(),
        "all_split_pass_choices":choices.iter().filter(|c|c["selected"]["all_splits_pass"]==true).count(),
        "choices":choices,"refused_or_completed_records":own["records"],
        "full35_result":false,"fresh_transformer_comparison":false}),
    )
}
pub fn integrate(root: &Path, paired: &Path, decoded: &Path) -> Result<Value> {
    let p = protocol()?;
    let own = summarize(root)?;
    if own != read(&root.join("summary.json"))?
        || p["parent_fingerprint"] != convex_stress_paired::cohort::fingerprint()
    {
        return Err("changed certified summary or parent source".into());
    }
    // This recursively checks the paired summary and complete decoded parent,
    // which retains the earlier candidates. Never substitute a partial parent.
    let parent = convex_stress_paired::cohort::integrate(paired, decoded)?;
    if parent["choices"].as_array().map(Vec::len) != Some(cases()?.len()) {
        return Err("complete parent comparison required".into());
    }
    let mut result = combine(&own, &parent, number(&p, "ideal_ce_tolerance")?)?;
    result["source_fingerprint"] = fingerprint().into();
    result["protocol"] = p;
    result["certified_summary_sha256"] =
        hash(&bytes(&root.join("summary.json"), 64 * 1024 * 1024)?).into();
    result["parent"] = json!({"paired_root":fs::canonicalize(paired)?,"decoded_root":fs::canonicalize(decoded)?,
        "paired_summary_sha256":hash(&bytes(&paired.join("summary.json"),64*1024*1024)?),"comparison":parent});
    Ok(result)
}
