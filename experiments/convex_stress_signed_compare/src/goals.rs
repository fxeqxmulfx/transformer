//! Raw99 and exact99 per task are distinct from the original success flag and
//! from pooled-field accuracy. Never substitute the pooled aggregate for either.
use super::*;
pub(crate) fn metric(m: &Value) -> Result<Value> {
    let raw = raw_accuracy(m).as_f64().ok_or("missing raw goal metric")?;
    // The original field evaluator stores pre-readout exact examples in
    // native_correct, then overwrites sequence_accuracy with pooled readout.
    let (exact, basis) = if let Some(correct) = m.get("native_correct") {
        let correct = correct.as_u64().ok_or("invalid native exact count")?;
        let examples = m["examples"]
            .as_u64()
            .ok_or("missing native example count")?;
        if examples == 0 || correct > examples || m["targets"] != examples {
            return Err("native field goals require original single-target examples".into());
        }
        let exact = correct as f64 / examples as f64;
        if (raw - exact).abs() > 1e-12 {
            return Err("native goal counts disagree".into());
        }
        (exact, "native_correct / examples before field readout")
    } else {
        (
            m["sequence_accuracy"]
                .as_f64()
                .ok_or("missing exact goal metric")?,
            "sequence_accuracy",
        )
    };
    if [raw, exact]
        .iter()
        .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
    {
        return Err("invalid per-task goal metric".into());
    }
    Ok(
        json!({"raw_accuracy":raw,"exact_accuracy":exact,"raw99":raw>=0.99,
        "exact99":exact>=0.99,"both99":raw>=0.99&&exact>=0.99,"exact_basis":basis}),
    )
}
pub(crate) fn splits(scores: &Value) -> Result<Value> {
    let mut out = serde_json::Map::new();
    for (name, m) in scores.as_object().ok_or("missing split goals")? {
        out.insert(name.clone(), metric(m)?);
    }
    if out.is_empty() {
        return Err("empty split goals".into());
    }
    Ok(Value::Object(out))
}
pub(crate) fn summarize(summary: &Value) -> Result<Value> {
    // Reuse the independent original-denominator/native-count consistency check.
    counts::summarize(summary)?;
    let mut arms = vec![];
    for arm in summary["arms"].as_array().ok_or("missing goal policies")? {
        let mut choices = vec![];
        for choice in arm["choices"].as_array().ok_or("missing goal choices")? {
            let primary = choice["primary_split"]
                .as_str()
                .ok_or("missing primary goal split")?;
            let goals = splits(&choice["metrics"])?;
            if goals.get(primary).is_none() {
                return Err("missing primary goal metric".into());
            }
            let held: Vec<_> = goals
                .as_object()
                .unwrap()
                .iter()
                .filter(|(s, _)| s.as_str() != "train")
                .collect();
            if held.is_empty() {
                return Err("missing held-out goal scope".into());
            }
            choices.push(
                json!({"task":choice["task"],"training_mode":choice["training_mode"],
                "reference":choice["reference"],"original_primary_pass":choice["primary_pass"],
                "primary":goals[primary],"splits":goals,
                "all_heldout_raw99":held.iter().all(|(_,v)|v["raw99"]==true),
                "all_heldout_exact99":held.iter().all(|(_,v)|v["exact99"]==true),
                "all_heldout_both99":held.iter().all(|(_,v)|v["both99"]==true)}),
            );
        }
        let mut modes = vec![];
        for mode in ["full_batch", "minibatch", "forward_only"] {
            let rows: Vec<_> = choices
                .iter()
                .filter(|r| r["training_mode"] == mode)
                .collect();
            if rows.len() != 35 {
                return Err("missing per-task goal outcomes".into());
            }
            modes.push(json!({"mode":mode,"cases":35,
                "original_successes":rows.iter().filter(|r|r["original_primary_pass"]==true).count(),
                "primary_raw99":rows.iter().filter(|r|r["primary"]["raw99"]==true).count(),
                "primary_exact99":rows.iter().filter(|r|r["primary"]["exact99"]==true).count(),
                "primary_both99":rows.iter().filter(|r|r["primary"]["both99"]==true).count(),
                "all_heldout_both99":rows.iter().filter(|r|r["all_heldout_both99"]==true).count()}));
        }
        arms.push(json!({"initializer":arm["initializer"],"modes":modes,"choices":choices}));
    }
    Ok(json!(arms))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn field_readout_success_does_not_hide_raw_or_exact_errors() -> Result<()> {
        let m = json!({"targets":100,"examples":100,"native_correct":98,"native_token_accuracy":0.98,
            "correct":100,"token_accuracy":1.,"sequence_accuracy":1.});
        let goal = metric(&m)?;
        assert_eq!(goal["raw99"], false);
        assert_eq!(goal["exact99"], false);
        assert_eq!(goal["exact_accuracy"], 0.98);
        let mut bad = m;
        bad["examples"] = json!(50);
        assert!(metric(&bad).is_err());
        Ok(())
    }
    #[test]
    fn high_aggregate_cannot_hide_a_failed_task_in_either_policy() -> Result<()> {
        let records: Vec<_> = (0..35)
            .map(|i| json!({"task":format!("task{i}"),"status":"complete"}))
            .collect();
        let choices:Vec<_>=(0..35).flat_map(|i|["full_batch","minibatch","forward_only"].map(move|mode|{
            let correct=if i==0 {98}else{100};
            json!({"task":format!("task{i}"),"training_mode":mode,"reference":"fixture",
                "primary_split":"validation","primary_pass":true,"metrics":{"validation":{
                "targets":100,"examples":100,"native_correct":correct,"native_token_accuracy":correct as f64/100.,
                "correct":100,"token_accuracy":1.,"sequence_accuracy":1.}}})
        })).collect();
        let mut summary = json!({"tasks":35,"choices_per_arm":105,"process_failures":0,"records":records,
            "arms":[{"initializer":"minimum_ce","choices":choices},{"initializer":"first_small_ce","choices":choices}]});
        for arm in summarize(&summary)?.as_array().unwrap() {
            for mode in arm["modes"].as_array().unwrap() {
                assert_eq!(mode["original_successes"], 35);
                assert_eq!(mode["primary_raw99"], 34);
                assert_eq!(mode["primary_exact99"], 34);
                assert_eq!(mode["all_heldout_both99"], 34);
            }
        }
        for arm in counts::summarize(&summary)?.as_array().unwrap() {
            assert!(
                arm["modes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|m| m["accuracy"].as_f64().unwrap() > 0.99)
            );
        }
        summary["arms"][0]["choices"].as_array_mut().unwrap().pop();
        assert!(summarize(&summary).is_err());
        Ok(())
    }
}
