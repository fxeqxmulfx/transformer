//! Count raw target tokens across the unchanged primary splits. Native counts
//! take priority over pooled-field counts. Larger tasks receive more weight.
//! Ported unchanged from task306 at 65aa7961.
use super::*;

pub(crate) fn summarize(summary: &Value) -> Result<Value> {
    let records = summary["records"]
        .as_array()
        .ok_or("missing original records")?;
    let tasks: BTreeSet<_> = records.iter().map(|r| r["task"].as_str()).collect();
    let arms = summary["arms"].as_array().ok_or("missing policies")?;
    if summary["tasks"] != 35
        || summary["choices_per_arm"] != 105
        || summary["process_failures"] != 0
        || records.len() != 35
        || tasks.len() != 35
        || tasks.contains(&None)
        || records.iter().any(|r| r["status"] != "complete")
        || arms.len() != 2
        || arms
            .iter()
            .map(|r| r["initializer"].as_str())
            .collect::<BTreeSet<_>>()
            != BTreeSet::from([Some("minimum_ce"), Some("first_small_ce")])
    {
        return Err("complete original task and policy scope required".into());
    }
    let mut output = Vec::new();
    for arm in arms {
        let choices = arm["choices"]
            .as_array()
            .ok_or("missing original choices")?;
        let mut seen = BTreeSet::new();
        let mut totals = BTreeMap::<String, (u64, u64, usize)>::new();
        let mut imperfect = Vec::new();
        for choice in choices {
            let task = choice["task"].as_str().ok_or("missing chosen task")?;
            let mode = choice["training_mode"]
                .as_str()
                .ok_or("missing chosen mode")?;
            if !tasks.contains(&Some(task))
                || !["full_batch", "minibatch", "forward_only"].contains(&mode)
                || !seen.insert((task, mode))
            {
                return Err("invalid or duplicate original choice".into());
            }
            let split = choice["primary_split"]
                .as_str()
                .ok_or("missing primary split")?;
            let m = &choice["metrics"][split];
            let targets = m["targets"].as_u64().ok_or("missing raw denominator")?;
            let correct = m
                .get("native_correct")
                .or_else(|| m.get("correct"))
                .and_then(Value::as_u64)
                .ok_or("missing raw correct count")?;
            let accuracy = m
                .get("native_token_accuracy")
                .or_else(|| m.get("token_accuracy"))
                .and_then(Value::as_f64)
                .ok_or("missing raw accuracy")?;
            if targets == 0
                || correct > targets
                || !accuracy.is_finite()
                || (accuracy - correct as f64 / targets as f64).abs() > 1e-12
            {
                return Err("raw counts and accuracy disagree".into());
            }
            let entry = totals.entry(mode.into()).or_default();
            entry.0 = entry
                .0
                .checked_add(targets)
                .ok_or("raw denominator overflow")?;
            entry.1 = entry.1.checked_add(correct).ok_or("raw count overflow")?;
            entry.2 += 1;
            if correct != targets {
                imperfect.push(
                    json!({"task":task,"mode":mode,"targets":targets,"correct":correct,
                "errors":targets-correct,"raw_accuracy":accuracy,"reference":choice["reference"]}),
                );
            }
        }
        if choices.len() != 105 || totals.len() != 3 || totals.values().any(|r| r.2 != 35) {
            return Err("missing original task/mode choices".into());
        }
        let modes: Vec<_> = totals.into_iter().map(|(mode,(targets,correct,cases))|json!({"mode":mode,"cases":cases,
            "targets":targets,"correct":correct,"errors":targets-correct,"accuracy":correct as f64/targets as f64})).collect();
        output.push(json!({"initializer":arm["initializer"],"modes":modes,"imperfect":imperfect}));
    }
    Ok(json!(output))
}
