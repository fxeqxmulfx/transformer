//! Report standalone endpoints and selected regressions after selection.
//! These diagnostics never feed the ordinary raw-CE selector.
use super::*;

pub(crate) fn changes(parent: &Value, arms: &[Value]) -> Result<Value> {
    let mut reports = vec![];
    for arm in arms {
        let prior = parent["arms"]
            .as_array()
            .ok_or("missing original arms")?
            .iter()
            .find(|a| a["initializer"] == arm["initializer"])
            .ok_or("missing initializer")?;
        let mut replacements = vec![];
        for row in arm["choices"].as_array().ok_or("missing choices")? {
            let old = prior["choices"]
                .as_array()
                .ok_or("missing prior choices")?
                .iter()
                .find(|r| r["task"] == row["task"] && r["training_mode"] == row["training_mode"])
                .ok_or("missing prior cell")?;
            if old["reference"] == row["reference"] {
                continue;
            }
            let mut deltas = vec![];
            if let (Some(before), Some(after)) =
                (old["metrics"].as_object(), row["metrics"].as_object())
            {
                for (split, metrics) in before {
                    for key in ["loss", "token_accuracy", "sequence_accuracy"] {
                        if let (Some(a), Some(b)) = (
                            metrics[key].as_f64(),
                            after.get(split).and_then(|m| m[key].as_f64()),
                        ) {
                            if !a.is_finite() || !b.is_finite() {
                                return Err("nonfinite comparison metric".into());
                            }
                            deltas.push(
                                json!({"split":split,"metric":key,"before":a,"after":b,"delta":b-a,
                                "regression":if key=="loss"{b>a+1e-12}else{b<a-1e-12}}),
                            );
                        }
                    }
                }
            }
            replacements.push(
                json!({"task":row["task"],"training_mode":row["training_mode"],
                "prior_reference":old["reference"],"reference":row["reference"],
                "prior_primary_pass":old["primary_pass"],"primary_pass":row["primary_pass"],
                "primary_regression":old["primary_pass"]==true && row["primary_pass"]!=true,
                "new_primary_pass":old["primary_pass"]!=true && row["primary_pass"]==true,
                "metric_deltas":deltas}),
            );
        }
        reports.push(json!({"initializer":arm["initializer"],"replacements":replacements.len(),
            "primary_regressions":replacements.iter().filter(|r|r["primary_regression"]==true).count(),
            "new_primary_passes":replacements.iter().filter(|r|r["new_primary_pass"]==true).count(),
            "changed_choices":replacements}));
    }
    Ok(json!({"used_for_selection":false,"tolerance":1e-12,"arms":reports}))
}
