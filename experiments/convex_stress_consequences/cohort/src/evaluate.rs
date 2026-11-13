use super::*;
pub(super) fn evaluate(
    bank: &Bank,
    x: &Inputs,
    parents: &[Value],
    references: &Value,
) -> Result<Value> {
    let mut candidates = vec![];
    for (index, c) in bank.candidates.iter().enumerate() {
        let mut all_metrics = serde_json::Map::new();
        for (name, examples) in &x.task.splits {
            all_metrics.insert(name.clone(), metrics::evaluate(&c.value.model, examples)?);
        }
        let ce = c.value.raw_audit["train_ce"]
            .as_f64()
            .ok_or("missing actual original CE")?;
        if (all_metrics["train"]["loss"]
            .as_f64()
            .ok_or("missing train metric")?
            - ce)
            .abs()
            > 1e-10
        {
            return Err("raw CE and original metric disagree".into());
        }
        let primary = all_metrics[&x.task.selection]["sequence_accuracy"]
            .as_f64()
            .ok_or("missing primary accuracy")?;
        let heldout: Vec<_> = all_metrics
            .iter()
            .filter(|(name, _)| name.as_str() != "train")
            .map(|(_, m)| {
                m["sequence_accuracy"]
                    .as_f64()
                    .ok_or("missing held-out accuracy")
            })
            .collect::<std::result::Result<_, _>>()?;
        if heldout.is_empty() {
            return Err("no original held-out splits".into());
        }
        candidates.push(json!({"id":format!("consequence-{index}"),"family":"decoded_consequences","vocab":x.task.vocab,
            "train_ce":ce,"model_sha256":c.model_sha256,"output_sha256":c.output_sha256,
            "view":c.view,"candidate":c.candidate,"duplicate_of":c.duplicate_of,
            "audit":c.value.raw_audit,"metrics":all_metrics,"primary_pass":primary>=0.99,
            "all_splits_pass":heldout.iter().all(|&a| a>=0.99),
            "certified_within_fixed_view":c.value.scope.full_table_certified_within_family,"scope":c.value.scope,
            "constructor_reference":true,"gradient_training_run":false}));
    }
    let mut choices = vec![];
    for old in parents {
        let r = references
            .as_array()
            .ok_or("missing sealed references")?
            .iter()
            .find(|r| r["mode"] == old["training_mode"])
            .ok_or("missing mode reference")?;
        let selected = if r["selected"] == "parent" {
            old["selected"].clone()
        } else if r["selected"].is_null() {
            Value::Null
        } else {
            candidates
                .iter()
                .find(|c| c["id"] == r["selected"])
                .ok_or("selected model absent")?
                .clone()
        };
        choices.push(json!({"task":x.record["task"],"training_mode":old["training_mode"],"tier":old["tier"],
            "selection":x.task.selection,"status":if selected.is_null(){"no_model"}else{"selected"},
            "primary_pass":selected["primary_pass"]==true,"all_splits_pass":selected["all_splits_pass"]==true,
            "selected":selected,"baseline":old,"sealed_reference":r}));
    }
    Ok(json!({"candidates":candidates,"choices":choices}))
}
