use super::*;
pub(super) fn evaluate(
    proposal: &search::Report,
    report: &adapter::Report,
    x: &Inputs,
    parents: &[Value],
    references: &Value,
) -> Result<Value> {
    let mut candidates = vec![];
    for (index, c) in proposal.candidates.iter().enumerate() {
        let mut metrics = serde_json::Map::new();
        for (name, examples) in &x.task.splits {
            metrics.insert(name.clone(), metrics::evaluate(&c.model, examples)?);
        }
        let ce = c.audit["train_ce"]
            .as_f64()
            .ok_or("missing original raw CE")?;
        if (metrics["train"]["loss"]
            .as_f64()
            .ok_or("missing train CE")?
            - ce)
            .abs()
            > 1e-10
        {
            return Err("ordinary original metric disagrees with raw CE".into());
        }
        let primary = metrics[&x.task.selection]["sequence_accuracy"]
            .as_f64()
            .ok_or("missing primary accuracy")?;
        let heldout: Vec<_> = metrics
            .iter()
            .filter(|(name, _)| name.as_str() != "train")
            .map(|(_, m)| {
                m["sequence_accuracy"]
                    .as_f64()
                    .ok_or("missing held-out accuracy")
            })
            .collect::<std::result::Result<_, _>>()?;
        if heldout.is_empty() {
            return Err("no original held-out split".into());
        }
        let e = &report.entries[index];
        candidates.push(json!({"id":format!("identified-{index}"),"family":"identified_modular","vocab":x.task.vocab,
            "train_ce":ce,"model_sha256":hash(&serde_json::to_vec(&c.model)?),"audit":c.audit,"metrics":metrics,
            "primary_pass":primary>=0.99,"all_splits_pass":heldout.iter().all(|&a|a>=0.99),
            "certified_within_fixed_view":e.certificate.certified,"certificate_reason":e.certificate.reason,"scope":e.scope,
            "reused_immutable_modular_model":true,"constructor_reference":true,"gradient_training_run":false}));
    }
    let mut choices = vec![];
    for old in parents {
        let r = references
            .as_array()
            .ok_or("missing sealed references")?
            .iter()
            .find(|r| r["mode"] == old["training_mode"])
            .ok_or("missing sealed mode reference")?;
        let selected = if r["selected"] == "parent" {
            old["selected"].clone()
        } else if r["selected"].is_null() {
            Value::Null
        } else {
            candidates
                .iter()
                .find(|c| c["id"] == r["selected"])
                .ok_or("selected candidate absent")?
                .clone()
        };
        choices.push(json!({"task":x.record["task"],"training_mode":old["training_mode"],"tier":old["tier"],"selection":x.task.selection,
            "status":if selected.is_null(){"no_model"}else{"selected"},"primary_pass":selected["primary_pass"]==true,
            "all_splits_pass":selected["all_splits_pass"]==true,"selected":selected,"baseline":old,
            "identified_candidates":candidates,"sealed_reference":r}));
    }
    Ok(json!({"candidates":candidates,"choices":choices}))
}
