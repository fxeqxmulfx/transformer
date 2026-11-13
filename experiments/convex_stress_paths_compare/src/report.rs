//! Read the pinned replay metrics only after reference selection is fixed.
use crate::*;
pub(crate) fn passes(metrics: &Value, primary: &str) -> Result<(bool, bool)> {
    let values = metrics.as_object().ok_or("missing original metrics")?;
    number(&metrics["train"], "loss")?;
    let heldout: Vec<_> = values
        .iter()
        .filter(|(name, _)| name.as_str() != "train")
        .map(|(_, v)| number(v, "sequence_accuracy"))
        .collect::<Result<_>>()?;
    check(
        !heldout.is_empty() && heldout.iter().all(|a| *a <= 1.),
        "missing or invalid held-out accuracies",
    )?;
    Ok((
        number(&metrics[primary], "sequence_accuracy")? >= 0.99,
        heldout.iter().all(|a| *a >= 0.99),
    ))
}
pub(crate) fn build(suite: &Value, loaded: &Loaded, selection: &Value) -> Result<Value> {
    let mut arms = vec![];
    for arm in rows(selection, "arms")? {
        let mut choices = vec![];
        for selected in rows(arm, "choices")? {
            let name = text(selected, "task")?;
            let mode = text(selected, "mode")?;
            let original = loaded
                .prior
                .iter()
                .find(|r| r.task == name && r.mode == mode)
                .ok_or("missing original reference")?;
            let row = if let Some(reference) = selected["selected"].as_str() {
                let offer = loaded
                    .prior
                    .iter()
                    .chain(&loaded.retained)
                    .find(|r| r.reference == reference)
                    .ok_or("unknown selected reference")?;
                let (primary, all) = passes(&offer.metrics, &offer.primary)?;
                json!({"task":name,"training_mode":mode,"status":"selected","reference":reference,
                    "prior_reference":original.reference,"train_ce":offer.train_ce,"primary_split":offer.primary,
                    "metrics":offer.metrics,"saved_endpoint":offer.saved,"primary_pass":primary,"all_splits_pass":all})
            } else {
                check(
                    original.train_ce.is_none(),
                    "no-model selection discarded an existing endpoint",
                )?;
                json!({"task":name,"training_mode":mode,"status":"no_model","reference":null,
                    "prior_reference":original.reference,"saved_endpoint":original.saved,"primary_pass":false,"all_splits_pass":false})
            };
            choices.push(row);
        }
        let by_mode: Vec<_> = MODES.iter().map(|mode| {
            let rows: Vec<_> = choices.iter().filter(|r| r["training_mode"] == *mode).collect();
            json!({"training_mode":mode,"cases":rows.len(),"primary_passes":rows.iter().filter(|r| r["primary_pass"]==true).count(),
                "all_split_passes":rows.iter().filter(|r| r["all_splits_pass"]==true).count(),"no_model":rows.iter().filter(|r| r["status"]=="no_model").count()})
        }).collect();
        arms.push(json!({"initializer":arm["initializer"],"full35_pass":choices.iter().all(|r|r["all_splits_pass"]==true),"choices":choices,"by_mode":by_mode}));
    }
    check(
        loaded.transformers.len() == rows(suite, "tasks")?.len()
            && loaded.depth_records.len() == 336,
        "comparison dropped required evidence",
    )?;
    Ok(
        json!({"source_fingerprint":fingerprint(),"protocol":protocol()?,"arms":arms,
        "choose_between_initializers":false,"complete_parent_snapshots":loaded.snapshots,
        "depth_outcomes":loaded.depth_records,"transformer_endpoints":loaded.transformers,
        "constructor_accounting":loaded.accounting,"all_parent_costs_charged_once":true,
        "new_gradient_updates":0,"new_prediction_evaluations":0,"joint_optimizer_run":false,
        "matched_compute_comparison":false,"unconditional_identification":false,
        "comparison_scope":"reviewed saved endpoints; exact original success rules; both initializer arms retained"}),
    )
}
