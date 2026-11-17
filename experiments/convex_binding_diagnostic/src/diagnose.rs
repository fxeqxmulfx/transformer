//! Original full-class scoring is independently retained beside role-aware counts.
use crate::{
    Result,
    counts::{Context, Counts, Observation},
};
use baseline::data::Example;
use bilinear::{experiment as original, train::Record};
use growth::model::Model;
use serde_json::{Value, json};
pub fn split(model: &Model, examples: &[Example]) -> Result<Value> {
    let mut raw = growth::score::evaluate(model, examples)?;
    raw["raw_and_exact_99"] = json!(
        raw["metrics"]["token_accuracy"].as_f64().ok_or("raw")? >= 0.99
            && raw["metrics"]["sequence_accuracy"]
                .as_f64()
                .ok_or("exact")?
                >= 0.99
    );
    let mut all = Counts::default();
    let mut selected: [Counts; 2] = std::array::from_fn(|_| Counts::default());
    let mut positions: [Counts; 7] = std::array::from_fn(|_| Counts::default());
    let mut digest = Vec::new();
    for example in examples {
        if example.task != "mqar"
            || example.tokens.len() != 12
            || example.targets.len() != 12
            || example.targets.iter().filter(|&&t| t >= 0).count() != 1
            || !example.prompt.is_empty()
            || example.field.is_some()
        {
            return Err("original one-answer MQAR example changed".into());
        }
        for (position, &target) in example.targets.iter().enumerate() {
            if target < 0 {
                continue;
            }
            let record = Record {
                prefix: &example.tokens[..=position],
                target: target as usize,
            };
            let context = Context::new(&record)?;
            let logits = model.logits(record.prefix)?;
            let observation = Observation::new(&context, &logits)?;
            all.observe(&observation);
            selected[context.selected].observe(&observation);
            positions[context.query_position - 5].observe(&observation);
            digest.extend_from_slice(&(record.prefix.len() as u64).to_le_bytes());
            for t in record.prefix {
                digest.extend_from_slice(&t.to_le_bytes());
            }
            digest.extend_from_slice(&(logits.len() as u64).to_le_bytes());
            for x in &logits {
                digest.extend_from_slice(&x.to_bits().to_le_bytes());
            }
        }
    }
    let metrics = &raw["metrics"];
    if all.total != examples.len()
        || all.categories.iter().sum::<usize>() != all.total
        || metrics["targets"] != all.total
        || metrics["correct"] != all.categories[0]
        || metrics["exact"] != all.categories[0]
        || (metrics["loss"].as_f64().ok_or("loss")? - all.full_ce / all.total as f64).abs() > 2e-12
        || raw["raw_logits_sha256"] != original::hash(&digest)
        || selected.iter().map(|x| x.total).sum::<usize>() != all.total
        || positions.iter().map(|x| x.total).sum::<usize>() != all.total
    {
        return Err("binding counts disagree with original full raw metrics".into());
    }
    Ok(
        json!({"original":raw,"overall":all.report(),"by_selected_pair":selected.iter().enumerate().map(|(i,c)|json!({"pair":i,"counts":c.report()})).collect::<Vec<_>>(),"by_query_position":positions.iter().enumerate().map(|(i,c)|json!({"position":i+5,"counts":c.report()})).collect::<Vec<_>>()}),
    )
}
