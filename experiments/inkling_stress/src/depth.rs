//! Measure the same saved weights and examples with different shared-pass budgets.
use crate::{data::Task, engine::Engine, paired::RowScore, text};
use minibatch::Result;
use serde_json::{Map, Value, json};

pub fn evaluate(
    engine: &mut Engine,
    task: &Task,
    trained: &Value,
    budgets: &[usize],
) -> Result<Value> {
    let mut splits = vec![task.selection.as_str()];
    if task.selection != "test" && task.splits.contains_key("test") {
        splits.push("test");
    }
    let mut scores = Map::new();
    let mut baseline = std::collections::BTreeMap::new();
    let mut paired = Map::new();
    for passes in std::iter::once(text::PASSES).chain(budgets.iter().copied()) {
        if scores.contains_key(&passes.to_string()) {
            continue;
        }
        engine.evaluation_passes = passes;
        let mut measured = Map::new();
        let mut compared = Map::new();
        for &name in &splits {
            let score = RowScore::measure(engine, &task.splits[name])?;
            measured.insert(
                name.to_owned(),
                json!({"exact":score.exact,"total":score.total}),
            );
            if passes == text::PASSES {
                baseline.insert(name, score);
            } else {
                compared.insert(name.to_owned(), baseline[name].compare(&score)?);
            }
        }
        let primary = &measured[&task.selection];
        if passes == text::PASSES
            && (primary["exact"] != trained["exact"] || primary["total"] != trained["total"])
        {
            return Err(
                "trained-depth replay disagrees with the stored primary observation".into(),
            );
        }
        scores.insert(passes.to_string(), json!(measured));
        if !compared.is_empty() {
            paired.insert(passes.to_string(), json!(compared));
        }
    }
    let mut record = json!({"model":text::MODEL, "case":trained["case"], "seed":trained["seed"],
        "step":trained["step"], "trained_passes":text::PASSES, "primary":task.selection,
        "checkpoint_sha256":trained["checkpoint_sha256"],"binding":trained["binding"],"scores":scores});
    if !paired.is_empty() {
        record["paired_vs_trained"] = json!(paired);
    }
    Ok(record)
}
