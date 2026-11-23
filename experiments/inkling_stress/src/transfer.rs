//! Independently replay a trained checkpoint on Phasor's original length probes.
use crate::{
    data::{Example, Task},
    depth,
    engine::Engine,
    paired::RowScore,
};
use minibatch::Result;
use serde_json::{Map, Value, json};

#[path = "../../phasor_value_transport/src/stress/recall_gap.rs"]
mod recall_gap;

#[derive(Clone, Copy)]
pub enum Probe {
    RecallGap,
    DepthLength,
}

fn scorer_example(row: basis::data::Example) -> Example {
    Example {
        task: row.task,
        tokens: row.tokens,
        targets: row.targets,
        changes: row.changes,
        prompt: row.prompt,
        answer: row.answer,
        limit: row.limit,
        field: row.field,
        row_index: row.row_index,
    }
}

pub fn evaluate(
    engine: &mut Engine,
    task: &Task,
    row: &Value,
    trained: &Value,
    probe: Probe,
    sizes: &[usize],
    passes: usize,
) -> Result<Value> {
    let name = row["name"].as_str().ok_or("missing task name")?;
    let mut record = depth::evaluate(engine, task, trained, &[])?;
    if passes != crate::text::PASSES {
        record["probe_passes"] = json!(passes);
    }
    let mut scores = Map::new();
    let mut paired = Map::new();
    for &size in sizes {
        let (digest, examples) = match probe {
            Probe::RecallGap => {
                let test = task
                    .splits
                    .get("test")
                    .ok_or("missing independent recall test")?;
                let source: Vec<_> = test
                    .iter()
                    .map(|r| basis::wire::example(&r.task, r.tokens.clone(), r.targets.clone()))
                    .collect();
                recall_gap::delay(name, &source, size)?
            }
            Probe::DepthLength => {
                if size == 0 {
                    return Err("depth length must be positive".into());
                }
                let (spec, reference) = basis::length::specification(name)?;
                let anchor = format!("test/length-{reference}");
                let count = task
                    .splits
                    .get(&anchor)
                    .ok_or("missing reference length")?
                    .len();
                let expected = row["data_fingerprints"][&anchor]
                    .as_str()
                    .ok_or("missing reference length fingerprint")?;
                basis::length::verified_rows(spec, reference, count, expected, size)?
            }
        };
        let examples: Vec<_> = examples.into_iter().map(scorer_example).collect();
        engine.evaluation_passes = passes;
        let score = RowScore::measure(engine, &examples)?;
        if passes != crate::text::PASSES {
            engine.evaluation_passes = crate::text::PASSES;
            let baseline = RowScore::measure(engine, &examples)?;
            paired.insert(size.to_string(), baseline.compare(&score)?);
        }
        scores.insert(
            size.to_string(),
            json!({"exact":score.exact,"total":score.total,"split_sha256":digest}),
        );
    }
    record["probe"] = json!(match probe {
        Probe::RecallGap => "recall-gap",
        Probe::DepthLength => "depth-length",
    });
    record["extended"] = json!(scores);
    if !paired.is_empty() {
        record["probe_paired_vs_trained"] = json!(paired);
    }
    Ok(record)
}
