//! Original native data and declared metric groups, ported from49105730.
//! The grouping is evaluation/eligibility metadata, never a model feature.
use super::*;
pub(crate) struct Inputs {
    pub task: data::Task,
    pub input: Option<discovery::Input>,
    pub policies: Option<[training::Policy; 3]>,
    pub readout: Option<Vec<usize>>,
    pub record: Value,
}
pub(crate) fn assemble(
    task: data::Task,
    name: &str,
    pair: bool,
    batch: usize,
    fingerprints: Value,
) -> Result<Inputs> {
    super::name(name)?;
    let (input, policies, readout) = if pair {
        convex_stress_relabel::check_layout(&task)?;
        let train = task
            .splits
            .get("train")
            .filter(|v| !v.is_empty())
            .ok_or("missing original training pairs")?;
        let input = discovery::Input {
            n: task.vocab,
            vocabulary: task.vocab,
            policy: discovery::Policy::default(),
            observations: train
                .iter()
                .map(|e| {
                    [
                        e.tokens[1] as usize + task.vocab * e.tokens[0] as usize,
                        e.targets[1] as usize,
                    ]
                })
                .collect(),
        };
        let policies = training::policies(input.observations.len(), batch)?;
        let field = train[0].field.is_some();
        if train.iter().any(|e| e.field.is_some() != field) {
            return Err("mixed original metric readout".into());
        }
        let q = if field {
            (0..task.vocab)
                .map(|c| (c as i64 - task.vocab as i64 / 2).rem_euclid(97) as usize)
                .collect()
        } else {
            (0..task.vocab).collect()
        };
        (Some(input), Some(policies), Some(q))
    } else {
        (None, None, None)
    };
    let record = json!({"task":name,"primary_split":task.selection,"vocabulary":task.vocab,
        "kernel_planned":pair,"original_batch":batch,"data_fingerprints":fingerprints,
        "input_sha256":input.as_ref().map(|x|serde_json::to_vec(x).map(|b|hash(&b))).transpose()?,"policies":policies,
        "readout":readout,"readout_changes_features_or_ce":false});
    Ok(Inputs {
        task,
        input,
        policies,
        readout,
        record,
    })
}
pub(crate) fn build(suite: &Value, name: &str, planned: &[String]) -> Result<Inputs> {
    let rows: Vec<_> = suite["tasks"]
        .as_array()
        .ok_or("missing original tasks")?
        .iter()
        .filter(|r| r["name"] == name)
        .collect();
    if rows.len() != 1 {
        return Err("unknown or repeated original task".into());
    }
    assemble(
        data::build(rows[0])?,
        name,
        planned.iter().any(|t| t == name),
        rows[0]["case"]["batch"]
            .as_u64()
            .ok_or("missing original batch size")? as usize,
        rows[0]["data_fingerprints"].clone(),
    )
}
pub(crate) fn check(inv: &Value, task: &str, x: &Inputs) -> Result<()> {
    let records: Vec<_> = inv["inputs"]
        .as_array()
        .ok_or("missing input inventory")?
        .iter()
        .filter(|r| r["task"] == task)
        .collect();
    if records.len() != 1 || *records[0] != x.record {
        return Err("original data, policy or declared metric changed".into());
    }
    Ok(())
}
