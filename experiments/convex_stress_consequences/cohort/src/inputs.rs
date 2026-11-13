use super::*;
pub(super) struct Inputs {
    pub task: data::Task,
    pub input: view::Input,
    pub record: Value,
}
pub(super) fn assemble(task: data::Task, name: &str, data_fingerprints: Value) -> Result<Inputs> {
    convex_stress_relabel::check_layout(&task)?;
    let p = protocol()?;
    let input = view::Input {
        categorical: learner::Request {
            n: task.vocab,
            vocabulary: task.vocab,
            observations: task.splits["train"]
                .iter()
                .map(|e| {
                    [
                        e.tokens[1] as usize + task.vocab * e.tokens[0] as usize,
                        e.targets[1] as usize,
                    ]
                })
                .collect(),
            policy: policy()?,
        },
        minimum_strip_support: p["minimum_strip_support"]
            .as_u64()
            .ok_or("missing strip support")? as usize,
        max_views: p["max_views"].as_u64().ok_or("missing view cap")? as usize,
    };
    let slots = view::catalogue(&input)?;
    let record = json!({"task":name,"vocab":task.vocab,"selection":task.selection,
        "data_fingerprints":data_fingerprints,"training_input_sha256":hash(&serde_json::to_vec(&input)?),
        "views_proposed":slots.len(),"catalogue_sha256":hash(&serde_json::to_vec(&slots)?)});
    Ok(Inputs {
        task,
        input,
        record,
    })
}
pub(super) fn inputs(name: &str) -> Result<Inputs> {
    if !tasks()?.iter().any(|t| t == name) {
        return Err("unplanned consequence task".into());
    }
    let suite: Value = serde_json::from_str(SUITE)?;
    let row = suite["tasks"]
        .as_array()
        .ok_or("missing full suite")?
        .iter()
        .find(|r| r["name"] == name)
        .ok_or("unknown original task")?;
    assemble(data::build(row)?, name, row["data_fingerprints"].clone())
}
pub(super) fn check_input(inv: &Value, name: &str, x: &Inputs) -> Result<()> {
    let rows: Vec<_> = inv["inputs"]
        .as_array()
        .ok_or("missing inventory inputs")?
        .iter()
        .filter(|r| r["task"] == name)
        .collect();
    if rows.len() != 1 || *rows[0] != x.record {
        return Err("original input fingerprint differs".into());
    }
    Ok(())
}
