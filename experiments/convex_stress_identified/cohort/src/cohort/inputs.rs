use super::*;
pub(super) struct Inputs {
    pub task: data::Task,
    pub input: search::Input,
    pub record: Value,
}
pub(super) fn assemble(task: data::Task, name: &str, data_fingerprints: Value) -> Result<Inputs> {
    convex_stress_relabel::check_layout(&task)?;
    let p = protocol()?;
    let input = search::Input {
        categorical: coding::Input {
            n: task.vocab,
            vocab: task.vocab,
            observations: task.splits["train"]
                .iter()
                .map(|e| {
                    [
                        e.tokens[1] as usize + task.vocab * e.tokens[0] as usize,
                        e.targets[1] as usize,
                    ]
                })
                .collect(),
            policy: serde_json::from_value(p["coding"].clone())?,
        },
        minimum_strip_support: p["minimum_strip_support"]
            .as_u64()
            .ok_or("missing strip support")? as usize,
        max_views: p["max_views"].as_u64().ok_or("missing strip cap")? as usize,
    };
    let bound = input.validate()? + 8 * 1024 * 1024 * 1024;
    if bound > 56 * 1024 * 1024 * 1024 {
        return Err("original proposal/JSON forecast exceeds56GiB".into());
    }
    let record = json!({"task":name,"vocab":task.vocab,"selection":task.selection,"data_fingerprints":data_fingerprints,
        "training_input_sha256":hash(&serde_json::to_vec(&input)?),"parent_buffer_preflight_bytes":bound});
    Ok(Inputs {
        task,
        input,
        record,
    })
}
pub(super) fn inputs(name: &str) -> Result<Inputs> {
    if !tasks()?.iter().any(|t| t == name) {
        return Err("unplanned identification task".into());
    }
    let suite: Value = serde_json::from_str(SUITE)?;
    let row = suite["tasks"]
        .as_array()
        .ok_or("missing original suite")?
        .iter()
        .find(|r| r["name"] == name)
        .ok_or("unknown original task")?;
    assemble(data::build(row)?, name, row["data_fingerprints"].clone())
}
pub(super) fn check_input(inv: &Value, name: &str, x: &Inputs) -> Result<()> {
    let rows = inv["inputs"]
        .as_array()
        .ok_or("missing original input inventory")?;
    if rows.iter().filter(|r| r["task"] == name).count() != 1
        || rows.iter().find(|r| r["task"] == name) != Some(&x.record)
    {
        return Err("original task/input fingerprint differs from inventory".into());
    }
    Ok(())
}
