use super::*;
pub(super) struct Inputs {
    pub task: data::Task,
    pub config: Configuration,
    pub record: Value,
}
/// Port of metrics.rs field_prediction's grouping. Adding the per-example base
/// permutes groups and leaves their fibers unchanged. No future/answer is read.
fn readout(task: &data::Task) -> Result<(Vec<usize>, &'static str)> {
    let train = task
        .splits
        .get("train")
        .ok_or("missing original training split")?;
    let field = train.first().ok_or("empty training split")?.field.is_some();
    if train.iter().any(|e| e.field.is_some() != field) {
        return Err("mixed declared metric readout".into());
    }
    Ok(if field {
        (
            (0..task.vocab)
                .map(|c| (c as i64 - task.vocab as i64 / 2).rem_euclid(97) as usize)
                .collect(),
            "original_residue_pooling",
        )
    } else {
        ((0..task.vocab).collect(), "original_raw_identity")
    })
}
pub(super) fn assemble(task: data::Task, name: &str, fingerprints: Value) -> Result<Inputs> {
    super::name(name)?;
    convex_stress_relabel::check_layout(&task)?;
    let p = protocol()?;
    let (readout, kind) = readout(&task)?;
    let config = Configuration {
        input: component::Input {
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
            policy: serde_json::from_value(p["discovery"].clone())?,
        },
        training: serde_json::from_value(p["training"].clone())?,
        prefix: serde_json::from_value(p["prefix"].clone())?,
        certificate: serde_json::from_value(p["certificate"].clone())?,
        readout: Some(readout),
    };
    let record = json!({"task":name,"vocabulary":task.vocab,"primary_split":task.selection,
        "data_fingerprints":fingerprints,"configuration_sha256":hash(&serde_json::to_vec(&config)?),
        "readout":kind,"construction_inputs":"original observed categorical pairs and raw labels",
        "readout_changes_features_or_ce":false});
    Ok(Inputs {
        task,
        config,
        record,
    })
}
pub(super) fn build(suite: &Value, name: &str) -> Result<Inputs> {
    let rows: Vec<_> = suite["tasks"]
        .as_array()
        .ok_or("missing suite tasks")?
        .iter()
        .filter(|r| r["name"] == name)
        .collect();
    if rows.len() != 1 {
        return Err("unknown or duplicate original task".into());
    }
    // The inherited native generator verifies every actual original fingerprint.
    assemble(
        data::build(rows[0])?,
        name,
        rows[0]["data_fingerprints"].clone(),
    )
}
pub(super) fn check(inv: &Value, name: &str, x: &Inputs) -> Result<()> {
    let records: Vec<_> = inv["inputs"]
        .as_array()
        .ok_or("missing input records")?
        .iter()
        .filter(|r| r["task"] == name)
        .collect();
    if records.len() != 1 || *records[0] != x.record {
        return Err("original input/metric/configuration binding changed".into());
    }
    Ok(())
}
