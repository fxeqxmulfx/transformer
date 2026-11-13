//! Original supervised prefixes only. Context and eligibility are derived from
//! training inputs; labels only enter the unchanged categorical observations.
use super::*;
pub(crate) struct Inputs {
    pub task: data::Task,
    pub configuration: Option<binding::Request>,
    pub refusal: Option<String>,
    pub record: Value,
}
pub(crate) fn assemble(task: data::Task, name: &str, fingerprints: Value) -> Result<Inputs> {
    super::name(name)?;
    let train = task
        .splits
        .get("train")
        .filter(|s| !s.is_empty())
        .ok_or("missing original training examples")?;
    let (mut count, mut tokens, mut context) = (0usize, 0usize, 0usize);
    for e in train {
        if e.tokens.len() != e.targets.len()
            || e.changes.len() != e.targets.len()
            || e.tokens.iter().any(|&x| x < 0 || x as usize >= task.vocab)
            || e.targets
                .iter()
                .any(|&y| y != -100 && (y < 0 || y as usize >= task.vocab))
        {
            return Err("malformed original training example".into());
        }
        for (i, _) in e.targets.iter().enumerate().filter(|(_, y)| **y >= 0) {
            count += 1;
            tokens = tokens
                .checked_add(i + 1)
                .ok_or("original prefix count overflow")?;
            context = context.max(i + 1);
        }
    }
    let eligible = context > 2;
    let mut refusal = if !eligible {
        Some("training prefixes have no context beyond two tokens".to_string())
    } else if !(2..=1024).contains(&task.vocab)
        || context > 4096
        || count > 200_000
        || tokens > 20_000_000
    {
        Some("original prefix/vocabulary storage exceeds declared input caps".to_string())
    } else {
        None
    };
    let p = protocol()?;
    let configuration = if refusal.is_none() {
        let observations = train
            .iter()
            .flat_map(|e| {
                e.targets
                    .iter()
                    .enumerate()
                    .filter(|(_, y)| **y >= 0)
                    .map(|(i, &y)| component::Observation {
                        tokens: e.tokens[..=i].iter().map(|&x| x as usize).collect(),
                        label: y as usize,
                    })
            })
            .collect();
        let request = binding::Request {
            discovery: component::Request {
                input: component::Input {
                    vocabulary: task.vocab,
                    context,
                    observations,
                },
                policy: serde_json::from_value(p["discovery"].clone())?,
            },
            training: serde_json::from_value(p["training"].clone())?,
        };
        match request.validate() {
            Ok(_) => Some(request),
            Err(error) => {
                refusal = Some(error.to_string());
                None
            }
        }
    } else {
        None
    };
    let record = json!({"task":name,"vocabulary":task.vocab,"primary_split":task.selection,
        "data_fingerprints":fingerprints,"observed_targets":count,"stored_prefix_tokens":tokens,
        "training_context":context,"shape_eligible":eligible,"input_refusal":refusal,
        "configuration_sha256":configuration.as_ref().map(|c| serde_json::to_vec(c).map(|b| hash(&b))).transpose()?,
        "selection_inputs":"original visible supervised training prefixes and raw labels"});
    Ok(Inputs {
        task,
        configuration,
        refusal,
        record,
    })
}
pub(crate) fn build(suite: &Value, name: &str) -> Result<Inputs> {
    let rows: Vec<_> = suite["tasks"]
        .as_array()
        .ok_or("missing suite tasks")?
        .iter()
        .filter(|r| r["name"] == name)
        .collect();
    if rows.len() != 1 {
        return Err("unknown or repeated original task".into());
    }
    // The inherited generator verifies all original split fingerprints.
    assemble(
        data::build(rows[0])?,
        name,
        rows[0]["data_fingerprints"].clone(),
    )
}
pub(crate) fn check(inv: &Value, name: &str, input: &Inputs) -> Result<()> {
    let records: Vec<_> = inv["inputs"]
        .as_array()
        .ok_or("missing original input records")?
        .iter()
        .filter(|r| r["task"] == name)
        .collect();
    if records.len() != 1 || *records[0] != input.record {
        return Err("original input/shape/configuration binding changed".into());
    }
    Ok(())
}
