//! The complete supplied-prefix synthetic subgroup. Native adapters retain
//! BOS, all target masks and all original short, long and novelty splits.
use crate::{
    Result,
    data::{CaseSpec, OFFSET_PROTOCOL, PROTOCOL, Protocol},
};
use baseline::{data::Task, hash};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};
use synthetic_data::{generative::typed, sequences};

pub const NAME: &str = "composed-retrieval-synthetic-prefix9-output-offsets-v1";

#[derive(Deserialize)]
struct Source {
    vocabulary: usize,
    context: usize,
    training_targets: usize,
    metadata_sha256: String,
}
#[derive(Deserialize)]
struct Definition {
    name: String,
    training_protocol_sha256: String,
    plan_sha256: String,
    sequences_sha256: String,
    typed_sha256: String,
    remaining_sha256: String,
    train: usize,
    batch: usize,
    selection: String,
    metric: String,
    target: f64,
    cases: BTreeMap<String, Source>,
}
pub fn configure(p: &mut Protocol) -> Result<()> {
    let d: Definition = serde_json::from_str(PROTOCOL)?;
    if d.name != NAME
        || d.cases.len() != 9
        || d.training_protocol_sha256 != hash(OFFSET_PROTOCOL.as_bytes())
        || d.sequences_sha256 != hash(sequences::PROTOCOL.as_bytes())
        || d.typed_sha256 != hash(typed::PROTOCOL.as_bytes())
        || d.remaining_sha256 != hash(synthetic_data::PROTOCOL.as_bytes())
        || p.output_offset_bound != Some(16.)
    {
        return Err("changed synthetic cohort or training protocol".into());
    }
    p.cases = d
        .cases
        .into_iter()
        .map(|(id, s)| {
            (
                id,
                CaseSpec {
                    vocabulary: s.vocabulary,
                    context: s.context,
                    batch: d.batch,
                    train: d.train,
                    selection: d.selection.clone(),
                    metric: d.metric.clone(),
                    target: d.target,
                    metadata_sha256: Some(s.metadata_sha256),
                    source_fingerprints: None,
                },
            )
        })
        .collect();
    p.name = d.name;
    Ok(())
}

pub fn original(id: &str, spec: &CaseSpec) -> Result<(Task, Value)> {
    let d = match id {
        "suite-mqar" | "suite-lookup" | "suite-blocks" | "suite-dyck" => {
            sequences::dataset::build(id)?
        }
        "suite-dyck-2" | "suite-dyck-3" => typed::dataset::build(id)?,
        "suite-crasp-depth-1" | "suite-crasp-depth-2" | "suite-crasp-depth-3" => {
            synthetic_data::dataset::build(id)?
        }
        _ => return Err("outside the original aligned synthetic cases".into()),
    };
    let meta = d.metadata();
    if spec.metadata_sha256.as_deref() != Some(&hash(&serde_json::to_vec(&meta)?))
        || d.task.vocab != spec.vocabulary
        || d.task.selection != spec.selection
        || d.task.splits["train"].len() != spec.train
        || d.task
            .splits
            .values()
            .flatten()
            .map(|r| r.tokens.len())
            .max()
            != Some(spec.context)
        || d.task.splits.values().flatten().any(|r| {
            !r.prompt.is_empty()
                || !r.answer.is_empty()
                || r.field.is_some()
                || r.row_index.is_some()
                || r.limit != 0
                || r.tokens.len() != r.targets.len()
                || r.targets.len() != r.changes.len()
        })
    {
        return Err(format!("source tensors, masks or metadata differ on {id}").into());
    }
    Ok((d.task, meta))
}

pub fn preflight(p: &Protocol, inputs: &Path) -> Result<Value> {
    let d: Definition = serde_json::from_str(PROTOCOL)?;
    let bytes = fs::read(inputs.join("plan.json"))?;
    if hash(&bytes) != d.plan_sha256 {
        return Err("original DiffGPT plan changed".into());
    }
    let plan: Value = serde_json::from_slice(&bytes)?;
    let rows = plan["cases"].as_array().ok_or("original case roster")?;
    let mut entries = BTreeMap::new();
    let mut one_pass = 0u64;
    for (id, spec) in &p.cases {
        let matching: Vec<_> = rows.iter().filter(|r| r["name"] == id.as_str()).collect();
        let [row] = matching.as_slice() else {
            return Err("missing or duplicate original case".into());
        };
        if row["batch"] != spec.batch
            || row["data_seed"] != p.data_seed
            || row["target"] != spec.target
            || row["metric"] != spec.metric
            || row["data"]["label"] != id.as_str()
        {
            return Err("original case configuration changed".into());
        }
        let (_, meta) = original(id, spec)?;
        if meta["splits"]["train"]["targets"] != d.cases[id].training_targets {
            return Err("training target multiplicity changed".into());
        }
        for split in meta["splits"]
            .as_object()
            .ok_or("split dimensions")?
            .values()
        {
            one_pass += 8 * spec.vocabulary as u64 * split["targets"].as_u64().ok_or("targets")?;
        }
        entries.insert(id,json!({"dimensions":{"vocabulary":spec.vocabulary,"context":spec.context,"splits":meta["splits"]},
            "split_fingerprints":meta}));
    }
    let fits = p.cases.len() * p.seeds.len() * p.modes.len() * 2;
    Ok(
        json!({"protocol_sha256":hash(PROTOCOL.as_bytes()),"cases":entries,"fits":fits,
        "raw_bytes":one_pass * 4 * p.seeds.len() as u64 * p.modes.len() as u64}),
    )
}
