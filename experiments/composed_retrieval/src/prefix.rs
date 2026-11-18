//! Source-verified synthetic cohorts. Only the harness sees the case,
//! original tensors, generation rules and source fingerprints.
use crate::{
    Result,
    data::{CaseSpec, OFFSET_PROTOCOL, PROTOCOL, Protocol},
};
use baseline::{data::Task, hash};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};
use synthetic_data::{generative, generative::typed, sequences};

const REWRITE_NAME: &str = "composed-retrieval-rewrite6-lookup-output-offsets-v1";
const PREFIX_NAME: &str = "composed-retrieval-synthetic-prefix9-output-offsets-v1";
pub const NAME: &str = if cfg!(feature = "rewrite-cohort") {
    REWRITE_NAME
} else {
    PREFIX_NAME
};

#[derive(Deserialize)]
pub(crate) struct Source {
    pub vocabulary: usize,
    pub context: usize,
    pub training_targets: usize,
    pub metadata_sha256: String,
}
#[derive(Deserialize)]
struct Definition {
    name: String,
    training_protocol_sha256: String,
    plan_sha256: String,
    sequences_sha256: String,
    typed_sha256: String,
    remaining_sha256: String,
    generative_sha256: Option<String>,
    train: usize,
    batch: usize,
    selection: String,
    metric: String,
    target: f64,
    cases: BTreeMap<String, Source>,
}
pub fn configure(p: &mut Protocol) -> Result<()> {
    configure_from(p, PROTOCOL, cfg!(feature = "rewrite-cohort"))
}
pub(crate) fn configure_from(p: &mut Protocol, definition: &str, rewrite: bool) -> Result<()> {
    let d: Definition = serde_json::from_str(definition)?;
    let roster_matches = if rewrite {
        d.cases.len() == generative::CASES.len() + 1
            && generative::CASES.iter().all(|id| d.cases.contains_key(*id))
            && d.cases.contains_key("suite-lookup")
            && d.generative_sha256 == Some(hash(generative::PROTOCOL.as_bytes()))
    } else {
        d.cases.len() == 9 && d.generative_sha256.is_none()
    };
    if d.name != (if rewrite { REWRITE_NAME } else { PREFIX_NAME })
        || !roster_matches
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
    #[cfg(feature = "rewrite-cohort")]
    if generative::CASES.contains(&id) {
        let d = generative::dataset::build(id)?;
        let meta = generative::dataset::metadata(&d);
        let mut context = 0;
        for row in d.task.splits.values().flatten() {
            crate::generation::validate_source(row, spec.vocabulary)?;
            context = context.max(row.prompt.len() + row.limit - 1);
        }
        if spec.metadata_sha256.as_deref() != Some(&hash(&serde_json::to_vec(&meta)?))
            || spec.context != context
            || spec.vocabulary != d.task.vocab
            || spec.selection != d.task.selection
            || d.task.splits["train"].len() != spec.train
        {
            return Err(
                format!("source tensors, generation limits or metadata differ on {id}").into(),
            );
        }
        return Ok((d.task, meta));
    }
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
    inventory_from(p, inputs, PROTOCOL, cfg!(feature = "rewrite-cohort"))
}
pub(crate) fn inventory_from(
    p: &Protocol,
    inputs: &Path,
    definition: &str,
    rewrite: bool,
) -> Result<Value> {
    let d: Definition = serde_json::from_str(definition)?;
    source_inventory(
        p,
        inputs,
        definition,
        &d.plan_sha256,
        &d.cases,
        rewrite,
        original,
    )
}

pub(crate) fn source_inventory(
    p: &Protocol,
    inputs: &Path,
    definition: &str,
    plan_sha256: &str,
    sources: &BTreeMap<String, Source>,
    rollout: bool,
    load: impl Fn(&str, &CaseSpec) -> Result<(Task, Value)>,
) -> Result<Value> {
    let bytes = fs::read(inputs.join("plan.json"))?;
    if hash(&bytes) != plan_sha256 {
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
        let source_label = if id == "random-lm-control" {
            "control-seed0"
        } else {
            id
        };
        if row["kind"] != "synthetic"
            || row["batch"] != spec.batch
            || row["data_seed"] != p.data_seed
            || row["target"] != spec.target
            || row["metric"] != spec.metric
            || row["data"]["label"] != source_label
        {
            return Err(format!("original synthetic case configuration changed on {id}").into());
        }
        let (task, meta) = load(id, spec)?;
        let mut counts = BTreeMap::new();
        for (name, rows) in &task.splits {
            let targets = rows
                .iter()
                .flat_map(|r| &r.targets)
                .filter(|&&t| t >= 0)
                .count();
            let limits: usize = rows.iter().map(|r| r.limit).sum();
            one_pass += 8 * spec.vocabulary as u64 * (targets + limits) as u64;
            counts.insert(
                name,
                json!({"examples":rows.len(),"targets":targets,
                "max_context":rows.iter().map(|r|r.tokens.len()).max().ok_or("empty split")?}),
            );
        }
        let splits = json!(counts);
        if !meta["splits"].is_null() && splits != meta["splits"] {
            return Err("source row dimensions disagree".into());
        }
        if splits["train"]["targets"] != sources[id].training_targets {
            return Err("training target multiplicity changed".into());
        }
        entries.insert(id,json!({"dimensions":{"vocabulary":spec.vocabulary,"context":spec.context,"splits":splits},
            "split_fingerprints":meta}));
    }
    let fits = p.cases.len() * p.seeds.len() * p.modes.len() * 2;
    let bytes = one_pass * 4 * p.seeds.len() as u64 * p.modes.len() as u64;
    let size_key = if rollout {
        "raw_bytes_upper_bound"
    } else {
        "raw_bytes"
    };
    Ok(
        json!({"protocol_sha256":hash(definition.as_bytes()),"cases":entries,"fits":fits,(size_key):bytes}),
    )
}
