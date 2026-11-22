//! Pinned GPT catalog configurations and held-out splits for extension trials.
//! Only this data adapter sees the catalog case name or answer records.
use super::*;
use basis::sequences;

const PLAN: &str =
    include_str!("../../../../.lake/reproduction/gpt/research/current_grokking/plan.json");

fn source_case(name: &str) -> Result<Value> {
    let registry: Value = serde_json::from_str(sequences::CATALOG)?;
    if hash(PLAN.as_bytes()) != registry["input_sha256"]["plan"] {
        return Err("GPT catalog plan fingerprint changed".into());
    }
    let plan: Value = serde_json::from_str(PLAN)?;
    let case = plan["cases"]
        .as_array()
        .ok_or("missing case plan")?
        .iter()
        .find(|c| c["name"] == name)
        .ok_or("unknown catalog case")?;
    if case["metric"] != "sequence_accuracy" || case["data_seed"] != 1 {
        return Err("expected a seed-1 sequence-accuracy task".into());
    }
    Ok(case.clone())
}

fn sequence_task(d: sequences::dataset::Dataset) -> Task {
    let source = d.task;
    let splits = source
        .splits
        .into_iter()
        .map(|(name, rows)| {
            let rows = rows
                .into_iter()
                .map(|e| Example {
                    task: e.task,
                    tokens: e.tokens,
                    targets: e.targets,
                    changes: e.changes,
                    prompt: e.prompt,
                    answer: e.answer,
                    limit: e.limit,
                    field: e.field,
                    row_index: e.row_index,
                })
                .collect();
            (name, rows)
        })
        .collect();
    Task {
        splits,
        vocab: source.vocab,
        selection: source.selection,
    }
}

pub(super) fn load(name: &str) -> Result<(Value, Task)> {
    let case = source_case(name)?;
    let (task, fingerprints) = if case["kind"] == "arithmetic" {
        let entries = arithmetic::inventory::cases()?;
        let entry = entries
            .iter()
            .find(|r| r["case"] == case)
            .ok_or("arithmetic configuration differs from original plan")?;
        let data = dataset::build(entry)?.ok_or("missing catalog split")?;
        (
            arithmetic_task(data),
            entry["history"]["data_fingerprints"].clone(),
        )
    } else {
        let d = if basis::CASES.contains(&name) {
            basis::dataset::build(name).map(|(d, _)| d)
        } else if parity::CASES.contains(&name) {
            parity::dataset::build(name)
        } else {
            sequences::dataset::build(name)
        }
        .map_err(|e| -> Box<dyn std::error::Error> { e })?;
        let fingerprints = json!(d.original_fingerprints);
        (sequence_task(d), fingerprints)
    };
    let identity = hash(&serde_json::to_vec(&json!([
        case,
        fingerprints,
        task.selection
    ]))?);
    let row = json!({"name":name,"tier":"catalog","selection":task.selection,
        "train_examples":task.splits["train"].len(),"heldout_examples":task.splits[&task.selection].len(),
        "case":case,"catalog_sha256":identity});
    Ok((row, task))
}
