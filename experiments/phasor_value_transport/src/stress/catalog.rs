//! Pinned GPT catalog configurations and held-out splits for extension trials.
//! Only this data adapter sees the catalog case name or answer records.
use super::*;
use basis::{generative, previous as remaining_synthetic, sequences};

const PLAN: &str =
    include_str!("../../../../.lake/reproduction/gpt/research/current_grokking/plan.json");

pub(super) fn registered(contents: &str) -> Result<Vec<(&str, &str)>> {
    let original = cases()?;
    let mut registry = Vec::new();
    for line in contents.lines() {
        let (tier, name) = line.split_once('\t').ok_or("invalid extension row")?;
        if !matches!(tier, "short" | "long")
            || name.is_empty()
            || name == "arithmetic-cubic-product"
            || original.iter().any(|r| r["name"] == name)
            || registry.iter().any(|(existing, _)| *existing == name)
        {
            return Err("invalid or duplicate extension membership".into());
        }
        source_case(name)?;
        registry.push((name, tier));
    }
    Ok(registry)
}

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

/// The official scratchpad-free parity catalog includes entirely unseen
/// 32- and 64-bit prompts with the same bit, answer and EOS token IDs as Basis.
/// Add them only after training, while loading the selected checkpoint.
pub(super) fn add_parity_probes(task: &mut Task) -> Result<()> {
    let (_, mut other) = load("suite-parity-none")?;
    for length in [32, 64] {
        let key = format!("test/length-{length}");
        let examples = other
            .splits
            .remove(&key)
            .ok_or("missing catalog parity probe")?;
        for e in &examples {
            if e.prompt.len() != length + 2
                || e.answer.len() != 2
                || !e
                    .tokens
                    .iter()
                    .chain(&e.answer)
                    .all(|&v| (0..task.vocab as i64).contains(&v))
            {
                return Err("incompatible Basis parity probe encoding".into());
            }
        }
        task.splits
            .insert(format!("test/catalog-parity-{length}"), examples);
    }
    Ok(())
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
        } else if frequency::CASES.contains(&name) {
            frequency::dataset::build(name)
        } else if generative::CASES.contains(&name) {
            generative::dataset::build(name)
        } else if generative::typed::CASES.contains(&name) {
            generative::typed::dataset::build(name)
        } else if remaining_synthetic::CASES.contains(&name) {
            remaining_synthetic::dataset::build(name)
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
