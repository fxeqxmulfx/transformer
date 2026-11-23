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

fn definition(name: &str) -> Result<Value> {
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
    if case["data_seed"] != 1 {
        return Err("expected a seed-1 source dataset".into());
    }
    Ok(case.clone())
}

fn source_case(name: &str) -> Result<Value> {
    let case = definition(name)?;
    if case["metric"] != "sequence_accuracy" {
        return Err("expected a seed-1 sequence-accuracy task".into());
    }
    Ok(case)
}

pub(super) fn sequence_example(e: basis::data::Example) -> Example {
    Example {
        task: e.task,
        tokens: e.tokens,
        targets: e.targets,
        changes: e.changes,
        prompt: e.prompt,
        answer: e.answer,
        limit: e.limit,
        field: e.field,
        decoder: None,
        row_index: e.row_index,
    }
}

fn sequence_task(source: basis::data::Task) -> Task {
    let splits = source
        .splits
        .into_iter()
        .map(|(name, rows)| (name, rows.into_iter().map(sequence_example).collect()))
        .collect();
    Task {
        splits,
        vocab: source.vocab,
        selection: source.selection,
    }
}

/// Reuse the original modular-chain generator and check its three dense
/// tensor fingerprints against the independently audited seed-one record.
fn modular_chain(name: &str, case: &Value) -> Result<(Task, Value)> {
    use composed_retrieval::chain_data;

    let number = |field: &str| -> Result<usize> {
        Ok(usize::try_from(
            case["data"][field]
                .as_u64()
                .ok_or("missing chain dimension")?,
        )?)
    };
    let prime = number("prime")?;
    let steps = number("steps")?;
    let seed = case["data_seed"]
        .as_u64()
        .ok_or("missing chain data seed")?;
    let protocol: Value = serde_json::from_str(composed_retrieval::data::CHAIN_PROTOCOL)?;
    let expected = protocol["cases"][name]
        .as_str()
        .ok_or("missing chain tensor fingerprint")?;
    if protocol["data_seed"] != seed
        || protocol["chain_source_manifest_sha256"] != hash(chain_data::SOURCES.as_bytes())
    {
        return Err("modular-chain source provenance differs".into());
    }
    let source = chain_data::build(prime, steps, seed).map_err(|e| e.to_string())?;
    let mut fingerprints = BTreeMap::new();
    for (split, rows) in &source.splits {
        let pin = chain_data::fingerprints(rows, source.vocab).map_err(|e| e.to_string())?;
        fingerprints.insert(split, pin.dense_tensor_sha256);
    }
    let fingerprints = json!(fingerprints);
    if hash(&serde_json::to_vec(&fingerprints)?) != expected {
        return Err("modular-chain tensors differ from the original source".into());
    }
    Ok((sequence_task(source), fingerprints))
}

/// The reviewed GPT probe port validates both original tensor fingerprints
/// before releasing the order-discrimination records to the stress harness.
pub(super) fn probe_data(case: &Value) -> basis::Result<(probes::dataset::Dataset, Value)> {
    let source = probes::cases()?
        .into_iter()
        .find(|r| r["case"] == *case)
        .ok_or("probe configuration differs from the frozen GPT source")?;
    let helper = env::var_os("PHASOR_GPT_PROBE_HELPER")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../.lake/reproduction/gpt-probe-reference")
        });
    let original = probes::dataset::build(&source, &helper)?;
    Ok((original, source["history"]["data_fingerprints"].clone()))
}

fn mechanism_probe(case: &Value) -> basis::Result<(Task, Value, Option<numeric_probe::Teachers>)> {
    let (original, hashes) = probe_data(case)?;
    let mut teachers = original
        .numeric_step
        .zip(original.numeric_offset)
        .map(|(step, offset)| numeric_probe::Teachers {
            step,
            offset,
            splits: BTreeMap::new(),
        });
    let splits = original
        .splits
        .into_iter()
        .map(|(split, data)| {
            if let (Some(t), Some(words)) = (&mut teachers, data.teacher_words) {
                t.splits.insert(split.clone(), words);
            }
            let examples = data
                .records
                .into_iter()
                .map(|row| Example {
                    task: row.task,
                    tokens: row.tokens,
                    targets: row.targets,
                    changes: row.changes,
                    prompt: row.prompt,
                    answer: row.answer,
                    limit: row.generation_limit,
                    field: None,
                    decoder: None,
                    row_index: None,
                })
                .collect();
            (split, examples)
        })
        .collect();
    Ok((
        Task {
            splits,
            vocab: original.vocabulary,
            selection: original.selection,
        },
        hashes,
        teachers,
    ))
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

/// Extend the catalog's scratchpad-free test generator past its published
/// 64-bit split. A second oracle recounts every source bit and EOS target.
/// These sets enter evaluation only after a checkpoint has been selected.
pub(super) fn add_extended_parity_probes(
    task: &mut Task,
    lengths: &[usize],
) -> basis::Result<BTreeMap<String, String>> {
    let protocol: Value = serde_json::from_str(parity::PROTOCOL)?;
    let mut spec = protocol["definitions"]["suite-parity-none"].clone();
    spec.as_object_mut()
        .and_then(|s| s.remove("longer_lengths"))
        .filter(|lengths| lengths == &json!([32, 64]))
        .ok_or("unexpected source parity probe lengths")?;
    spec["min_length"] = Value::Null;
    let mut hashes = BTreeMap::new();
    for &length in lengths {
        spec["length"] = json!(length);
        // The source generator validates this bound even without index hints.
        spec["number_limit"] = json!(length);
        let rows = parity::dataset::generate(&spec, "test", 256)?;
        for row in &rows {
            basis::independent::check(&spec, row)?;
        }
        let distinct: std::collections::BTreeSet<_> = rows.iter().map(|r| &r.prompt).collect();
        if distinct.len() != rows.len() {
            return Err("duplicate extended parity test prompt".into());
        }
        let split = format!("test/extended-parity-{length}");
        hashes.insert(split.clone(), basis::wire::split_hash(&spec, "test", &rows));
        task.splits
            .insert(split, rows.into_iter().map(sequence_example).collect());
    }
    Ok(hashes)
}

pub(super) fn load(name: &str) -> Result<(Value, Task)> {
    let case = source_case(name)?;
    load_sequence(name, case)
}

/// Keep unquantized numeric teachers exclusively in the external scorer.
pub(super) fn load_declared(name: &str) -> Result<(Value, Task, Option<numeric_probe::Teachers>)> {
    let case = definition(name)?;
    if case["metric"] == "sequence_accuracy" {
        let (row, task) = load_sequence(name, case)?;
        return Ok((row, task, None));
    }
    if source_target::is_scalar_anchor(&case) {
        let (source, fingerprints) = noisy_anchors::build(&case).map_err(|e| e.to_string())?;
        let (row, task) = with_identity(name, case, fingerprints, sequence_task(source))?;
        return Ok((row, task, None));
    }
    if case["kind"] == "text" && case["metric"] == "negative_loss" {
        let (source, fingerprints) = text_window::build(&case).map_err(|e| e.to_string())?;
        let (row, task) = with_identity(name, case, fingerprints, sequence_task(source))?;
        return Ok((row, task, None));
    }
    if case["metric"] != "negative_mse" || case["kind"] != "probe" {
        return Err(format!("unsupported source metric {}", case["metric"]).into());
    }
    let (task, hashes, teachers) = mechanism_probe(&case).map_err(|e| e.to_string())?;
    let teachers = teachers.ok_or("numeric probe lacks its unquantized teachers")?;
    let (row, task) = with_identity(name, case, hashes, task)?;
    Ok((row, task, Some(teachers)))
}

fn load_sequence(name: &str, case: Value) -> Result<(Value, Task)> {
    let (task, fingerprints) = if case["kind"] == "arithmetic" {
        let entries = arithmetic::inventory::cases()?;
        let entry = entries
            .iter()
            .find(|r| r["case"] == case)
            .ok_or("arithmetic configuration differs from original plan")?;
        let data = dataset::build_catalog(entry)?;
        (
            arithmetic_task(data),
            entry["history"]["data_fingerprints"].clone(),
        )
    } else if case["kind"] == "chain" {
        modular_chain(name, &case)?
    } else if name == "associative-dividend-n2-l1-w0" {
        let control = cases()?.into_iter().find(|r| r["name"] == associative_copy::CONTROL)
            .ok_or("missing canonical divisor-copy reference")?;
        let (task, hashes) = associative_copy::build(&case, &control)
            .map_err(|e| -> Box<dyn std::error::Error> { e })?;
        (sequence_task(task), hashes)
    } else if case["kind"] == "associative"
        && matches!(case["data"]["labels"].as_str(), Some("division" | "random")) {
        let (task, hashes) = associative_equations::build(&case)
            .map_err(|e| -> Box<dyn std::error::Error> { e })?;
        (sequence_task(task), hashes)
    } else if case["kind"] == "differenced_division" {
        let order = case["data"]["order"]
            .as_u64()
            .ok_or("missing difference order")?;
        let seed = case["data_seed"].as_u64().ok_or("missing division data seed")?;
        let source = differenced::build(u32::try_from(order)?, seed)?;
        let [train, validation] = source.fingerprints;
        (
            pair_task(name, source.vocabulary, source.train, source.validation),
            json!({"train": train, "validation": validation}),
        )
    } else if case["kind"] == "probe" {
        let (task, hashes, teachers) = mechanism_probe(&case).map_err(|err| err.to_string())?;
        if teachers.is_some() {
            return Err("expected a categorical sequence-accuracy probe".into());
        }
        (task, hashes)
    } else if let Some(original) = cases()?.into_iter().find(|r| r["name"] == name) {
        for key in ["kind", "data", "data_seed"] {
            if case[key] != original["case"][key] {
                return Err("canonical and broader dataset definitions disagree".into());
            }
        }
        let task = data::build(&original)?;
        if original["selection"] != task.selection {
            return Err("canonical source selection changed".into());
        }
        (task, original["data_fingerprints"].clone())
    } else if case["kind"] == "fashion" {
        let dir = std::env::var_os("PHASOR_FASHION_DATA")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../.lake/reproduction/fashion-mnist"));
        let (source, hashes) = fashion::build(&case, &dir)
            .map_err(|e| -> Box<dyn std::error::Error> { e })?;
        (sequence_task(source), hashes)
    } else if case["kind"] == "recurrence"
        && matches!(case["data"]["tokenizer"].as_str(), Some("mash" | "mash-half")) {
        let (source, hashes, decoders) = temporal::build_mash(&case)
            .map_err(|e| -> Box<dyn std::error::Error> { e })?;
        let mut task = sequence_task(source);
        for row in task.splits.values_mut().flatten() {
            let index = usize::try_from(row.row_index.ok_or("missing temporal decoder index")?)?;
            row.decoder = Some(*decoders.get(index).ok_or("bad temporal decoder index")?);
        }
        (task, hashes)
    } else if case["kind"] == "recurrence" {
        let (task, hashes) = temporal::build(&case)
            .map_err(|e| -> Box<dyn std::error::Error> { e })?;
        (sequence_task(task), hashes)
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
        (sequence_task(d.task), fingerprints)
    };
    with_identity(name, case, fingerprints, task)
}

pub(super) fn identity(case: &Value, fingerprints: &Value, selection: &Value) -> Result<String> {
    Ok(hash(&serde_json::to_vec(&json!([
        case,
        fingerprints,
        selection
    ]))?))
}

fn with_identity(
    name: &str,
    case: Value,
    fingerprints: Value,
    task: Task,
) -> Result<(Value, Task)> {
    let identity = identity(&case, &fingerprints, &json!(task.selection))?;
    let row = json!({"name":name,"tier":"catalog","selection":task.selection,
        "train_examples":task.splits["train"].len(),"heldout_examples":task.splits[&task.selection].len(),
        "case":case,"catalog_sha256":identity,"data_fingerprints":fingerprints});
    Ok((row, task))
}

/// List the source metric and actual native adapter availability before training.
pub fn inventory(registry: &str) -> Result<()> {
    let plan: Value = serde_json::from_str(PLAN)?;
    let added: BTreeMap<_, _> = registered(registry)?.into_iter().collect();
    let cohort = cases()?;
    let mut records = Vec::new();
    for case in plan["cases"].as_array().ok_or("missing source cases")? {
        let name = case["name"].as_str().ok_or("missing source name")?;
        let membership = if let Some(row) = cohort.iter().find(|r| r["name"] == name) {
            format!(
                "canonical-{}",
                row["tier"].as_str().ok_or("missing canonical tier")?
            )
        } else if let Some(tier) = added.get(name) {
            format!("{tier}-extension")
        } else if name == "arithmetic-cubic-product" {
            "NV11".into()
        } else {
            "broader".into()
        };
        let mut record = json!({"case":name,"membership":membership,"metric":case["metric"],
            "target":case["target"],"budget":case["max_updates"],"observe":case["observe"],
            "after_target":case["after_target"]});
        match load_declared(name) {
            Ok((row, _, _)) => {
                record["source_adapter"] = json!("ready");
                record["effective_observe"] = json!(source_target::observation_interval(
                    case, row["train_examples"].as_u64().ok_or("missing training row count")? as usize
                )?);
                record["selection"] = row["selection"].clone();
                record["heldout_examples"] = row["heldout_examples"].clone();
                record["catalog_sha256"] = row["catalog_sha256"].clone();
            }
            Err(error) => {
                record["source_adapter"] = json!("unavailable");
                record["reason"] = json!(error.to_string());
            }
        }
        records.push(record);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "source_plan_sha256":hash(PLAN.as_bytes()),"cases":records
        }))?
    );
    Ok(())
}
