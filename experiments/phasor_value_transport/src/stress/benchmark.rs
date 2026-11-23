//! Fixed-budget comparisons and the catalog's declared target-and-retention protocol.
use super::source_target::SourceTarget;
use super::*;
use crate::head::Head;
use std::io::{BufRead, BufReader};

fn score_split(
    engine: &mut Engine,
    examples: &[Example],
    split: &str,
    teachers: Option<&numeric_probe::Teachers>,
) -> Result<Value> {
    if let Some(teachers) = teachers {
        return teachers.score(engine, split, examples);
    }
    let score = metrics::evaluate(engine, examples, true)?;
    let correct = score["exact"]
        .as_u64()
        .ok_or("missing exact-sequence score")?;
    let mut counts = json!({"correct":correct,"total":examples.len()});
    if examples[0].decoder.is_some() {
        counts["native_correct"] = json!(score["native_correct"].as_u64().ok_or("missing native token count")?);
    }
    Ok(counts)
}

pub(super) fn score_checkpoint(
    cfg: Config,
    task: &Task,
    bytes: &[u8],
) -> Result<(Value, Option<Value>)> {
    score_checkpoint_with(cfg, task, bytes, None)
}

fn score_checkpoint_with(
    cfg: Config,
    task: &Task,
    bytes: &[u8],
    teachers: Option<&numeric_probe::Teachers>,
) -> Result<(Value, Option<Value>)> {
    if bytes.len() != layout::total_count(&cfg) * 4 {
        return Err("fixed-budget checkpoint shape mismatch".into());
    }
    let weights: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    layout::validate(&cfg, &weights)?;
    let mut engine = Engine::for_scoring(cfg, &weights)?;
    let mut scores = BTreeMap::new();
    for (split, examples) in &task.splits {
        if split == &task.selection || split == "test" || split.starts_with("test/") {
            scores.insert(split, score_split(&mut engine, examples, split, teachers)?);
        }
    }
    let causality = if matches!(
        engine.config.head,
        Head::Transition
            | Head::Recurrent
            | Head::Bounded
            | Head::InputProduct
            | Head::PhasePrefix
            | Head::InputPhase
            | Head::AffinePhase
            | Head::BilinearPhase
            | Head::SharedBilinear
            | Head::ScalarPhase
            | Head::PrefixToeplitz
            | Head::TiedToeplitz
            | Head::TwoFrame
            | Head::TripleTransport
            | Head::BipartiteTriple
            | Head::LookupTripleTransport
            | Head::TriplePhase
            | Head::LookupTriplePhase
            | Head::PeerTriple(_)
            | Head::PeerTriplePhase(_)
            | Head::FourierTransport
            | Head::ReflectionPhase
            | Head::SingleReflection
            | Head::ContextRotation
            | Head::CrossedContext
            | Head::Householder
            | Head::AssociativeReflection
            | Head::PeriodicReflection
    ) {
        let sample = task.splits[&task.selection]
            .iter()
            .max_by_key(|e| e.tokens.len())
            .ok_or("missing held-out sequence for causality check")?;
        let tokens: Vec<_> = sample.tokens.iter().map(|&v| v as u32).collect();
        Some(memory_checks::trained_prefixes(&mut engine, &tokens)?)
    } else {
        None
    };
    Ok((json!(scores), causality))
}

pub enum Source<'a> {
    ComparedHead(&'a str),
    Catalog,
    DeclaredTarget,
}

/// Hash the ordered input sequences alone. For generative tasks use each
/// prompt, and for categorical tasks use its input tokens. Lengths delimit rows.
fn input_digest(examples: &[Example]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;
    let mut digest = Sha256::new();
    digest.update((examples.len() as u64).to_le_bytes());
    for e in examples {
        let input = if e.prompt.is_empty() {
            &e.tokens
        } else {
            &e.prompt
        };
        digest.update((input.len() as u64).to_le_bytes());
        for token in input {
            digest.update(token.to_le_bytes());
        }
    }
    let mut hex = String::with_capacity(64);
    for byte in digest.finalize() {
        write!(hex, "{byte:02x}").unwrap();
    }
    hex
}

pub fn run(source: Source, name: &str, seed: u32, updates: usize, out: &Path) -> Result<()> {
    let result = measure(source, name, seed, updates, out, true)?;
    println!("{result}");
    Ok(())
}

fn directory(out: &Path, arm: &str, name: &str, seed: u32, updates: usize) -> std::path::PathBuf {
    out.join(arm)
        .join(name)
        .join(format!("seed{seed}-{updates}"))
}

fn weight_bytes(e: &Engine) -> Result<Vec<u8>> {
    let values = e.parameters.read()?;
    layout::validate(&e.config, &values)?;
    Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
}

fn verified_file(path: &Path, digest: &Value) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if digest != &hash(&bytes) {
        return Err(format!("checkpoint digest differs at {}", path.display()).into());
    }
    Ok(bytes)
}

/// Replay the frozen catalog record and return the very bytes it authenticates.
pub(super) fn replay_catalog(
    name: &str,
    seed: u32,
    updates: usize,
    out: &Path,
) -> Result<(Value, Vec<u8>)> {
    let result = measure(Source::Catalog, name, seed, updates, out, false)?;
    let file = directory(out, MODEL, name, seed, updates).join("weights.f32");
    let bytes = verified_file(&file, &result["weights_sha256"])?;
    Ok((result, bytes))
}

/// Score transfer inputs at a completed fixed optimizer budget, including
/// observations after the first passing update. See RecallParsed and
/// TripleTransport.FixedEvaluation for the delayed-recall case.
pub fn fixed_probes(
    arm: &str,
    name: &str,
    seed: u32,
    updates: usize,
    out: &Path,
    probes: ProbeSet,
) -> Result<()> {
    let (row, task, fingerprints) = prepare_evaluation(name, probes)?;
    let mut cfg = task_config(&row, &task, false)?;
    cfg.head = Head::parse(arm)?;
    let result = measure(Source::ComparedHead(arm), name, seed, updates, out, false)?;
    let file = directory(out, arm, name, seed, updates).join("weights.f32");
    let scores = transfer_scores(
        cfg,
        &task,
        &file,
        &result["weights_sha256"],
        &result["scores"],
    )?;
    println!(
        "{}",
        json!({"protocol":result["protocol"],"first_99":result["first_99"],
            "weights_sha256":result["weights_sha256"],"scored_with_kernite":KERNITE_VERSION,
            "probe_fingerprints":fingerprints,"scores":scores})
    );
    Ok(())
}

fn transfer_scores(
    cfg: Config,
    task: &Task,
    file: &Path,
    digest: &Value,
    original: &Value,
) -> Result<Value> {
    let bytes = verified_file(file, digest)?;
    let (scores, _) = score_checkpoint(cfg, task, &bytes)?;
    for (split, expected) in original
        .as_object()
        .ok_or("missing original split scores")?
    {
        if scores.get(split) != Some(expected) {
            return Err(
                format!("transfer replay disagrees on {split} at {}", file.display()).into(),
            );
        }
    }
    Ok(scores)
}

/// Evaluate the first crossing and the final scheduled checkpoint on the same
/// transfer examples, including the final checkpoint after an unsuccessful search.
pub fn target_probes(
    name: &str,
    seed: u32,
    updates: usize,
    out: &Path,
    probes: ProbeSet,
    rows: Option<std::ops::Range<usize>>,
) -> Result<()> {
    let result = measure(Source::DeclaredTarget, name, seed, updates, out, false)?;
    let (row, mut task, teachers) = catalog::load_declared(name)?;
    if teachers.is_some() {
        return Err("these transfer probes require a sequence-accuracy case".into());
    }
    let fingerprints = add_probes(name, probes, &row, &mut task)?;
    let mut cfg = task_config(&row, &task, false)?;
    let part = if let Some(range) = rows {
        cfg.max_batch = 1;
        Some(probe_rows::restrict(&mut task, &fingerprints, range)?)
    } else {
        None
    };
    let dir = directory(out, MODEL, name, seed, updates).join("declared-target");
    let checkpoint =
        |file: &str, step: &Value, digest: &Value, original: &Value| -> Result<Value> {
            let scores = transfer_scores(cfg.clone(), &task, &dir.join(file), digest, original)?;
            Ok(json!({"step":step,"weights_sha256":digest,"scores":scores}))
        };
    let (first, last) = at_target_checkpoints(&result, checkpoint)?;
    let mut report = json!({"protocol":result["protocol"],"scored_with_kernite":KERNITE_VERSION,
        "probe_fingerprints":fingerprints,"first":first,"final":last});
    if let Some(info) = part {
        report["probe_rows"] = info;
    }
    println!("{report}");
    Ok(())
}

fn at_target_checkpoints(
    result: &Value,
    checkpoint: impl Fn(&str, &Value, &Value, &Value) -> Result<Value>,
) -> Result<(Value, Value)> {
    let target = &result["source_target"];
    let first = &target["first"];
    let first = if first.is_null() {
        Value::Null
    } else {
        checkpoint(
            "first-target.f32",
            &first["step"],
            &first["weights_sha256"],
            &first["scores"],
        )?
    };
    let last = checkpoint(
        "weights.f32",
        &target["final_step"],
        &result["weights_sha256"],
        &result["scores"],
    )?;
    Ok((first, last))
}

/// Replay the declared source scorer on both saved weight arrays, then change
/// one induction rule at a time without retraining or changing the examples.
pub fn target_ablation(name: &str, seed: u32, updates: usize, out: &Path) -> Result<()> {
    let result = measure(Source::DeclaredTarget, name, seed, updates, out, false)?;
    let (row, mut task, teachers) = catalog::load_declared(name)?;
    if parity_case(name) {
        catalog::add_parity_probes(&mut task)?;
    }
    let cfg = task_config(&row, &task, false)?;
    let Head::PeerTriplePhase(read) = cfg.head else {
        return Err("catalog-ablate requires an induction head".into());
    };
    let variants = read.induction_controls();
    if variants.is_empty() {
        return Err("the selected head has no induction read".into());
    }
    let dir = directory(out, MODEL, name, seed, updates).join("declared-target");
    let (first, last) = at_target_checkpoints(&result, |file, step, digest, original| {
        let bytes = verified_file(&dir.join(file), digest)?;
        let score = |head| {
            let mut control = cfg.clone();
            control.head = head;
            score_checkpoint_with(control, &task, &bytes, teachers.as_ref())
                .map(|(scores, _)| scores)
        };
        let baseline = score(cfg.head)?;
        if &baseline != original {
            return Err(format!("original checkpoint scores changed for {file}").into());
        }
        let mut ablations = BTreeMap::new();
        for &(name, variant) in &variants {
            ablations.insert(name, score(Head::PeerTriplePhase(variant))?);
        }
        Ok(json!({"step":step,"weights_sha256":digest,"scores":baseline,"ablations":ablations}))
    })?;
    println!(
        "{}",
        json!({"protocol":result["protocol"],"scored_with_kernite":KERNITE_VERSION,
        "first":first,"final":last})
    );
    Ok(())
}

fn measure(
    source: Source,
    name: &str,
    seed: u32,
    updates: usize,
    out: &Path,
    train_missing: bool,
) -> Result<Value> {
    let (head, model, arm, row, mut task, teachers) = match source {
        Source::ComparedHead(arm) => {
            let head = Head::parse(arm)?;
            let (row, task) = load(name)?;
            (head, head.id(), arm, row, task, None)
        }
        Source::Catalog => {
            let (row, task) = catalog::load(name)?;
            (architecture::HEAD, MODEL, MODEL, row, task, None)
        }
        Source::DeclaredTarget => {
            let (row, task, teachers) = catalog::load_declared(name)?;
            (architecture::HEAD, MODEL, MODEL, row, task, teachers)
        }
    };
    let schedule = row["case"]["max_updates"].as_u64().unwrap() as usize;
    let observe = super::source_target::observation_interval(&row["case"], task.splits["train"].len())?;
    let batch = row["case"]["batch"].as_u64().unwrap() as usize;
    let target = matches!(source, Source::DeclaredTarget)
        .then(|| SourceTarget::from_case(&row["case"]))
        .transpose()?;
    if updates == 0
        || updates > schedule
        || (updates != schedule && !updates.is_multiple_of(observe))
    {
        return Err("budget must end at a scheduled observation within the task horizon".into());
    }
    let mut cfg = task_config(&row, &task, true)?;
    cfg.head = head;
    let mut protocol = json!({"model":model, "case":name, "seed":seed, "updates":updates,
        "schedule_updates":schedule, "observe":observe, "batch":batch, "selection":task.selection,
        "train_examples":task.splits["train"].len(), "heldout_examples":task.splits[&task.selection].len(),
        "suite_sha256":suite_hash(&row), "kernite_version":KERNITE_VERSION,
        "fp32_affines":precision::FP32_AFFINE_REFERENCE, "bf16":precision::BF16,
        "tf32_backward":precision::TF32_BACKWARD, "parameters":layout::independent_count(&cfg),
        "stored_parameters":layout::total_count(&cfg), "head_parameters":head.parameters(&cfg)});
    let mut initialization = None;
    if let Some(rule) = target {
        protocol["source_target"] = json!({"metric":row["case"]["metric"],"threshold":rule.threshold,
            "after_target":rule.after_target,"search_updates":updates});
    }
    if matches!(source, Source::Catalog | Source::DeclaredTarget) {
        let weights = layout::initialize_model(&cfg, seed);
        let bytes: Vec<_> = weights.iter().flat_map(|w| w.to_le_bytes()).collect();
        protocol["initial_weights_sha256"] = json!(hash(&bytes));
        protocol["shape"] =
            json!({"width":cfg.width,"layers":cfg.layers,"heads":cfg.heads,"vocabulary":cfg.vocab});
        protocol["ordered_inputs_sha256"] = json!({
            "train":input_digest(&task.splits["train"]),
            "held_out":input_digest(&task.splits[&task.selection])
        });
        initialization = Some(weights);
    }
    let mut dir = directory(out, arm, name, seed, updates);
    if target.is_some() {
        dir.push("declared-target");
    }
    let result_file = dir.join("result.json");
    if train_missing {
        fs::create_dir_all(&dir)?;
    } else if !result_file.is_file() {
        return Err("replay requires a completed fixed-budget cell".into());
    }
    let lock = fs::File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join("cell.lock"))?;
    lock.lock()?;
    let weights_file = dir.join("weights.f32");
    let history_file = dir.join("observations.jsonl");
    let saved: Option<Value> = if result_file.try_exists()? {
        let result: Value = serde_json::from_slice(&fs::read(&result_file)?)?;
        if result["protocol"] != protocol {
            return Err("saved benchmark has a different protocol".into());
        }
        Some(result)
    } else if train_missing {
        None
    } else {
        return Err("catalog result disappeared before replay".into());
    };
    let (gpu, final_step) = if let Some(result) = &saved {
        let step = if target.is_some() {
            result["source_target"]["final_step"]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n <= schedule)
                .ok_or("source target checkpoint has an invalid final step")?
        } else {
            updates
        };
        (result["gpu"].clone(), step)
    } else {
        let weights = initialization.unwrap_or_else(|| layout::initialize_model(&cfg, seed));
        let mut e = Engine::new(cfg, &weights)?;
        let mut training = Training::new(&e, seed, schedule)?;
        let held = &task.splits[&task.selection];
        let mut history = fs::File::create(&history_file)?;
        let mut stop = updates;
        let mut first_hit = false;
        for step in 0..=schedule {
            if step.is_multiple_of(observe) || step == stop {
                let score = score_split(&mut e, held, &task.selection, teachers.as_ref())?;
                let correct = score["correct"]
                    .as_u64()
                    .ok_or("missing exact-sequence score")?;
                let mse = score["mse"].as_f64();
                let mut point =
                    json!({"step":step,"correct":correct,"examples_seen":training.seen});
                if let Some(value) = mse {
                    point["mse"] = json!(value);
                }
                if let Some(rule) = target
                    && !first_hit
                    && rule.met(correct, held.len() as u64, mse)?
                {
                    first_hit = true;
                    stop = rule.stop_after_crossing(step, observe, schedule);
                    let bytes = weight_bytes(&e)?;
                    publish(&dir, "first-target.f32", &bytes)?;
                    point["target_weights_sha256"] = json!(hash(&bytes));
                }
                writeln!(history, "{point}")?;
            }
            if step == stop {
                break;
            }
            training.update(&mut e, &task.splits["train"], batch, step)?;
        }
        history.sync_all()?;
        publish(&dir, "weights.f32", &weight_bytes(&e)?)?;
        (json!(e.cuda.name()), stop)
    };
    if matches!(name, "basis-parity16" | "basis-parity20" | "basis-parity24") {
        catalog::add_parity_probes(&mut task)?;
    }
    let mut cfg = task_config(&row, &task, false)?;
    cfg.head = head;
    let bytes = fs::read(weights_file)?;
    let (scores, causality) = score_checkpoint_with(cfg.clone(), &task, &bytes, teachers.as_ref())?;
    let train = task.splits["train"].len();
    let held = task.splits[&task.selection].len() as u64;
    let epoch = train.div_ceil(batch);
    let steps = (0..final_step)
        .step_by(observe)
        .chain(std::iter::once(final_step));
    let mut history = BufReader::new(fs::File::open(history_file)?).lines();
    let (mut first_99, mut initial, mut last) = (None, 0, 0);
    let mut last_mse = None;
    let mut crossing = None;
    for step in steps {
        let point: Value = serde_json::from_str(&history.next().ok_or("missing observation")??)?;
        last = point["correct"]
            .as_u64()
            .filter(|&n| n <= held)
            .ok_or("invalid score")?;
        last_mse = match (teachers.is_some(), point.get("mse")) {
            (false, None) => None,
            (true, Some(value)) => Some(
                value
                    .as_f64()
                    .filter(|mse| mse.is_finite() && *mse >= 0.)
                    .ok_or("invalid numeric observation")?,
            ),
            _ => return Err("observation metric differs from the source protocol".into()),
        };
        if point["step"] != step
            || point["examples_seen"] != step / epoch * train + step % epoch * batch
        {
            return Err("benchmark observation cadence or batch order count mismatch".into());
        }
        if step == 0 {
            initial = last;
        }
        if last * 100 >= 99 * held {
            first_99.get_or_insert(step);
        }
        if let Some(rule) = target
            && crossing.is_none()
            && rule.met(last, held, last_mse)?
        {
            if step > updates {
                return Err("first source target lies beyond the search window".into());
            }
            let digest = point["target_weights_sha256"]
                .as_str()
                .ok_or("first source target is missing its recorded weights digest")?;
            crossing = Some((step, last, last_mse, digest.to_owned()));
        }
    }
    if history.next().is_some()
        || scores[&task.selection]["correct"] != last
        || scores[&task.selection]["mse"].as_f64() != last_mse
    {
        return Err("history failed final held-out checkpoint replay".into());
    }
    let mut result = json!({"protocol":protocol, "initial_correct":initial, "first_99":first_99,
        "scores":scores, "weights_sha256":hash(&bytes), "gpu":gpu});
    if let Some(rule) = target {
        let expected = crossing.as_ref().map_or(updates, |(step, _, _, _)| {
            rule.stop_after_crossing(*step, observe, schedule)
        });
        if final_step != expected {
            return Err("checkpoint did not stop at the declared observation".into());
        }
        let first = if let Some((step, count, recorded_mse, digest)) = crossing {
            let first_bytes = fs::read(dir.join("first-target.f32"))?;
            if hash(&first_bytes) != digest {
                return Err("first source target checkpoint digest mismatch".into());
            }
            let (scores, _) = score_checkpoint_with(cfg, &task, &first_bytes, teachers.as_ref())?;
            if scores[&task.selection]["correct"] != count
                || scores[&task.selection]["mse"].as_f64() != recorded_mse
            {
                return Err("first source target checkpoint score mismatch".into());
            }
            json!({"step":step,"weights_sha256":digest,"scores":scores})
        } else {
            Value::Null
        };
        result["source_target"] = json!({"first":first,"final_step":final_step,
            "final_target_met":rule.met(last,held,last_mse)?,"budget_exhausted":final_step == schedule});
    }
    if let Some(probe) = causality {
        result["causality"] = probe;
    }
    if let Some(saved) = saved {
        if result != saved {
            return Err("fixed-budget result differs from independent replay".into());
        }
    } else {
        publish(&dir, "result.json", &serde_json::to_vec_pretty(&result)?)?;
    }
    Ok(result)
}
