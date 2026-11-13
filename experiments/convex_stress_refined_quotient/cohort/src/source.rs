//! Read immutable replay evidence; never rerun the categorical search here.
use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub root: PathBuf,
    pub binary: PathBuf,
    pub inventory_sha256: String,
    pub source_fingerprint: String,
    pub binary_sha256: String,
    pub producer_commit: String,
}
pub(crate) fn evidence(path: &Path) -> Result<Value> {
    Ok(json!({"path":path,"sha256":digest(path)?,"bytes":fs::metadata(path)?.len()}))
}
pub(crate) fn tree(path: &Path, files: &mut Vec<Value>) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let mut entries: Vec<_> = fs::read_dir(path)?.collect::<std::result::Result<_, _>>()?;
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        if entry.file_type()?.is_dir() {
            tree(&entry.path(), files)?;
        } else {
            files.push(evidence(&entry.path())?);
        }
    }
    Ok(())
}
pub(crate) fn exit(root: &Path, filename: &str) -> Result<i32> {
    Ok(fs::read_to_string(root.join(filename))?.trim().parse()?)
}
pub(crate) fn prepare(
    reference: &Reference,
    planned: &[String],
    all: &[String],
) -> Result<Vec<Value>> {
    pins(reference)?;
    validate_reference(reference, planned, all)?;
    planned.iter().map(|t| case_record(reference, t)).collect()
}
pub(crate) fn pins(reference: &Reference) -> Result<()> {
    let p = protocol()?;
    if reference.source_fingerprint != p["source_fingerprint"]
        || reference.source_fingerprint != parent::fingerprint()
        || reference.producer_commit != p["source_producer"]
        || reference.binary_sha256 != p["source_binary_sha256"]
    {
        return Err("discovery producer differs from the frozen original catalogue".into());
    }
    Ok(())
}
pub(crate) fn validate_reference(
    reference: &Reference,
    planned: &[String],
    all: &[String],
) -> Result<()> {
    let inv: Value = read(&reference.root.join("inventory.json"))?;
    if digest(&reference.binary)? != reference.binary_sha256
        || digest(&reference.root.join("inventory.json"))? != reference.inventory_sha256
        || inv["source_fingerprint"] != reference.source_fingerprint
        || inv["binary_sha256"] != reference.binary_sha256
        || inv["producer_commit"] != reference.producer_commit
        || inv["planned"] != json!(planned)
        || inv["tasks"] != json!(all)
    {
        return Err("source inventory, binary or complete task scope differs".into());
    }
    Ok(())
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Stage {
    pub prime: u16,
    pub file: String,
    pub sha256: String,
    pub bytes: u64,
    pub retained: Value,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Ledger {
    pub metadata: Value,
    pub stages: Vec<Stage>,
    pub seconds_per_stage: Vec<f64>,
    pub selected_stages: [Option<usize>; 3],
}
pub(crate) fn catalogue(
    reference: &Reference,
    task: &str,
) -> Result<(parent::Configuration, Ledger)> {
    let dir = reference.root.join(name(task)?).join("catalogue");
    let config: parent::Configuration = read(&dir.join("configuration.json"))?;
    let ledger: Ledger = read(&dir.join("ledger.json"))?;
    let primes = quotient::catalogue();
    if ledger.metadata != read::<Value>(&dir.join("metadata.json"))?
        || ledger.metadata["source_fingerprint"] != reference.source_fingerprint
        || ledger.metadata["binary_sha256"] != reference.binary_sha256
        || ledger.metadata["producer_commit"] != reference.producer_commit
        || ledger.metadata["configuration_sha256"] != hash(&serde_json::to_vec(&config)?)
        || ledger.metadata["primes"] != json!(primes)
        || ledger.stages.len() != primes.len()
        || ledger.seconds_per_stage.len() != primes.len()
        || ledger
            .seconds_per_stage
            .iter()
            .any(|s| !s.is_finite() || *s < 0.)
    {
        return Err("source catalogue metadata, stage scope or cost ledger differs".into());
    }
    for (i, (&prime, stage)) in primes.iter().zip(&ledger.stages).enumerate() {
        let file = format!("stage-{i:02}-p{prime}.json");
        if stage.prime != prime
            || stage.file != file
            || digest(&dir.join(&file))? != stage.sha256
            || fs::metadata(dir.join(&file))?.len() != stage.bytes
        {
            return Err("source stage order, bytes or hash changed".into());
        }
    }
    Ok((config, ledger))
}
pub(crate) fn case_record(reference: &Reference, task: &str) -> Result<Value> {
    let root = &reference.root;
    let dir = root.join(name(task)?);
    let run_exit = exit(root, &format!("{task}.run.exit"))?;
    let replay_exit = if run_exit == 0 {
        Some(exit(root, &format!("{task}.replay.exit"))?)
    } else {
        None
    };
    let mut artifacts = vec![];
    tree(&dir, &mut artifacts)?;
    for suffix in ["run.exit", "run.resources", "log"] {
        artifacts.push(evidence(&root.join(format!("{task}.{suffix}")))?);
    }
    if replay_exit.is_some() {
        for suffix in [
            "replay.exit",
            "replay.resources",
            "replay.json",
            "replay.log",
        ] {
            artifacts.push(evidence(&root.join(format!("{task}.{suffix}")))?);
        }
    }
    let complete = run_exit == 0 && replay_exit == Some(0);
    let cost = if complete {
        let (_, ledger) = catalogue(reference, task)?;
        let replay: Value = read(&root.join(format!("{task}.replay.json")))?;
        if replay["task"] != task
            || replay["replayed"] != true
            || replay["source_fingerprint"] != reference.source_fingerprint
            || replay["result_sha256"] != digest(&dir.join("result.json"))?
            || replay["all_actual_models_and_metrics_replayed"] != true
            || replay["all_ce_steps_replayed"] != true
            || replay["catalogue"]["replayed"] != true
            || replay["catalogue"]["all_original_stage_reports_replayed"] != true
            || replay["catalogue"]["stages"] != 55
            || replay["catalogue"]["ledger_sha256"] != digest(&dir.join("catalogue/ledger.json"))?
        {
            return Err("source case lacks matching complete independent replay evidence".into());
        }
        let seconds = ledger.seconds_per_stage.iter().sum::<f64>();
        if !seconds.is_finite() {
            return Err("nonfinite original constructor total".into());
        }
        Some(seconds)
    } else {
        None
    };
    Ok(
        json!({"task":task,"status":if complete {"complete"} else {"source_process_failure"},
        "run_exit":run_exit,"replay_exit":replay_exit,"artifacts":artifacts,
        "source_constructor_seconds":cost,"partial_costs_retained":!complete}),
    )
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Attempt {
    pub request: quotient::binding::Request,
    pub report: Option<quotient::binding::Report>,
    pub returned_error: Option<String>,
}
pub(crate) fn attempt(
    reference: &Reference,
    task: &str,
    config: &parent::Configuration,
    stage: &Stage,
) -> Result<Attempt> {
    let path = reference
        .root
        .join(name(task)?)
        .join("catalogue")
        .join(&stage.file);
    if digest(&path)? != stage.sha256 {
        return Err("source attempt hash changed".into());
    }
    let a: Attempt = read(&path)?;
    let expected = quotient::binding::Request {
        discovery: quotient::Request {
            input: config.input.clone(),
            prime: stage.prime,
        },
        training: config.training.clone(),
        prefix_policy: config.prefix.clone(),
        certificate_policy: config.certificate.clone(),
        readout: config.readout.clone(),
    };
    if a.request != expected {
        return Err("source attempt input/request differs".into());
    }
    match (&a.report, &a.returned_error) {
        (Some(r), None) if r.request == expected && r.discovery.request == expected.discovery => {}
        (None, Some(e)) if !e.is_empty() => {}
        _ => return Err("source attempt must retain one bound report or returned error".into()),
    }
    Ok(a)
}
