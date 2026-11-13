use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Attempt {
    pub source_sha256: String,
    pub report: Option<binding::Report>,
    pub refusal: Option<String>,
    pub seconds: f64,
}
fn construct(old: &source::Attempt, policies: &[training::Policy; 3]) -> Result<binding::Report> {
    let r = old
        .report
        .as_ref()
        .ok_or("source stage returned an error; discovery unavailable")?;
    if r.discovery.code.is_none() {
        return Err("source stage admitted no latent code".into());
    }
    binding::run(&r.discovery, policies.clone())
}
fn check(old: &source::Attempt, policies: &[training::Policy; 3], a: &Attempt) -> Result<()> {
    if !a.seconds.is_finite() || a.seconds < 0. {
        return Err("invalid refined construction time".into());
    }
    match (&a.report, &a.refusal) {
        (Some(report), None) => {
            let old = old
                .report
                .as_ref()
                .ok_or("refined report has no source discovery")?;
            binding::replay(&old.discovery, policies, report)
        }
        (None, Some(reason)) if !reason.is_empty() => {
            if construct(old, policies)
                .err()
                .map(|e| e.to_string())
                .as_ref()
                != Some(reason)
            {
                return Err("refined stage refusal does not replay".into());
            }
            Ok(())
        }
        _ => Err("refined attempt needs one report or refusal".into()),
    }
}
fn context(inv: &Value, task: &str, x: &inputs::Inputs) -> Result<(Reference, Value)> {
    let index = inputs::check(inv, task, x)?;
    let reference: Reference = serde_json::from_value(inv["request"]["discovery"].clone())?;
    source::validate_reference(
        &reference,
        &serde_json::from_value::<Vec<String>>(inv["planned"].clone())?,
        &serde_json::from_value::<Vec<String>>(inv["tasks"].clone())?,
    )?;
    let record = source::case_record(&reference, task)?;
    if record != inv["discoveries"][index] {
        return Err("pinned source case or replay evidence changed".into());
    }
    Ok((reference, record))
}
fn metadata(inv: &Value, input: &Value, source: &Value) -> Value {
    json!({"phase":"training_only","source_fingerprint":fingerprint(),"binary_sha256":inv["binary_sha256"],
        "producer_commit":inv["request"]["producer_commit"],"protocol":inv["protocol"],"input":input,
        "source_case":source,"heldout_evaluation":false,"parent_comparison":false})
}
pub(crate) fn retained(a: &Attempt) -> Result<Value> {
    let endpoints = a.report.as_ref().map(|r| r.endpoints.iter().map(|e| {
        Ok(json!({"refusal":e.refusal,"audit":e.audit,"training":e.training.as_ref().map(|t| -> Result<Value> {
            Ok(json!({"policy":t.policy,"train_ce":t.final_ce,
                "model_sha256":hash(&serde_json::to_vec(&t.model)?),"work":t.work,
                "scalar_work_forecast":t.scalar_work_forecast,"steps":t.steps.len()}))
        }).transpose()?}))
    }).collect::<Result<Vec<_>>>()).transpose()?;
    Ok(
        json!({"source_sha256":a.source_sha256,"refusal":a.refusal,"seconds":a.seconds,
        "endpoints":endpoints,"initialization":a.report.as_ref().map(|r|json!({
            "bounds":r.initialization.bounds,"observed_states":r.initialization.observed_states,
            "default_states":r.initialization.default_states,"unsupported_defaults":r.initialization.unsupported_defaults,
            "complex_products":r.initialization.complex_products,"ce_gradients":r.initialization.ce_gradients})),
        "new_discovery_work":0,"unseen_identification":false}),
    )
}
pub fn run_case(root: &Path, task: &str) -> Result<()> {
    run_with(
        root,
        task,
        &inventory(root)?,
        inputs::build(&suite()?, task)?,
    )
}
pub(crate) fn run_with(root: &Path, task: &str, inv: &Value, x: inputs::Inputs) -> Result<()> {
    let (reference, source) = context(inv, task, &x)?;
    let dir = root.join(name(task)?);
    fs::create_dir(&dir)?;
    write(
        &dir.join("metadata.json"),
        &metadata(inv, &x.record, &source),
    )?;
    write(&dir.join("input.json"), &x.input)?;
    write(&dir.join("policies.json"), &x.policies)?;
    let mut stages = vec![];
    if source["status"] == "complete" {
        let (config, ledger) = source::catalogue(&reference, task)?;
        if config.input != x.input {
            return Err("original source observations differ from native generator".into());
        }
        for (index, old_stage) in ledger.stages.iter().enumerate() {
            let old = source::attempt(&reference, task, &config, old_stage)?;
            let start = std::time::Instant::now();
            let result = construct(&old, &x.policies);
            let seconds = start.elapsed().as_secs_f64();
            let (report, refusal) = match result {
                Ok(r) => (Some(r), None),
                Err(e) => (None, Some(e.to_string())),
            };
            let a = Attempt {
                source_sha256: old_stage.sha256.clone(),
                report,
                refusal,
                seconds,
            };
            let file = format!("stage-{index:02}-p{}.json", old_stage.prime);
            write(&dir.join(&file), &a)?;
            stages.push(json!({"source_stage":old_stage,"source_seconds":ledger.seconds_per_stage[index],
                "file":file,"sha256":digest(&dir.join(&file))?,"bytes":fs::metadata(dir.join(&file))?.len(),
                "retained":retained(&a)?}));
        }
    }
    write(
        &dir.join("result.json"),
        &json!({"metadata":metadata(inv,&x.record,&source),
        "input_sha256":digest(&dir.join("input.json"))?,"policies_sha256":digest(&dir.join("policies.json"))?,
        "source_refusal":if source["status"] == "complete" {None} else {Some("original source case failed")},
        "stages":stages,"heldout_evaluation":false,"parent_comparison":false,"unseen_identification":false}),
    )
}
pub(crate) fn record(
    root: &Path,
    task: &str,
    inv: &Value,
    x: &inputs::Inputs,
    deep: bool,
) -> Result<Value> {
    let (reference, source) = context(inv, task, x)?;
    let dir = root.join(name(task)?);
    let result: Value = read(&dir.join("result.json"))?;
    if result["metadata"] != metadata(inv, &x.record, &source)
        || read::<Value>(&dir.join("metadata.json"))? != result["metadata"]
        || read::<quotient::Input>(&dir.join("input.json"))? != x.input
        || read::<[training::Policy; 3]>(&dir.join("policies.json"))? != x.policies
        || result["input_sha256"] != digest(&dir.join("input.json"))?
        || result["policies_sha256"] != digest(&dir.join("policies.json"))?
        || result["heldout_evaluation"] != false
        || result["parent_comparison"] != false
        || result["unseen_identification"] != false
    {
        return Err("refined result/input/policy/scope differs".into());
    }
    let stages = result["stages"]
        .as_array()
        .ok_or("missing refined stages")?;
    if source["status"] != "complete" {
        if !stages.is_empty() || result["source_refusal"] != "original source case failed" {
            return Err("failed original source silently gained models".into());
        }
        return Ok(result);
    }
    if !result["source_refusal"].is_null() {
        return Err("complete source was discarded".into());
    }
    let (config, ledger) = source::catalogue(&reference, task)?;
    if config.input != x.input || stages.len() != ledger.stages.len() {
        return Err("source data or55-stage coverage differs".into());
    }
    for (i, (saved, old_stage)) in stages.iter().zip(&ledger.stages).enumerate() {
        let file = format!("stage-{i:02}-p{}.json", old_stage.prime);
        if saved["source_stage"] != json!(old_stage)
            || saved["source_seconds"] != ledger.seconds_per_stage[i]
            || saved["file"] != file
            || saved["sha256"] != digest(&dir.join(&file))?
            || saved["bytes"] != fs::metadata(dir.join(&file))?.len()
        {
            return Err("refined stage or source hash/work differs".into());
        }
        let a: Attempt = read(&dir.join(&file))?;
        if a.source_sha256 != old_stage.sha256 || saved["retained"] != retained(&a)? {
            return Err("refined retained actual state differs".into());
        }
        if deep {
            check(
                &source::attempt(&reference, task, &config, old_stage)?,
                &x.policies,
                &a,
            )?;
        }
    }
    Ok(result)
}
pub(crate) fn seal(root: &Path, task: &str) -> Result<Value> {
    Ok(
        json!({"task":task,"replayed":true,"phase":"training_only","source_fingerprint":fingerprint(),
        "result_sha256":digest(&root.join(name(task)?).join("result.json"))?,
        "actual_complex_states_replayed":true,"all_ce_steps_replayed":true,
        "heldout_evaluation":false,"parent_comparison":false,"unseen_identification":false}),
    )
}
pub fn replay_case(root: &Path, task: &str) -> Result<Value> {
    replay_with(
        root,
        task,
        &inventory(root)?,
        inputs::build(&suite()?, task)?,
    )
}
pub(crate) fn replay_with(
    root: &Path,
    task: &str,
    inv: &Value,
    x: inputs::Inputs,
) -> Result<Value> {
    record(root, task, inv, &x, true)?;
    seal(root, task)
}
