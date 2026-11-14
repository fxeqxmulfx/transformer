//! Ports readout_kernel/cohort stage.rs at 5d4eee84. Retain all augmented
//! complete-basis outcomes, role alignments and raw expansions. Original
//! records, actual complex CE states and mode histories are independently bound.
use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Attempt {
    pub discovery: Option<discovery::Report>,
    pub discovery_error: Option<String>,
    pub head: Option<binding::Report>,
    pub head_error: Option<String>,
    pub discovery_seconds: f64,
    pub head_seconds: f64,
}
fn attempt(
    request: discovery::Request,
    records: &[records::Record],
    policies: &[training::Policy; 3],
) -> Attempt {
    let start = std::time::Instant::now();
    let discovery = discovery::run(request);
    let discovery_seconds = start.elapsed().as_secs_f64();
    match discovery {
        Err(e) => Attempt {
            discovery: None,
            discovery_error: Some(e.to_string()),
            head: None,
            head_error: None,
            discovery_seconds,
            head_seconds: 0.,
        },
        Ok(d) => {
            let start = std::time::Instant::now();
            let head = binding::run(&d, records, policies.clone());
            let head_seconds = start.elapsed().as_secs_f64();
            let (head, head_error) = match head {
                Ok(h) => (Some(h), None),
                Err(e) => (None, Some(e.to_string())),
            };
            Attempt {
                discovery: Some(d),
                discovery_error: None,
                head,
                head_error,
                discovery_seconds,
                head_seconds,
            }
        }
    }
}
fn check(
    request: &discovery::Request,
    records: &[records::Record],
    policies: &[training::Policy; 3],
    a: &Attempt,
) -> Result<()> {
    if [a.discovery_seconds, a.head_seconds]
        .iter()
        .any(|s| !s.is_finite() || *s < 0.)
    {
        return Err("invalid kernel stage duration".into());
    }
    match (&a.discovery, &a.discovery_error) {
        (Some(d), None) => {
            discovery::replay(request, d)?;
            match (&a.head, &a.head_error) {
                (Some(h), None) => binding::replay(d, records, policies, h),
                (None, Some(reason)) if !reason.is_empty() => {
                    if binding::run(d, records, policies.clone())
                        .err()
                        .map(|e| e.to_string())
                        .as_ref()
                        != Some(reason)
                    {
                        return Err("kernel head error does not replay".into());
                    }
                    Ok(())
                }
                _ => Err("kernel stage needs exactly one head outcome".into()),
            }
        }
        (None, Some(reason)) if !reason.is_empty() => {
            if a.head.is_some()
                || a.head_error.is_some()
                || a.head_seconds != 0.
                || discovery::run(request.clone())
                    .err()
                    .map(|e| e.to_string())
                    .as_ref()
                    != Some(reason)
            {
                return Err("kernel discovery error or absent head does not replay".into());
            }
            Ok(())
        }
        _ => Err("kernel stage needs exactly one discovery outcome".into()),
    }
}
pub(crate) fn retained(a: &Attempt) -> Result<Value> {
    let endpoints = a.head.as_ref().map(|h| h.endpoints.iter().map(|e| {
        Ok(json!({"refusal":e.refusal,"audit":e.audit,"training":e.training.as_ref().map(|t| -> Result<Value> {
            Ok(json!({"policy":t.policy,"train_ce":t.final_ce,
                "model_sha256":hash(&serde_json::to_vec(&t.model)?),"work":t.work,
                "scalar_work_forecast":t.scalar_work_forecast,"steps":t.steps.len()}))
        }).transpose()?}))
    }).collect::<Result<Vec<_>>>()).transpose()?;
    Ok(
        json!({"discovery_error":a.discovery_error,"head_error":a.head_error,
        "discovery_seconds":a.discovery_seconds,"head_seconds":a.head_seconds,
        "discovery":a.discovery.as_ref().map(|d|json!({"reason":d.reason,"work":d.work,
            "dimension":d.raw_code.as_ref().map(|c|c.dimension),
            "restricted":{"dimension":d.grouped_code.as_ref().map(|c|c.dimension),
                "aligned_tokens":d.alignments.len(),"matrix_shape":d.matrix_shape,"basis_audit":d.basis_audit,
                "buffer_preflight_bytes":d.buffer_preflight_bytes}})),
        "record_binding":a.head.as_ref().map(|h|json!({"records_sha256":h.records_sha256,
            "original_records":h.original_records,"buffer_preflight_bytes":h.buffer_preflight_bytes})),
        "initialization_refusal":a.head.as_ref().and_then(|h|h.initialization_refusal.as_ref()),
        "initialization":a.head.as_ref().and_then(|h|h.initialization.as_ref()).map(|i|json!({
            "bounds":i.bounds,"observed_states":i.observed_states,"default_states":i.default_states,
            "unmatched_states":i.unmatched_states,"unknown_axis_cells":i.unknown_axis_cells,
            "unmatched_active_cells":i.model.encoding.unmatched_active_cells,
            "complex_products":i.complex_products,"vector_additions":i.vector_additions,"ce_gradients":i.ce_gradients})),
        "endpoints":endpoints,"raw_label_identification":false,"target_readout_compatibility_proved":false,
        "shared_role_target_proved":false}),
    )
}
fn metadata(inv: &Value, input: &Value) -> Value {
    json!({"phase":"training_only","source_fingerprint":fingerprint(),"binary_sha256":inv["binary_sha256"],
        "producer_commit":inv["request"]["producer_commit"],"protocol":protocol().expect("compiled protocol"),
        "input":input,"primes":catalogue(),"heldout_evaluation":false,"parent_comparison":false,
        "declared_readout_used_in_discovery":true,"original_raw_ce_unchanged":true,
        "shared_role_constraints_used":true,"shared_role_target_proved":false,
        "original_record_multiplicity_retained":true})
}
fn filename(index: usize, prime: u16) -> String {
    format!("stage-{index:02}-p{prime}.json")
}
fn ledger(dir: &Path, index: usize, prime: u16, a: &Attempt) -> Result<Value> {
    let file = filename(index, prime);
    Ok(
        json!({"prime":prime,"file":file,"sha256":digest(&dir.join(&file))?,
        "bytes":fs::metadata(dir.join(&file))?.len(),"retained":retained(a)?}),
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
    inputs::check(inv, task, &x)?;
    let dir = root.join(name(task)?);
    fs::create_dir(&dir)?;
    write(&dir.join("metadata.json"), &metadata(inv, &x.record))?;
    write(&dir.join("input.json"), &x.input)?;
    write(&dir.join("records.json"), &x.records)?;
    write(&dir.join("readout.json"), &x.partition())?;
    write(&dir.join("policies.json"), &x.policies)?;
    let mut stages = vec![];
    for (index, prime) in catalogue().into_iter().enumerate() {
        evidence::reserve(&dir)?;
        let a = attempt(x.request(prime), &x.records, &x.policies);
        write(&dir.join(filename(index, prime)), &a)?;
        stages.push(ledger(&dir, index, prime, &a)?);
    }
    write(
        &dir.join("result.json"),
        &json!({"metadata":metadata(inv,&x.record),
        "input_sha256":digest(&dir.join("input.json"))?,"policies_sha256":digest(&dir.join("policies.json"))?,
        "readout_sha256":digest(&dir.join("readout.json"))?,
        "records_sha256":digest(&dir.join("records.json"))?,
        "stages":stages,"heldout_evaluation":false,"parent_comparison":false,
        "raw_label_identification":false,"target_readout_compatibility_proved":false,
        "shared_role_target_proved":false}),
    )
}
pub(crate) fn record(
    root: &Path,
    task: &str,
    inv: &Value,
    x: &inputs::Inputs,
    deep: bool,
) -> Result<Value> {
    inputs::check(inv, task, x)?;
    let dir = root.join(name(task)?);
    let result: Value = read(&dir.join("result.json"))?;
    if result["metadata"] != metadata(inv, &x.record)
        || read::<Value>(&dir.join("metadata.json"))? != result["metadata"]
        || read::<kernel::discovery::Input>(&dir.join("input.json"))? != x.input
        || read::<Vec<records::Record>>(&dir.join("records.json"))? != x.records
        || read::<Value>(&dir.join("readout.json"))? != x.partition()
        || read::<[training::Policy; 3]>(&dir.join("policies.json"))? != x.policies
        || result["input_sha256"] != digest(&dir.join("input.json"))?
        || result["policies_sha256"] != digest(&dir.join("policies.json"))?
        || result["readout_sha256"] != digest(&dir.join("readout.json"))?
        || result["records_sha256"] != digest(&dir.join("records.json"))?
        || result["heldout_evaluation"] != false
        || result["parent_comparison"] != false
        || result["raw_label_identification"] != false
        || result["target_readout_compatibility_proved"] != false
        || result["shared_role_target_proved"] != false
    {
        return Err("kernel original input, policy or scope differs".into());
    }
    let stages = result["stages"]
        .as_array()
        .ok_or("missing kernel stage ledger")?;
    if stages.len() != catalogue().len() {
        return Err("kernel stage coverage differs".into());
    }
    for ((index, prime), saved) in catalogue().into_iter().enumerate().zip(stages) {
        let a: Attempt = read(&dir.join(filename(index, prime)))?;
        if *saved != ledger(&dir, index, prime, &a)? {
            return Err("kernel retained state or file binding changed".into());
        }
        if deep {
            check(&x.request(prime), &x.records, &x.policies, &a)?;
        }
    }
    Ok(result)
}
pub(crate) fn seal(root: &Path, task: &str) -> Result<Value> {
    Ok(
        json!({"task":task,"replayed":true,"phase":"training_only","source_fingerprint":fingerprint(),
        "result_sha256":digest(&root.join(name(task)?).join("result.json"))?,
        "all_complete_restricted_bases_replayed":true,"raw_expansion_and_readout_replayed":true,
        "original_records_replayed":true,
        "actual_complex_states_replayed":true,"all_ce_steps_replayed":true,
        "heldout_evaluation":false,"parent_comparison":false,"raw_label_identification":false,
        "target_readout_compatibility_proved":false,"shared_role_target_proved":false}),
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
