//! Actual original causal histories and saved predecessors. Replaying a
//! history verifies existing work; wrapping makes no new optimizer updates.
use super::*;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Attempt {
    input_refusal: Option<String>,
    report: Option<binding::Report>,
    error: Option<String>,
    construction_seconds: f64,
}
pub(crate) struct Bundle {
    pub rows: Vec<Value>,
    pub models: BTreeMap<String, fallback::Model>,
}
fn attempt(
    pin: &Reference,
    stage: &Value,
    task: &str,
    x: &inputs::Inputs,
) -> Result<Option<(Attempt, String)>> {
    inputs::check(stage, task, x)?;
    let records: Vec<_> = stage["records"]
        .as_array()
        .ok_or("missing staged records")?
        .iter()
        .filter(|r| r["task"] == task)
        .collect();
    if records.len() != 1 {
        return Err("unknown or repeated stage record".into());
    }
    let r = records[0];
    if r["status"] != "complete" {
        return Ok(None);
    }
    let dir = pin.root.join(name(task)?);
    let sha = digest(&dir.join("attempt.json"))?;
    if r["result"]["attempt_sha256"] != sha
        || r["result"]["configuration_sha256"] != digest(&dir.join("configuration.json"))?
        || read::<Option<binding::Request>>(&dir.join("configuration.json"))? != x.configuration
    {
        return Err("actual staged configuration or parameter bytes changed".into());
    }
    let a: Attempt = read(&dir.join("attempt.json"))?;
    if a.input_refusal != x.refusal
        || !a.construction_seconds.is_finite()
        || a.construction_seconds < 0.
    {
        return Err("original input refusal or cost changed".into());
    }
    match (&x.configuration, &a.report, &a.error) {
        (Some(config), Some(report), None) => binding::replay(config, report)?,
        (None, None, None) => {}
        (Some(config), None, Some(error)) => match binding::run(config.clone()) {
            Err(actual) if actual.to_string() == *error => {}
            _ => return Err("staged constructor error does not replay".into()),
        },
        _ => return Err("contradictory or missing staged outcome".into()),
    }
    Ok(Some((a, sha)))
}
pub(crate) fn predecessor(root: &Path, old: &Value) -> Result<Option<(fallback::Parent, Value)>> {
    let saved = &old["saved_endpoint"];
    let (Some(relative), Some(expected)) =
        (saved["model_path"].as_str(), saved["model_sha256"].as_str())
    else {
        return Ok(None);
    };
    let path = parents::checkpoint_path(root, relative)?;
    if !path.exists() {
        return Ok(None);
    }
    if !sha(expected) || digest(&path)? != expected {
        return Err("predecessor parameters differ from selected saved bytes".into());
    }
    let raw: Value = read(&path)?;
    let parent = match fallback::Parent::from_checkpoint(raw) {
        Ok(p) => p,
        Err(_) => return Ok(None),
    };
    Ok(Some((parent, parents::evidence(&path)?)))
}
pub(crate) fn build(
    root: &Path,
    pin: &Reference,
    stage: &Value,
    parent: &Value,
    task: &str,
    x: &inputs::Inputs,
) -> Result<Bundle> {
    let mut rows = vec![];
    let mut models = BTreeMap::new();
    let mut loaded = None;
    let mut buffer_bytes = 0u64;
    for policy in POLICIES {
        for m in MODES {
            let selected = comparison::cell(parent, policy, task, m)?;
            let reference = format!("fallback:{policy}:{task}:{m}");
            let mut row = json!({"initializer":policy,"task":task,"training_mode":m,"prior_reference":selected["reference"],"reference":reference,"status":"unchanged","reason":"selected model is not a causal candidate"});
            if selected["reference"]
                .as_str()
                .is_some_and(|r| r.starts_with("causal-pairs:"))
            {
                let causal = comparison::cell(comparison::causal(parent), policy, task, m)?;
                let previous = comparison::cell(
                    &comparison::causal(parent)["complete_parent"],
                    policy,
                    task,
                    m,
                )?;
                if selected != causal
                    || selected["reference"] != format!("causal-pairs:{task}:{m}")
                    || causal["baseline"] != *previous
                {
                    return Err("selected causal head or exact predecessor ancestry changed".into());
                }
                row["status"] = json!("refused");
                row["reason"] = json!("exact predecessor checkpoint missing or unsupported");
                row["predecessor_reference"] = previous["reference"].clone();
                if loaded.is_none() {
                    loaded = Some(attempt(pin, stage, task, x)?);
                }
                if let Some((a, sha)) = loaded.as_ref().unwrap() {
                    if let Some(report) = &a.report {
                        let endpoints: Vec<_> = report
                            .endpoints
                            .iter()
                            .filter(|e| mode(e.policy.mode) == m)
                            .collect();
                        if endpoints.len() != 1 {
                            return Err("selected staged endpoint is missing or repeated".into());
                        }
                        let endpoint = endpoints[0];
                        let out = endpoint
                            .training
                            .as_ref()
                            .ok_or("selected causal endpoint has no model")?;
                        let audit = endpoint
                            .audit
                            .as_ref()
                            .ok_or("selected causal endpoint lacks audit")?;
                        if endpoint.error.is_some()
                            || selected["saved_endpoint"]["model_sha256"]
                                != fallback::digest(&out.model)?
                            || selected["saved_endpoint"]["report_sha256"] != *sha
                            || selected["train_ce"] != audit.train_ce
                        {
                            return Err("actual causal endpoint differs from selected head".into());
                        }
                        if let Some((prior, evidence)) = predecessor(root, previous)? {
                            let input = &x
                                .configuration
                                .as_ref()
                                .ok_or("selected causal input missing")?
                                .discovery
                                .input;
                            match fallback::Model::new(out.model.clone(), prior, input) {
                                Err(error) => {
                                    row["reason"] = json!(error.to_string());
                                }
                                Ok(model) => {
                                    let projected = buffer_bytes
                                        .checked_add(model.bounds().buffer_bytes)
                                        .ok_or("case buffer forecast overflow")?;
                                    let complete_buffers = projected
                                        .checked_add(report.buffer_preflight_bytes)
                                        .ok_or("retained history buffer forecast overflow")?;
                                    if complete_buffers > 56 * 1024 * 1024 * 1024 {
                                        row["reason"] = json!(
                                            "saved wrappers and retained history exceed56GiB case buffer forecast"
                                        );
                                    } else {
                                        let mut loss = 0.;
                                        for s in &input.observations {
                                            let p: Vec<_> =
                                                s.tokens.iter().map(|&t| t as i64).collect();
                                            if !model.known(&p)? {
                                                return Err(
                                                    "original training prefix was not covered"
                                                        .into(),
                                                );
                                            }
                                            let logits = model.logits(&p)?;
                                            if logits != out.model.logits_tokens(&s.tokens) {
                                                return Err(
                                                    "covered original logits changed".into()
                                                );
                                            }
                                            loss += training::ce(&logits, s.label);
                                        }
                                        loss /= input.observations.len() as f64;
                                        let original = training::score(
                                            &out.model,
                                            &out.model.observations(input)?,
                                        );
                                        if loss != original || (loss - audit.train_ce).abs() > 1e-8
                                        {
                                            return Err("covered raw CE is not identical".into());
                                        }
                                        row["status"] = json!("wrapped");
                                        row["reason"] = Value::Null;
                                        row["train_ce"] = selected["train_ce"].clone();
                                        row["actual_raw_ce"] = json!(loss);
                                        row["covered_records"] = json!(input.observations.len());
                                        row["predecessor_file"] = evidence;
                                        row["binding"] = json!(model.binding());
                                        row["bounds"] = json!(model.bounds());
                                        row["retained_history_buffer_bytes"] =
                                            json!(report.buffer_preflight_bytes);
                                        row["complete_case_buffer_bytes"] = json!(complete_buffers);
                                        row["prediction_limit"] = json!(model.prediction_limit());
                                        row["model_sha256"] = json!(fallback::digest(&model)?);
                                        row["model_file"] = json!(format!("{policy}-{m}.json"));
                                        row["report_sha256"] = json!(sha);
                                        row["new_gradient_updates"] = json!(0);
                                        buffer_bytes = projected;
                                        models.insert(reference, model);
                                    }
                                }
                            }
                        }
                    } else {
                        row["reason"] = json!("selected causal stage has no report");
                    }
                } else {
                    row["reason"] = json!("selected causal stage process failed");
                }
            }
            rows.push(row);
        }
    }
    Ok(Bundle { rows, models })
}
