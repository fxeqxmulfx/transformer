//! Fresh case lifecycle with sealed structural choices, actual tensor files and
//! replayed changed-model metrics. Approved unchanged metrics remain inherited.
use super::*;
fn metadata(inv: &Value, input: &Value, causal: &Value) -> Value {
    json!({"source_fingerprint":fingerprint(),"binary_sha256":inv["binary_sha256"],
      "producer_commit":inv["request"]["producer_commit"],"protocol":inv["protocol"],
      "input":input,"causal_input":causal,"metric_policy":"fresh_changed_models_and_explicit_reviewed_inheritance"})
}
fn manifest(
    request: &Request,
    all: &[String],
    inputs: &[Value],
    causal: &[Value],
) -> Result<Value> {
    if !commit(&request.producer_commit)
        || request.suite_sha256 != hash(SUITE.as_bytes())
        || !request.checkpoint_root.is_absolute()
        || all.len() != 35
        || inputs.len() != 35
        || causal.len() != 35
        || inputs.iter().zip(all).any(|(r, t)| r["task"] != *t)
        || causal.iter().zip(all).any(|(r, t)| r["task"] != *t)
    {
        return Err("invalid original request or complete input universe".into());
    }
    Ok(
        json!({"request":request,"source_fingerprint":fingerprint(),"binary_sha256":digest(&std::env::current_exe()?)?,
      "protocol":protocol()?,"tasks":all,"planned":planned()?,"inputs":inputs,"causal_inputs":causal,
      "choices_per_arm":105,"choose_between_initializers":false,"no_new_optimization":true,
      "unchanged_metrics_are_inherited":true,"changed_metrics_are_fresh":true}),
    )
}
pub fn prepare(root: &Path, request: Request) -> Result<()> {
    if root.exists() {
        return Err("fresh comparison directory required".into());
    }
    let suite = suite()?;
    let all = tasks(&suite)?;
    let planned = planned()?;
    parents::load_all(&request, &all, true)?;
    let inputs = all
        .iter()
        .map(|t| Ok(inputs::build(&suite, t, &planned)?.record))
        .collect::<Result<Vec<_>>>()?;
    let causal = all
        .iter()
        .map(|t| Ok(causal::inputs::build(&suite, t)?.record))
        .collect::<Result<Vec<_>>>()?;
    let inventory = manifest(&request, &all, &inputs, &causal)?;
    fs::create_dir(root)?;
    write(&root.join("request.json"), &request)?;
    write(&root.join("inventory.json"), &inventory)
}
pub(crate) fn inventory(
    root: &Path,
    complete_files: bool,
) -> Result<(Value, Request, BTreeMap<String, parents::Reviewed>)> {
    let request: Request = read(&root.join("request.json"))?;
    let inv: Value = read(&root.join("inventory.json"))?;
    let all = tasks(&suite()?)?;
    if inv
        != manifest(
            &request,
            &all,
            inv["inputs"].as_array().ok_or("original inputs omitted")?,
            inv["causal_inputs"]
                .as_array()
                .ok_or("causal inputs omitted")?,
        )?
    {
        return Err("comparison inventory, executable or scope changed".into());
    }
    let reviewed = parents::load_all(&request, &all, complete_files)?;
    Ok((inv, request, reviewed))
}
fn case_inputs(inv: &Value, task: &str) -> Result<(inputs::Inputs, causal::inputs::Inputs)> {
    let suite = suite()?;
    let x = inputs::build(&suite, task, &planned()?)?;
    inputs::check(inv, task, &x)?;
    let causal = causal::inputs::build(&suite, task)?;
    let rows = inv["causal_inputs"]
        .as_array()
        .ok_or("causal inputs omitted")?
        .iter()
        .filter(|r| r["task"] == task)
        .collect::<Vec<_>>();
    if rows.len() != 1 || *rows[0] != causal.record {
        return Err("causal original input changed".into());
    }
    Ok((x, causal))
}
fn tensor_values(bundle: &selected::Bundle) -> Result<BTreeMap<String, Value>> {
    let mut out = BTreeMap::new();
    for (key, model) in &bundle.models {
        out.insert(key.clone(), serde_json::to_value(model)?);
    }
    for (key, model) in &bundle.causal.models {
        if out
            .insert(key.clone(), serde_json::to_value(model)?)
            .is_some()
        {
            return Err("duplicate actual model reference".into());
        }
    }
    Ok(out)
}
fn sealed(
    dir: &Path,
    inv: &Value,
    task: &str,
    x: &inputs::Inputs,
    c: &causal::inputs::Inputs,
    bundle: &selected::Bundle,
) -> Result<Value> {
    let models = tensor_values(bundle)?
        .into_iter()
        .map(|(reference, state)| -> Result<Value> {
            let sha = hash(&serde_json::to_vec(&state)?);
            Ok(json!({"reference":reference,"file":format!("model-{sha}.json"),"model_sha256":sha}))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"task":task,"metadata":metadata(inv,&x.record,&c.record),"choices":bundle.choices,
      "models":models,"raw_training_case":bundle.raw_record,"shared_training_stages":bundle.shared_records,
      "causal_imports":bundle.causal.rows,"case_buffer_forecast":bundle.case_buffer_forecast,
      "selection_sha256":digest(&dir.join("selection.json"))?,"metrics_started":false,"no_new_optimization":true}),
    )
}
fn evaluate(
    task: &str,
    x: &inputs::Inputs,
    c: &causal::inputs::Inputs,
    bundle: &selected::Bundle,
) -> Result<Vec<Value>> {
    // Refuse an over-budget selected case before any endpoint's first score.
    for model in bundle.models.values() {
        outcomes::forecast(&x.task, model)?;
    }
    for model in bundle.causal.models.values() {
        causal::outcomes::forecast(&c.task, model)?;
    }
    let mut rows = outcomes::evaluate(task, x, bundle)?;
    for mut row in causal::outcomes::evaluate(c, &bundle.causal)? {
        if row["status"] != "evaluated" {
            return Err("changed causal endpoint was not evaluated".into());
        }
        row["metric_provenance"] = json!("fresh_changed_model");
        rows.push(row);
    }
    Ok(rows)
}
pub(crate) struct Context<'a> {
    pub inv: &'a Value,
    pub request: &'a Request,
    pub reviewed: &'a BTreeMap<String, parents::Reviewed>,
    pub all: &'a [String],
}
pub fn run_case(root: &Path, task: &str) -> Result<()> {
    name(task)?;
    let (inv, request, reviewed) = inventory(root, false)?;
    let (x, c) = case_inputs(&inv, task)?;
    let all = tasks(&suite()?)?;
    run_with(
        root,
        task,
        &Context {
            inv: &inv,
            request: &request,
            reviewed: &reviewed,
            all: &all,
        },
        &x,
        &c,
    )
}
pub(crate) fn run_with(
    root: &Path,
    task: &str,
    context: &Context<'_>,
    x: &inputs::Inputs,
    c: &causal::inputs::Inputs,
) -> Result<()> {
    name(task)?;
    let inv = context.inv;
    let dir = root.join(task);
    fs::create_dir(&dir)?;
    write(
        &dir.join("metadata.json"),
        &metadata(inv, &x.record, &c.record),
    )?;
    let start = std::time::Instant::now();
    let bundle = selected::build(context.request, context.reviewed, task, x, c, context.all)?;
    write(&dir.join("selection.json"), &bundle.choices)?;
    for (_, state) in tensor_values(&bundle)? {
        let sha = hash(&serde_json::to_vec(&state)?);
        let path = dir.join(format!("model-{sha}.json"));
        if !path.exists() {
            write(&path, &state)?;
        }
    }
    let seal = sealed(&dir, inv, task, x, c, &bundle)?;
    write(&dir.join("selection-seal.json"), &seal)?;
    let verification_seconds = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let outcomes = evaluate(task, x, c, &bundle)?;
    let metric_seconds = start.elapsed().as_secs_f64();
    write(
        &dir.join("result.json"),
        &json!({"task":task,"metadata":metadata(inv,&x.record,&c.record),
      "selection_seal_sha256":digest(&dir.join("selection-seal.json"))?,"seal":seal,"outcomes":outcomes,
      "verification_seconds":verification_seconds,"metric_seconds":metric_seconds,
      "no_new_optimization":true,"unchanged_metrics_are_inherited":true,"changed_metrics_are_fresh":true}),
    )
}
pub(crate) fn retained(root: &Path, task: &str, inv: &Value) -> Result<Value> {
    let (x, c) = case_inputs(inv, task)?;
    retained_with(root, task, inv, &x, &c)
}
pub(crate) fn retained_with(
    root: &Path,
    task: &str,
    inv: &Value,
    x: &inputs::Inputs,
    c: &causal::inputs::Inputs,
) -> Result<Value> {
    let dir = root.join(name(task)?);
    let result: Value = read(&dir.join("result.json"))?;
    let seal: Value = read(&dir.join("selection-seal.json"))?;
    if result["task"] != task
        || result["metadata"] != metadata(inv, &x.record, &c.record)
        || read::<Value>(&dir.join("metadata.json"))? != result["metadata"]
        || result["seal"] != seal
        || seal["metadata"] != result["metadata"]
        || result["selection_seal_sha256"] != digest(&dir.join("selection-seal.json"))?
        || seal["selection_sha256"] != digest(&dir.join("selection.json"))?
        || read::<Value>(&dir.join("selection.json"))? != seal["choices"]
        || seal["metrics_started"] != false
        || result["no_new_optimization"] != true
        || result["unchanged_metrics_are_inherited"] != true
        || result["changed_metrics_are_fresh"] != true
        || ["verification_seconds", "metric_seconds"]
            .iter()
            .any(|k| result[k].as_f64().is_none_or(|x| !x.is_finite() || x < 0.))
    {
        return Err("saved actual case or pre-metric selection seal changed".into());
    }
    let mut seen = BTreeSet::new();
    for model in seal["models"]
        .as_array()
        .ok_or("actual saved model inventory omitted")?
    {
        let sha = model["model_sha256"]
            .as_str()
            .filter(|s| crate::sha(s))
            .ok_or("invalid model digest")?;
        let file = format!("model-{sha}.json");
        if model["file"] != file
            || !seen.insert(
                model["reference"]
                    .as_str()
                    .ok_or("model reference omitted")?,
            )
        {
            return Err("noncanonical or duplicate saved model".into());
        }
        let state: Value = read(&dir.join(file))?;
        if hash(&serde_json::to_vec(&state)?) != sha {
            return Err("actual saved tensors changed".into());
        }
    }
    Ok(result)
}
pub(crate) fn replay_seal(root: &Path, task: &str) -> Result<Value> {
    Ok(
        json!({"task":task,"replayed":true,"source_fingerprint":fingerprint(),
      "result_sha256":digest(&root.join(name(task)?).join("result.json"))?,
      "selected_histories_and_tensors_replayed":true,"changed_model_metrics_replayed":true,
      "unchanged_metrics_provenance":"approved_immutable_parent","primary_rule":"original_sequence_accuracy_at_least_0.99"}),
    )
}
pub fn replay_case(root: &Path, task: &str) -> Result<Value> {
    let (inv, request, reviewed) = inventory(root, false)?;
    let (x, c) = case_inputs(&inv, task)?;
    let all = tasks(&suite()?)?;
    replay_with(
        root,
        task,
        &Context {
            inv: &inv,
            request: &request,
            reviewed: &reviewed,
            all: &all,
        },
        &x,
        &c,
    )
}
pub(crate) fn replay_with(
    root: &Path,
    task: &str,
    context: &Context<'_>,
    x: &inputs::Inputs,
    c: &causal::inputs::Inputs,
) -> Result<Value> {
    let saved = retained_with(root, task, context.inv, x, c)?;
    let bundle = selected::build(context.request, context.reviewed, task, x, c, context.all)?;
    let actual = sealed(&root.join(task), context.inv, task, x, c, &bundle)?;
    if saved["seal"] != actual {
        return Err("fresh selected sources, histories, tensors or choices differ".into());
    }
    for (_, state) in tensor_values(&bundle)? {
        let sha = hash(&serde_json::to_vec(&state)?);
        if read::<Value>(&root.join(task).join(format!("model-{sha}.json")))? != state {
            return Err("saved model differs from freshly bound actual source states".into());
        }
    }
    if saved["outcomes"] != json!(evaluate(task, x, c, &bundle)?) {
        return Err("fresh changed-model original metrics differ".into());
    }
    replay_seal(root, task)
}
