//! Seal each actual complex head before fresh original-source and new metrics.
//! Replay rebuilds complete tensors, original records and every metric value.
use super::*;

pub(crate) fn sealed_model(
    root: &Path,
    data: &data::Task,
    entry: &Value,
    batch: usize,
    sealed: &Value,
) -> Result<model::Model> {
    let file = entry["file"].as_str().ok_or("missing sealed source")?;
    if digest(&root.join(file))? != entry["source_model_sha256"] {
        return Err("sealed source changed before scoring".into());
    }
    let model = model::Model::construct(read(&root.join(file))?, data, entry, batch)?;
    if head::digest(&model)? != sealed["model_sha256"] {
        return Err("scored coefficients differ from pre-metric seal".into());
    }
    Ok(model)
}

fn execute(root: &Path, task: &str, replay: bool) -> Result<Value> {
    name(task)?;
    let inv = parent::inventory(root)?;
    let cases: Vec<_> = inv["cases"]
        .as_array()
        .ok_or("missing cases")?
        .iter()
        .filter(|r| r["task"] == task)
        .collect();
    if cases.len() != 1 {
        return Err("missing or repeated original task".into());
    }
    let case = cases[0];
    let request: Request = serde_json::from_value(inv["request"].clone())?;
    let files = parent::manifest(&request)?;
    let path = request.parent_root.join(task).join("result.json");
    parent::pinned(&path, &files)?;
    if digest(&path)? != case["parent_result_sha256"] {
        return Err("parent result changed".into());
    }
    let eligible = case["eligible_models"]
        .as_array()
        .ok_or("missing eligibility inventory")?;
    let dir = root.join(task);
    if !replay {
        fs::create_dir(&dir)?;
    }
    let dataset = if eligible.is_empty() {
        None
    } else {
        let suite: Value = serde_json::from_str(SUITE)?;
        let rows: Vec<_> = suite["tasks"]
            .as_array()
            .ok_or("missing suite")?
            .iter()
            .filter(|r| r["name"] == task)
            .collect();
        if rows.len() != 1 {
            return Err("original generator row missing or repeated".into());
        }
        let batch = rows[0]["case"]["batch"]
            .as_u64()
            .ok_or("missing original minibatch size")? as usize;
        Some((data::build(rows[0])?, batch))
    };
    let mut seals = Vec::new();
    let (mut buffers, mut work, mut updates) = (0, 0, 0);
    let mut checked_containers = BTreeSet::new();
    for (i, entry) in eligible.iter().enumerate() {
        let source_sha = entry["source_model_sha256"]
            .as_str()
            .ok_or("missing source hash")?;
        let file = entry["file"].as_str().ok_or("missing source file")?;
        if file != format!("sources/{source_sha}.json")
            || source_sha.len() != 64
            || !source_sha.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err("invalid extracted source path".into());
        }
        if digest(&root.join(file))? != source_sha {
            return Err("extracted source changed".into());
        }
        let container = entry["container"]
            .as_str()
            .ok_or("missing original container")?;
        if checked_containers.insert(container)
            && digest(Path::new(container))? != entry["container_sha256"]
        {
            return Err("original source container changed".into());
        }
        let (data, batch) = dataset
            .as_ref()
            .ok_or("eligible source without original data")?;
        let model = model::Model::construct(read(&root.join(file))?, data, entry, *batch)?;
        let metric_work = model.forecast(data)?;
        // Only one endpoint is live. Reserve three complete constructions for
        // fresh construction, replay loading, and the post-seal scoring pass.
        buffers = buffers.max(model.buffer_bytes());
        work += 3 * model.construction_work() + metric_work;
        if buffers > CASE_BUFFER_LIMIT || work > CASE_WORK_CAP {
            return Err("complete case exceeds head and metric forecasts".into());
        }
        let saved = dir.join(format!("model-{i}.json"));
        let sha = head::digest(&model)?;
        if replay {
            if digest(&saved)? != sha {
                return Err("saved head differs from fresh actual construction".into());
            }
            let loaded = model::Model::load(read(&saved)?, &model)?;
            if loaded != model {
                return Err("saved head load differs from reconstruction".into());
            }
        } else {
            write(&saved, &model)?;
        }
        seals.push(
            json!({"reference":entry["reference"],"training_mode":entry["training_mode"],
            "source_model_sha256":source_sha,"file":format!("model-{i}.json"),"model_sha256":sha,
            "model_info":model.info(),"metric_scalar_work_forecast":metric_work}),
        );
        updates += model.new_updates();
        // The complete head is on disk before its temporary state is released.
    }
    let seal = json!({"task":task,"source_fingerprint":fingerprint(),"models":seals,
        "parent_result_sha256":case["parent_result_sha256"],"selection_before_metrics":true,
        "new_optimizer_updates":updates,"endpoint_storage":"sequential disk-sealed models",
        "construction_passes_reserved":3,
        "case_buffer_forecast_bytes":buffers,"case_scalar_work_forecast":work});
    if replay {
        if read::<Value>(&dir.join("selection-seal.json"))? != seal {
            return Err("selection seal changed".into());
        }
    } else {
        write(&dir.join("selection-seal.json"), &seal)?;
    }
    let snapshot: Value = if eligible.is_empty() {
        Value::Null
    } else {
        read(&root.join("parent-snapshot.json"))?
    };
    let mut outcomes = Vec::new();
    for (entry, sealed) in eligible.iter().zip(&seals) {
        let reference = entry["reference"].as_str().ok_or("missing reference")?;
        let prior = import::previous(&snapshot, task, reference)?;
        let (data, batch) = dataset.as_ref().ok_or("missing metric data")?;
        let model = sealed_model(root, data, entry, *batch, sealed)?;
        let mut scores = serde_json::Map::new();
        for (split, examples) in &data.splits {
            let source_scores = metrics::evaluate(&model::Parent(&model), examples)?;
            if source_scores != prior["metrics"][split] {
                return Err(
                    format!("fresh source metrics differ from reviewed {task}/{split}").into(),
                );
            }
            let actual = metrics::evaluate(&model, examples)?;
            if raw_accuracy(&actual) != raw_accuracy(&source_scores) {
                return Err("guarded original raw accuracy changed".into());
            }
            scores.insert(split.clone(), actual);
        }
        let scores = Value::Object(scores);
        if (scores["train"]["loss"]
            .as_f64()
            .ok_or("missing actual train CE")?
            - prior["metrics"]["train"]["loss"]
                .as_f64()
                .ok_or("missing prior train CE")?)
        .abs()
            > 1e-10
        {
            return Err("mixture changed the observed ordinary CE".into());
        }
        let (primary, aux) = passes(&scores, &data.selection)?;
        outcomes.push(json!({"task":task,"parent_reference":reference,
            "reference":format!("margin-budget:{reference}"),"training_mode":entry["training_mode"],
            "primary_split":data.selection,"status":"selected","train_ce":scores["train"]["loss"],
            "metrics":scores,"parent_metrics":prior["metrics"],"primary_pass":primary,"all_splits_pass":aux,
            "model_sha256":sealed["model_sha256"],"model_info":sealed["model_info"],
            "metric_scalar_work_forecast":sealed["metric_scalar_work_forecast"],
            "source_metrics_reproduced":true,"raw_decisions_preserved":true,
            "metric_provenance":{"kind":"fresh_changed_model"}}));
    }
    Ok(
        json!({"task":task,"status":"complete","eligible_models":eligible.len(),"outcomes":outcomes,
        "selection_seal_sha256":digest(&dir.join("selection-seal.json"))?,
        "new_optimizer_updates":updates}),
    )
}

pub fn run_case(root: &Path, task: &str) -> Result<()> {
    let result = execute(root, task, false)?;
    write(&root.join(name(task)?).join("result.json"), &result)
}
pub fn replay_case(root: &Path, task: &str) -> Result<Value> {
    let result = execute(root, task, true)?;
    if read::<Value>(&root.join(name(task)?).join("result.json"))? != result {
        return Err("fresh metrics differ from saved result".into());
    }
    Ok(
        json!({"task":task,"status":"replayed","fresh_models":result["eligible_models"],
        "result_sha256":digest(&root.join(task).join("result.json"))?}),
    )
}
