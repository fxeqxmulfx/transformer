//! Construct and seal every selected context map before original metrics.
//! Replays reconstruct actual tensors from the pinned predecessor and compare
//! both new saved model bytes and complete original metric values.
use super::*;

fn entry<'a>(inv: &'a Value, task: &str) -> Result<&'a Value> {
    name(task)?;
    let rows: Vec<_> = inv["cases"]
        .as_array()
        .ok_or("missing prepared cases")?
        .iter()
        .filter(|r| r["task"] == task)
        .collect();
    if rows.len() != 1 {
        return Err("task absent or repeated in original scope".into());
    }
    Ok(rows[0])
}

fn execute(root: &Path, task: &str, replay: bool) -> Result<Value> {
    let inv = parent::inventory(root)?;
    let case = entry(&inv, task)?;
    let request: Request = serde_json::from_value(inv["request"].clone())?;
    let files = parent::manifest(&request)?;
    let original_path = request.parent_root.join(task).join("result.json");
    parent::pinned(&original_path, &files)?;
    if digest(&original_path)? != case["parent_result_sha256"] {
        return Err("original case binding changed".into());
    }
    let original: Value = read(&original_path)?;
    let eligible = case["eligible_models"]
        .as_array()
        .ok_or("missing eligibility inventory")?;
    let dir = root.join(task);
    if !replay {
        fs::create_dir(&dir)?;
    }
    let mut models = Vec::new();
    let mut seals = Vec::new();
    let mut case_buffer_bytes = 0u64;
    let mut case_metric_work = 0u64;
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
            return Err("original generator row missing/repeated".into());
        }
        Some(data::build(rows[0])?)
    };
    for (i, artifact) in eligible.iter().enumerate() {
        let reference = artifact["reference"]
            .as_str()
            .ok_or("missing original model reference")?;
        let path = request
            .parent_root
            .join(task)
            .join(artifact["file"].as_str().ok_or("missing model file")?);
        parent::pinned(&path, &files)?;
        if digest(&path)? != artifact["model_sha256"] {
            return Err("original model tensor bytes changed".into());
        }
        let data = dataset
            .as_ref()
            .ok_or("eligible model without original data")?;
        let model = model::Model::from_original(read(&path)?, data)?;
        let work = model.forecast(data)?;
        case_buffer_bytes += model.buffer_bytes();
        case_metric_work += work;
        if case_buffer_bytes > 56 * 1024 * 1024 * 1024
            || case_metric_work > kernel::limits::WORK_CAP
        {
            return Err("complete changed-model case exceeds memory/work forecast".into());
        }
        let saved = dir.join(format!("model-{i}.json"));
        let hash = context::digest(&model)?;
        if replay {
            if digest(&saved)? != hash {
                return Err("new saved context model differs from fresh construction".into());
            }
        } else {
            write(&saved, &model)?;
        }
        seals.push(json!({"reference":reference,"original_model_sha256":artifact["model_sha256"],
            "file":format!("model-{i}.json"),"model_sha256":hash,"model_info":model.info(),"metric_scalar_work_forecast":work}));
        models.push((reference.to_owned(), model));
    }
    let seal = json!({"task":task,"source_fingerprint":fingerprint(),"parent_result_sha256":case["parent_result_sha256"],
        "models":seals,"selection_before_metrics":true,"no_new_optimization":true,
        "case_buffer_forecast_bytes":case_buffer_bytes,"case_metric_work_forecast":case_metric_work});
    if replay {
        if read::<Value>(&dir.join("selection-seal.json"))? != seal {
            return Err("context selection seal changed".into());
        }
    } else {
        write(&dir.join("selection-seal.json"), &seal)?;
    }
    let mut outcomes = Vec::new();
    for ((reference, model), sealed) in models.iter().zip(&seals) {
        let data = dataset.as_ref().ok_or("missing metric data")?;
        let previous: Vec<_> = original["outcomes"]
            .as_array()
            .ok_or("missing original outcomes")?
            .iter()
            .filter(|r| r["reference"] == *reference)
            .collect();
        if previous.len() != 1 {
            return Err("original metric reference missing or repeated".into());
        }
        let prior = previous[0];
        let mut scores = serde_json::Map::new();
        for (split, examples) in &data.splits {
            scores.insert(split.clone(), metrics::evaluate(model, examples)?);
        }
        let scores = Value::Object(scores);
        let ce = scores["train"]["loss"]
            .as_f64()
            .ok_or("missing original raw training CE")?;
        let old_ce = prior["metrics"]["train"]["loss"]
            .as_f64()
            .ok_or("missing prior training CE")?;
        if (ce - old_ce).abs() > 1e-10 {
            return Err("context map changed original training CE".into());
        }
        let (primary, aux) = passes(&scores, &data.selection)?;
        outcomes.push(json!({"task":task,"parent_reference":reference,"reference":format!("context:{reference}"),
            "training_mode":prior["training_mode"],"primary_split":data.selection,"status":"selected",
            "metrics":scores,"parent_metrics":prior["metrics"],"primary_pass":primary,"all_splits_pass":aux,
            "train_ce":ce,"model_sha256":sealed["model_sha256"],"model_info":sealed["model_info"],
            "metric_scalar_work_forecast":sealed["metric_scalar_work_forecast"],"metric_provenance":{"kind":"fresh_changed_model"}}));
    }
    Ok(
        json!({"task":task,"status":"complete","eligible_models":eligible.len(),"outcomes":outcomes,
        "selection_seal_sha256":digest(&dir.join("selection-seal.json"))?,"no_new_optimization":true}),
    )
}

pub fn run_case(root: &Path, task: &str) -> Result<()> {
    let result = execute(root, task, false)?;
    write(&root.join(name(task)?).join("result.json"), &result)
}
pub fn replay_case(root: &Path, task: &str) -> Result<Value> {
    let result = execute(root, task, true)?;
    if read::<Value>(&root.join(name(task)?).join("result.json"))? != result {
        return Err("fresh context metrics differ from saved result".into());
    }
    Ok(
        json!({"task":task,"status":"replayed","fresh_models":result["eligible_models"],
        "result_sha256":digest(&root.join(task).join("result.json"))?}),
    )
}
