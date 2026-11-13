//! Entirely synthetic artifact trees; these never read original run outputs.
use crate::*;
use import::{Parent, Review};
use std::sync::atomic::{AtomicUsize, Ordering};
static SERIAL: AtomicUsize = AtomicUsize::new(0);
pub struct Temp(pub PathBuf);
impl Temp {
    pub fn new() -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "paths-compare-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
pub fn metrics(ce: f64, accuracy: f64) -> Value {
    json!({"train":{"loss":ce,"sequence_accuracy":1.},"validation":{"loss":2.,"sequence_accuracy":accuracy}})
}
pub fn suite() -> Result<Value> {
    let mut suite: Value = serde_json::from_str(SUITE)?;
    for t in suite["tasks"].as_array_mut().ok_or("missing tasks")? {
        t["data_fingerprints"] = json!({"train":hash(format!("synthetic-train-{}",t["name"]).as_bytes()),
            "validation":hash(format!("synthetic-validation-{}",t["name"]).as_bytes())});
    }
    Ok(suite)
}
pub fn bytes(path: &Path, text: &[u8]) -> Result<String> {
    fs::write(path, text)?;
    digest(path)
}
fn save(root: &Path, name: &str, value: &Value) -> Result<String> {
    let path = root.join(name);
    write(&path, value)?;
    digest(&path)
}
fn header(name: &str) -> Result<Value> {
    Ok(protocol()?["sources"][name].clone())
}
pub fn tree() -> Result<(Temp, Value, Request)> {
    let temp = Temp::new()?;
    let suite = suite()?;
    let mut request = Request {
        producer_commit: "1".repeat(40),
        suite_sha256: hash(&serde_json::to_vec(&suite)?),
        parents: BTreeMap::new(),
    };
    for name in [
        "consequences",
        "depth_original",
        "depth_retained",
        "transformer",
    ] {
        let root = temp.0.join(name);
        fs::create_dir(&root)?;
        let (inventory, summary) = match name {
            "consequences" => consequences(&root, &suite)?,
            "depth_original" | "depth_retained" => depth(&root, &suite, name == "depth_retained")?,
            _ => transformer(&root, &suite)?,
        };
        save(&root, "inventory.json", &inventory)?;
        save(&root, "summary.json", &summary)?;
        let review = root.join("review");
        fs::create_dir(&review)?;
        fs::copy(root.join("summary.json"), review.join("summary.json"))?;
        bytes(&review.join("process.exit"), b"0\n")?;
        bytes(&review.join("reviewed.commit"), "2".repeat(40).as_bytes())?;
        request.parents.insert(
            name.into(),
            Parent {
                summary_sha256: digest(&root.join("summary.json"))?,
                inventory_sha256: digest(&root.join("inventory.json"))?,
                root,
                review: Review {
                    summary: review.join("summary.json"),
                    exit: review.join("process.exit"),
                    commit: review.join("reviewed.commit"),
                },
            },
        );
    }
    Ok((temp, suite, request))
}
fn consequences(root: &Path, suite: &Value) -> Result<(Value, Value)> {
    let mut inv = header("consequences")?;
    let parent = json!({"fixture_only":true,"earlier_failures":["kept"],"all_prior_costs":47});
    inv["parent_sha256"] = json!(save(root, "parent.json", &parent)?);
    let mut records = vec![];
    let names: Vec<_> = rows(suite, "tasks")?
        .iter()
        .take(23)
        .map(|r| r["name"].clone())
        .collect();
    inv["tasks"] = json!(names);
    for (index, name) in names.iter().enumerate() {
        let name = name.as_str().ok_or("fixture task")?;
        let dir = root.join(name);
        fs::create_dir(&dir)?;
        if index == 0 {
            bytes(&root.join(format!("{name}.run.exit")), b"9\n")?;
            let log = bytes(
                &root.join(format!("{name}.log")),
                b"synthetic failed constructor\n",
            )?;
            bytes(&dir.join("partial.bin"), b"preserve partial outcome")?;
            records.push(
                json!({"task":name,"status":"process_failure","run_exit":9,"replay_exit":null,
                "prior_choices_retained":3,"run_log_sha256":log}),
            );
            continue;
        }
        bytes(&root.join(format!("{name}.run.exit")), b"0\n")?;
        bytes(&root.join(format!("{name}.replay.exit")), b"0\n")?;
        let input = json!({"task":name,"data_fingerprints":task(suite,name)?["data_fingerprints"]});
        let mut metadata = header("consequences")?;
        metadata["input"] = input.clone();
        save(&dir, "metadata.json", &metadata)?;
        let model = json!({"synthetic_model":name});
        let bank = json!({"candidates":[{"model_sha256":hash(&serde_json::to_vec(&model)?),"value":{"model":model}}]});
        let selection = json!({"input_sha256":save(&dir,"input.json",&input)?,
            "parents_sha256":save(&dir,"parents.json",&json!(["complete synthetic parents"]))?,
            "bank_sha256":save(&dir,"bank.json",&bank)?,"views":[]});
        let selection_hash = save(&dir, "selection.json", &selection)?;
        let result = json!({"metadata":metadata,"selection":selection,"ce_gradients":0});
        let result_hash = save(&dir, "result.json", &result)?;
        let replay = json!({"task":name,"replayed":true,"all_view_reports_replayed":true,"raw_latent_ce_metrics_audit":true,
            "source_fingerprint":inv["source_fingerprint"],"selection_sha256":selection_hash,"result_sha256":result_hash});
        save(root, &format!("{name}.replay.json"), &replay)?;
        records.push(json!({"task":name,"status":"complete","result":result,"seal":replay}));
    }
    let mut choices = vec![];
    for t in rows(suite, "tasks")? {
        for mode in MODES {
            let no_model = t["name"] == "gpt-probe-branch-amplitudes" && mode == "forward_only";
            choices.push(json!({"task":t["name"],"training_mode":mode,"selection":"validation",
            "status":if no_model {"no_model"} else {"selected"},
            "selected":if no_model {Value::Null} else {json!({"train_ce":1.,"metrics":metrics(1.,0.5),"fixture_only":true})}}));
        }
    }
    let mut summary = header("consequences")?;
    for (key,value) in json!({"full35_result":true,"choices_compared":105,"tasks":23,"records":records,"choices":choices,
        "process_failures":1,"complete_parent":parent,"constructor_accounting":{"completed_seconds":12.,"charged_once":true}}).as_object().unwrap() {
        summary[key] = value.clone();
    }
    Ok((inv, summary))
}
fn depth(root: &Path, suite: &Value, retained: bool) -> Result<(Value, Value)> {
    let name = if retained {
        "depth_retained"
    } else {
        "depth_original"
    };
    let mut inv = header(name)?;
    let cases = crate::depth::expected(retained)?;
    let mut choices = vec![];
    for (index, case) in cases.iter().enumerate() {
        let mut id = format!(
            "{}--{}--{}--{}--d{}",
            text(case, "mode")?,
            text(case, "task")?,
            text(case, "architecture")?,
            text(case, "transport")?,
            case["depth"]
        );
        if retained {
            id.push_str(&format!(
                "--retained-{}",
                text(case, "retained_initializer")?
            ));
        }
        let dir = root.join(&id);
        fs::create_dir(&dir)?;
        let mut metadata = header(name)?;
        metadata["case"] = case.clone();
        metadata["data_fingerprints"] =
            task(suite, text(case, "task")?)?["data_fingerprints"].clone();
        save(&dir, "metadata.json", &metadata)?;
        let mut result = header(name)?;
        result["case"] = case.clone();
        let mut replay = json!({"case":case,"source_fingerprint":inv["source_fingerprint"],"status":"replay_pass"});
        if !retained && index < 18 {
            result["status"] = json!("ineligible_memory_budget");
            result["heldout_scored"] = json!(false);
            replay["memory_rejection"] = json!(true);
        } else {
            result["status"] = json!("complete");
            result["train_ce"] = json!(0.25);
            result["metrics"] = metrics(0.25, 1.);
            result["primary_split"] = json!("validation");
            result["primary_pass"] = json!(true);
            result["all_splits_pass"] = json!(true);
            result["timing_seconds"] = json!({"total":1.});
            for key in [
                "gradient_updates",
                "optimizer_backward_examples",
                "optimizer_backward_prefixes",
                "diagnostic_backward_prefixes",
                "line_search_forward_prefixes",
            ] {
                result[key] = json!(0);
            }
            result["model_sha256"] = json!(bytes(
                &dir.join("model.bin"),
                format!("synthetic-{id}").as_bytes()
            )?);
            replay["model_sha256"] = result["model_sha256"].clone();
            replay["original_metrics_equal"] = json!(true);
            replay["train_ce"] = result["train_ce"].clone();
            if retained {
                let sha = bytes(&dir.join("features.bin"), b"synthetic features")?;
                result["feature_cache"] = json!({"bytes":18,"sha256":sha});
                replay["feature_cache_audit"] = result["feature_cache"].clone();
                let mut choice = result.clone();
                choice["selected_id"] = json!(id);
                choices.push(choice);
            }
        }
        replay["result_sha256"] = json!(save(&dir, "result.json", &result)?);
        save(root, &format!("{id}.replay.json"), &replay)?;
    }
    inv["cases"] = json!(cases);
    let complete = cases.len() - if retained { 0 } else { 18 };
    let summary = json!({"source_fingerprint":inv["source_fingerprint"],"cells":cases.len(),"complete":complete,
        "memory_ineligible":cases.len()-complete,"summed_candidate_seconds":complete,"choices":choices,"choice_between_retained_initializers":false});
    Ok((inv, summary))
}
fn transformer(root: &Path, suite: &Value) -> Result<(Value, Value)> {
    let mut inv = header("transformer")?;
    let (mut cases, mut specs, mut rows) = (vec![], vec![], vec![]);
    for (index, t) in crate::rows(suite, "tasks")?.iter().enumerate() {
        let name = text(t, "name")?;
        let case = json!({"task":name,"seed":1});
        let id = format!("transformer--{name}--seed1");
        let dir = root.join(&id);
        fs::create_dir(&dir)?;
        let spec = json!({"case":case,"id":id,"data_fingerprints":t["data_fingerprints"]});
        let metadata = json!({"specification":spec,"producer_commit":inv["producer_commit"]});
        save(&dir, "metadata.json", &metadata)?;
        let checkpoint = json!({"file":"at-000001.bin","file_sha256":bytes(&dir.join("at-000001.bin"),b"synthetic endpoint")?,"body_sha256":hash(b"synthetic body")});
        let before = json!({"file":"before-000001.bin","file_sha256":bytes(&dir.join("before-000001.bin"),b"synthetic earlier state")?});
        let observation = json!({"step":1,"checkpoint":checkpoint,"before":before});
        let observations_hash = bytes(
            &dir.join("observations.jsonl"),
            serde_json::to_string(&observation)?.as_bytes(),
        )?;
        let updates_hash = bytes(&dir.join("updates.jsonl"), b"{\"synthetic\":true}")?;
        let result = json!({"id":id,"case":case,"status":"complete","metadata":metadata,"endpoint":{"checkpoint":checkpoint},
            "metrics":metrics(0.5,if index == 0 {0.} else {1.}),"primary_split":"validation","progress":{"completed_steps":1},
            "observations":1,"observations_sha256":observations_hash,"updates_sha256":updates_hash,
            "stop_reason":if index == 0 {"original_update_cap"} else {"primary_target"},"total_seconds_including_data":2.});
        let result_hash = save(&dir, "result.json", &result)?;
        let replay = json!({"id":id,"source_fingerprint":inv["source_fingerprint"],"replayed":true,"result_sha256":result_hash,
            "updates_sha256":updates_hash,"observations_sha256":observations_hash,"endpoint_body_sha256":checkpoint["body_sha256"],
            "observations_replayed":1,"final_update_bit_exact":true,"original_metrics_replayed":true,"complete_training_rerun":false});
        save(root, &format!("{id}.replay.json"), &replay)?;
        let mut row = result;
        row["checkpoint_body_sha256"] = checkpoint["body_sha256"].clone();
        row["primary_pass"] = json!(index != 0);
        row["all_splits_pass"] = json!(index != 0);
        rows.push(row);
        cases.push(case);
        specs.push(spec);
    }
    inv["cases"] = json!(cases);
    inv["specifications"] = json!(specs);
    let mut summary = header("transformer")?;
    for (key, value) in
        json!({"cells":35,"all35_conventional_transformer_evaluated":true,"results":rows,
        "primary_pass":34,"all_split_pass":34,"original_cap_failures":1})
        .as_object()
        .unwrap()
    {
        summary[key] = value.clone();
    }
    Ok((inv, summary))
}
