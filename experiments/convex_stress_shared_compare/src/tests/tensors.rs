//! Actual selected CM34/CM40 histories on a synthetic categorical problem.
//! Unused upstream endpoint outcomes are mocked as unavailable; their complete
//! field catalogue is retained. Production imports require the approved pins.
use super::*;
struct Fixture {
    _temp: Temp,
    all: Vec<String>,
    task: String,
    request: Request,
    reviewed: BTreeMap<String, parents::Reviewed>,
    x: inputs::Inputs,
    c: causal::inputs::Inputs,
}
impl Fixture {
    fn new() -> Result<Self> {
        let temp = Temp::new()?;
        let all = tasks(&suite()?)?;
        let name = all[0].clone();
        let mut x = inputs::assemble(task(), &name, true, 2, json!({"synthetic":true}))?;
        x.readout = Some(vec![0, 1, 2, 0]);
        x.record["readout"] = json!([0, 1, 2, 0]);
        let c = causal::inputs::assemble(task(), &name, json!({"synthetic":true}))?;
        let input = x.input.as_ref().unwrap();
        let policies = x.policies.as_ref().unwrap();
        let records = input
            .observations
            .iter()
            .enumerate()
            .map(|(id, &[cell, label])| shared::records::Record { id, cell, label })
            .collect::<Vec<_>>();
        let sp = shared::training::policies(records.len(), 2)?;
        let mut refs = BTreeMap::new();
        let mut reviewed = BTreeMap::new();
        reviewed.insert(
            "parent".into(),
            parents::Reviewed {
                summary: parent(&all),
            },
        );
        for key in ["parent", "raw", "shared", "causal"] {
            let dir = temp.0.join(key);
            fs::create_dir(&dir)?;
            refs.insert(key.into(), reference(dir));
        }
        for key in ["raw", "shared"] {
            let dir = refs[key].root.join(&name);
            fs::create_dir(&dir)?;
            write(&dir.join("input.json"), input)?;
            if key == "raw" {
                write(&dir.join("policies.json"), policies)?;
            } else {
                write(&dir.join("policies.json"), &sp)?;
                write(&dir.join("records.json"), &records)?;
                write(
                    &dir.join("readout.json"),
                    &json!({"classes":[0,1,2,0],"answers":3}),
                )?;
            }
            let mut stages = vec![];
            for (i, prime) in catalogue().into_iter().enumerate() {
                let file = format!("stage-{i:02}-p{prime}.json");
                let a = if key == "raw" {
                    let d = discovery::run(discovery::Request {
                        input: input.clone(),
                        prime,
                    })?;
                    let h = if prime == 3 {
                        Some(binding::run(&d, policies.clone())?)
                    } else {
                        None
                    };
                    json!({"discovery":d,"discovery_error":Value::Null,"head":h,
                      "head_error":if prime==3{Value::Null}else{json!("synthetic upstream endpoint unavailable")},
                      "discovery_seconds":0.,"head_seconds":0.})
                } else {
                    let d = shared::discovery::run(shared::discovery::Request {
                        input: input.clone(),
                        readout: vec![0, 1, 2, 0],
                        answers: 3,
                        prime,
                    })?;
                    let h = shared::binding::run(&d, &records, sp.clone())?;
                    json!({"discovery":d,"discovery_error":Value::Null,"head":h,"head_error":Value::Null,"discovery_seconds":0.,"head_seconds":0.})
                };
                write(&dir.join(&file), &a)?;
                stages.push(json!({"prime":prime,"file":file,"sha256":digest(&dir.join(&file))?,"bytes":fs::metadata(dir.join(&file))?.len(),
                  "retained":{"discovery_seconds":0.,"head_seconds":0.,"discovery_error":a["discovery_error"],"head_error":a["head_error"]}}));
            }
            let result = json!({"metadata":{"input":x.record},"input_sha256":digest(&dir.join("input.json"))?,
              "policies_sha256":digest(&dir.join("policies.json"))?,"stages":stages,"heldout_evaluation":false,
              "parent_comparison":false,"raw_label_identification":false,"target_readout_compatibility_proved":false});
            write(&dir.join("result.json"), &result)?;
            let mut artifacts = vec![];
            parents::tree(&dir, &mut artifacts)?;
            reviewed.insert(
                key.into(),
                parents::Reviewed {
                    summary: json!({"planned":[name],"inputs":[x.record],
              "records":[{"task":name,"status":"complete","result":result,"artifacts":artifacts}]}),
                },
            );
        }
        reviewed.insert(
            "causal".into(),
            parents::Reviewed {
                summary: json!({"inputs":[c.record]}),
            },
        );
        let request = Request {
            producer_commit: "a".repeat(40),
            suite_sha256: hash(SUITE.as_bytes()),
            parents: refs,
            checkpoint_root: temp.0.clone(),
        };
        Ok(Self {
            _temp: temp,
            all,
            task: name,
            request,
            reviewed,
            x,
            c,
        })
    }
    fn build(&self) -> Result<selected::Bundle> {
        selected::build(
            &self.request,
            &self.reviewed,
            &self.task,
            &self.x,
            &self.c,
            &self.all,
        )
    }
    fn refresh(&mut self, key: &str) -> Result<()> {
        let dir = self.request.parents[key].root.join(&self.task);
        let result: Value = read(&dir.join("result.json"))?;
        let mut artifacts = vec![];
        parents::tree(&dir, &mut artifacts)?;
        let summary = &mut self.reviewed.get_mut(key).unwrap().summary;
        summary["records"][0]["result"] = result;
        summary["records"][0]["artifacts"] = json!(artifacts);
        Ok(())
    }
}
#[test]
fn selected_actual_histories_extend_all_modes_and_heldout_labels_do_not_select() -> Result<()> {
    let mut f = Fixture::new()?;
    let first = f.build()?;
    assert_eq!(first.models.len(), 3);
    assert!(
        first.choices["choices"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["transformation"]["status"] == "extended")
    );
    let fresh = outcomes::evaluate(&f.task, &f.x, &first)?;
    assert_eq!(fresh.len(), 3);
    assert!(
        fresh
            .iter()
            .all(|r| r["primary_pass"] == true && r["metric_provenance"] == "fresh_changed_model")
    );
    assert!(
        fresh
            .iter()
            .all(|r| r["prediction_masks"]["validation"]["supplement_route_calls"] == 1)
    );
    f.x.task.splits.get_mut("validation").unwrap()[0].targets[1] = 1;
    let second = f.build()?;
    assert_eq!(first.choices, second.choices);
    let changed = outcomes::evaluate(&f.task, &f.x, &second)?;
    assert!(changed.iter().all(|r| r["primary_pass"] == false));
    Ok(())
}
#[test]
fn rewritten_source_ledgers_cannot_authorize_changed_tensors_or_record_order() -> Result<()> {
    let mut f = Fixture::new()?;
    f.build()?;
    let dir = f.request.parents["raw"].root.join(&f.task);
    let path = dir.join("stage-01-p3.json");
    let saved: Value = read(&path)?;
    let mut changed = saved.clone();
    changed["head"]["endpoints"][0]["training"]["model"]["coefficients"][0] = json!(19.);
    overwrite(&path, &changed);
    let result_path = dir.join("result.json");
    let result: Value = read(&result_path)?;
    let mut rebuilt = result.clone();
    rebuilt["stages"][1]["sha256"] = json!(digest(&path)?);
    rebuilt["stages"][1]["bytes"] = json!(fs::metadata(&path)?.len());
    overwrite(&result_path, &rebuilt);
    f.refresh("raw")?;
    assert!(f.build().is_err());
    overwrite(&path, &saved);
    overwrite(&result_path, &result);
    f.refresh("raw")?;
    let path = f.request.parents["shared"]
        .root
        .join(&f.task)
        .join("records.json");
    let mut records: Value = read(&path)?;
    records.as_array_mut().unwrap().swap(0, 1);
    overwrite(&path, &records);
    f.refresh("shared")?;
    assert!(f.build().is_err());
    Ok(())
}
#[test]
fn metric_domain_and_work_refusals_precede_actual_predictions() -> Result<()> {
    let f = Fixture::new()?;
    let bundle = f.build()?;
    let model = bundle.models.values().next().unwrap();
    let mut invalid = task();
    invalid.splits.get_mut("validation").unwrap()[0].tokens[0] = 4;
    assert!(outcomes::forecast(&invalid, model).is_err());
    let mut oversized = task();
    let e = &mut oversized.splits.get_mut("validation").unwrap()[0];
    e.prompt = vec![0, 1];
    e.limit = 4095;
    assert!(outcomes::forecast(&oversized, model).is_err());
    let mut expensive = task();
    let mut sample = expensive.splits["validation"][0].clone();
    sample.prompt = vec![0, 1];
    sample.limit = 4094;
    expensive
        .splits
        .insert("validation".into(), vec![sample; 10_000]);
    assert!(outcomes::forecast(&expensive, model).is_err());
    assert!(model.logits(&[0]).is_err());
    assert!(model.logits(&[0, -1]).is_err());
    assert!(model.logits(&vec![0; 4097]).is_err());
    Ok(())
}

#[test]
fn actual_saved_case_replays_and_rejects_forged_metrics_or_partial_resume() -> Result<()> {
    let f = Fixture::new()?;
    let root = f._temp.0.join("case-run");
    fs::create_dir(&root)?;
    let inv = json!({"binary_sha256":"f".repeat(64),"request":f.request,"protocol":protocol()?});
    let context = execute::Context {
        inv: &inv,
        request: &f.request,
        reviewed: &f.reviewed,
        all: &f.all,
    };
    execute::run_with(&root, &f.task, &context, &f.x, &f.c)?;
    execute::replay_with(&root, &f.task, &context, &f.x, &f.c)?;
    let result_path = root.join(&f.task).join("result.json");
    let saved: Value = read(&result_path)?;
    assert_eq!(saved["seal"]["metrics_started"], false);
    assert_eq!(saved["seal"]["models"].as_array().unwrap().len(), 3);
    assert!(execute::run_with(&root, &f.task, &context, &f.x, &f.c).is_err());
    assert_eq!(read::<Value>(&result_path)?, saved);
    let mut changed = saved.clone();
    changed["outcomes"][0]["metrics"]["validation"]["sequence_accuracy"] = json!(0.);
    changed["outcomes"][0]["primary_pass"] = json!(false);
    overwrite(&result_path, &changed);
    assert!(execute::replay_with(&root, &f.task, &context, &f.x, &f.c).is_err());
    overwrite(&result_path, &saved);
    let file = saved["seal"]["models"][0]["file"].as_str().unwrap();
    let path = root.join(&f.task).join(file);
    let mut model: Value = read(&path)?;
    model["state"]["prior"]["coefficients"][0] = json!(19.);
    overwrite(&path, &model);
    assert!(execute::replay_with(&root, &f.task, &context, &f.x, &f.c).is_err());
    Ok(())
}
