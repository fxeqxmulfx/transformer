//! Actual CM38 histories, original repeated prefixes and a long query. The
//! ancestry is the reviewed refined -> causal -> quotient shape used here.
use super::*;
use convex_stress_fallback::{self as fallback, causal as core};
fn example(tokens: Vec<i64>, label: i64) -> data::Example {
    let n = tokens.len();
    let mut targets = vec![-100; n];
    targets[n - 1] = label;
    data::Example {
        task: "synthetic".into(),
        tokens,
        targets,
        changes: vec![true; n],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    }
}
fn task() -> data::Task {
    let mut long = vec![1; 65];
    long[10] = 0;
    long[64] = 5;
    data::Task {
        vocab: 7,
        selection: "validation".into(),
        splits: BTreeMap::from([
            (
                "train".into(),
                vec![
                    example(vec![1, 1, 6], 5),
                    example(vec![0, 1, 6], 4),
                    example(vec![0, 1, 6], 4),
                ],
            ),
            (
                "validation".into(),
                vec![example(long.clone(), 4), example(vec![0, 0, 6], 4)],
            ),
            ("auxiliary".into(), vec![example(long, 2)]),
        ]),
    }
}
#[test]
fn actual_causal_fallback_keeps_primary_success_and_replays_both_saved_models() -> Result<()> {
    let temp = Temp::new()?;
    let all = tasks(&suite()?)?;
    let name = &all[0];
    let x = inputs::assemble(task(), name, false, 64, json!({"synthetic":true}))?;
    let c = causal::inputs::assemble(task(), name, json!({"synthetic":true}))?;
    let config = c.configuration.as_ref().unwrap();
    let report = core::binding::run(config.clone())?;
    let mut previous = config.discovery.clone();
    for r in &mut previous.input.observations {
        *r.tokens.last_mut().unwrap() = 5;
    }
    let previous = core::model::initialize(&core::discovery::run(previous)?)?.model;
    let checkpoint = temp.0.join("prior.json");
    write(&checkpoint, &fallback::Parent::Causal(previous))?;
    let stage_root = temp.0.join("causal");
    fs::create_dir(&stage_root)?;
    let dir = stage_root.join(name);
    fs::create_dir(&dir)?;
    write(&dir.join("configuration.json"), &c.configuration)?;
    write(
        &dir.join("attempt.json"),
        &json!({"input_refusal":null,"report":report,"error":null,"construction_seconds":0.25}),
    )?;
    let report_sha = digest(&dir.join("attempt.json"))?;
    let stage = json!({"inputs":[c.record],"records":[{"task":name,"status":"complete","result":{
      "attempt_sha256":report_sha,"configuration_sha256":digest(&dir.join("configuration.json"))?}}]});
    let mut parent = fixtures::parent(&all);
    let mut quotient = parent["complete_parent"]["complete_parent"].clone();
    for arm in quotient["arms"].as_array_mut().unwrap() {
        for row in arm["choices"].as_array_mut().unwrap() {
            if row["task"] == *name {
                row["saved_endpoint"] =
                    json!({"model_path":"prior.json","model_sha256":digest(&checkpoint)?});
            }
        }
    }
    let mut actual = parent["complete_parent"].clone();
    actual["complete_parent"] = quotient.clone();
    for (a, arm) in actual["arms"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        for row in arm["choices"].as_array_mut().unwrap() {
            if row["task"] != *name {
                continue;
            }
            let m = row["training_mode"].as_str().unwrap().to_owned();
            let index = MODES.iter().position(|s| *s == m).unwrap();
            let endpoint = &report.endpoints[index];
            let model = &endpoint.training.as_ref().unwrap().model;
            let measured = c
                .task
                .splits
                .iter()
                .map(|(split, examples)| Ok((split.clone(), metrics::evaluate(model, examples)?)))
                .collect::<Result<serde_json::Map<String, Value>>>()?;
            let measured = Value::Object(measured);
            let (primary, aux) = passes(&measured, &c.task.selection)?;
            let old = quotient["arms"][a]["choices"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["task"] == *name && r["training_mode"] == m)
                .unwrap();
            *row = json!({"task":name,"training_mode":m,"status":"selected","reference":format!("causal-pairs:{name}:{m}"),
              "train_ce":endpoint.audit.as_ref().unwrap().train_ce,"primary_split":"validation","metrics":measured,
              "primary_pass":primary,"all_splits_pass":aux,"baseline":old,
              "saved_endpoint":{"model_sha256":fallback::digest(model)?,"report_sha256":report_sha}});
        }
    }
    parent["arms"] = actual["arms"].clone();
    parent["complete_parent"] = actual;
    let mut refs = BTreeMap::new();
    for key in parents::KEYS {
        refs.insert(
            key.into(),
            reference(if key == "causal" {
                stage_root.clone()
            } else {
                temp.0.join(key)
            }),
        );
    }
    let request = Request {
        producer_commit: "a".repeat(40),
        suite_sha256: hash(SUITE.as_bytes()),
        parents: refs,
        checkpoint_root: temp.0.clone(),
    };
    let reviewed = BTreeMap::from([
        (
            "parent".into(),
            parents::Reviewed {
                summary: parent.clone(),
            },
        ),
        (
            "raw".into(),
            parents::Reviewed {
                summary: json!({"planned":[]}),
            },
        ),
        ("shared".into(), parents::Reviewed { summary: json!({}) }),
        ("causal".into(), parents::Reviewed { summary: stage }),
    ]);
    let root = temp.0.join("run");
    fs::create_dir(&root)?;
    let inv = json!({"binary_sha256":"f".repeat(64),"request":request,"protocol":protocol()?});
    let context = execute::Context {
        inv: &inv,
        request: &request,
        reviewed: &reviewed,
        all: &all,
    };
    execute::run_with(&root, name, &context, &x, &c)?;
    execute::replay_with(&root, name, &context, &x, &c)?;
    let result: Value = read(&root.join(name).join("result.json"))?;
    assert_eq!(result["outcomes"].as_array().unwrap().len(), 6);
    for row in result["outcomes"].as_array().unwrap() {
        assert_eq!(row["primary_pass"], true);
        assert_eq!(row["all_splits_pass"], false);
        assert_eq!(row["metric_provenance"], "fresh_changed_model");
        assert_eq!(row["covered_records"], 3);
        assert_eq!(
            row["prediction_masks"]["validation"]["supported_mask"],
            json!([false, true])
        );
    }
    let mut cases = inherited_cases(&parent, &all)?;
    cases[0]["result"] = result;
    let arms = summary::render(&parent, &cases, &all, &request.parents["parent"])?;
    assert!(arms.iter().all(|a| {
        a["by_mode"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["primary_passes"] == 35 && m["all_split_passes"] == 34)
    }));
    fs::write(&checkpoint, "changed predecessor")?;
    assert!(execute::replay_with(&root, name, &context, &x, &c).is_err());
    Ok(())
}
