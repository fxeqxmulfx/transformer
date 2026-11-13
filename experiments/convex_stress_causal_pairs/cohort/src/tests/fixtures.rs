use super::*;
use std::{
    os::unix::fs::PermissionsExt,
    sync::atomic::{AtomicU64, Ordering},
};
static SERIAL: AtomicU64 = AtomicU64::new(0);
pub(super) struct Fixture {
    pub temp: PathBuf,
    pub run: PathBuf,
    pub suite: Value,
    pub request: Request,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.temp).unwrap();
    }
}
pub(super) fn overwrite<T: Serialize>(path: &Path, value: &T) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
pub(super) fn task(two_tokens: bool) -> data::Task {
    let examples = (0..4)
        .map(|i| {
            let (a, b, label) = (i / 2, i % 2, i + 2);
            let prompt = if two_tokens { vec![] } else { vec![a, b, 6] };
            data::Example {
                task: "synthetic-categorical".into(),
                tokens: if two_tokens {
                    vec![a, b]
                } else {
                    vec![a, b, 6, label]
                },
                targets: if two_tokens {
                    vec![-100, label]
                } else {
                    vec![-100, -100, label, 17]
                },
                changes: if two_tokens {
                    vec![false, true]
                } else {
                    vec![false, false, true, true]
                },
                prompt,
                answer: if two_tokens { vec![] } else { vec![label, 17] },
                limit: if two_tokens { 0 } else { 2 },
                field: None,
                row_index: None,
            }
        })
        .collect::<Vec<_>>();
    data::Task {
        vocab: 20,
        selection: "validation".into(),
        splits: BTreeMap::from([
            ("train".into(), examples.clone()),
            ("validation".into(), examples.clone()),
            ("test".into(), examples),
        ]),
    }
}
pub(super) fn input(i: usize) -> Result<inputs::Inputs> {
    inputs::assemble(
        task(i == 2),
        &format!("fixture-{i}"),
        json!({"synthetic":true}),
    )
}
pub(super) fn parent_value() -> Result<(Value, Vec<String>)> {
    let names: Vec<_> = (0..35).map(|i| format!("fixture-{i}")).collect();
    let arms: Vec<_> = POLICIES.iter().map(|policy| {
        let mut rows = vec![];
        for (i, task) in names.iter().enumerate() {
            for mode in MODES {
                rows.push(if i == 1 && mode == MODES[2] {
                    json!({"task":task,"training_mode":mode,"status":"no_model","reference":null,"train_ce":null,
                        "primary_pass":false,"all_splits_pass":false,"reason":"retained original refusal"})
                } else {
                    json!({"task":task,"training_mode":mode,"status":"selected","reference":format!("prior:{policy}:{task}:{mode}"),
                        "train_ce":1.0,"primary_split":"validation","primary_pass":i!=0,"all_splits_pass":i!=0,
                        "metrics":{"train":{"loss":1.0,"sequence_accuracy":0.5},
                            "validation":{"sequence_accuracy":if i==0{0.0}else{1.0}},"test":{"sequence_accuracy":if i==0{0.0}else{1.0}}}})
                });
            }
        }
        json!({"initializer":policy,"choices":rows})
    }).collect();
    let p = protocol()?;
    Ok((
        json!({"source_fingerprint":p["parent_source_fingerprint"],"producer_commit":p["parent_producer_commit"],
        "protocol":{"parent_fixture":true},"parent_sha256":"c".repeat(64),"arms":arms,"choices_per_arm":105,
        "tasks":23,"records":names[..23].iter().map(|n|json!({"task":n,"status":"process_failure","cost":23})).collect::<Vec<_>>(),
        "choose_between_initializers":false,"all35_transformer_endpoints_retained":true,
        "constructor_accounting":{"charged_once_across_both_arms":true,"all_prior_costs_retained_in_complete_parent":true},
        "complete_parent":{"depth_outcomes":(0..336).collect::<Vec<_>>(),"transformer_endpoints":names,
            "all_parent_costs_charged_once":true,"failure":"retained ancestor failure","cost":12345}}),
        names,
    ))
}
impl Fixture {
    pub(super) fn new() -> Result<Self> {
        let temp = PathBuf::from(format!(
            "/tmp/causal-original-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir(&temp)?;
        let root = temp.join("parent");
        fs::create_dir(&root)?;
        let (summary, names) = parent_value()?;
        write(&root.join("summary.json"), &summary)?;
        write(
            &root.join("request.json"),
            &json!({"parent":{"summary_sha256":"c".repeat(64)}}),
        )?;
        write(
            &root.join("old-model.json"),
            &json!({"original_state":[3,2,1],"cost":456}),
        )?;
        let binary = temp.join("parent-binary");
        // A synthetic frozen parent summarizer. Its input root is generated
        // locally from PID/counter, never from shell-interpolated user text.
        fs::write(
            &binary,
            format!("#!/bin/sh\nexec cat '{}/summary.json'\n", root.display()),
        )?;
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o700))?;
        write(
            &root.join("inventory.json"),
            &json!({"source_fingerprint":summary["source_fingerprint"],
            "producer_commit":summary["producer_commit"],"binary_sha256":digest(&binary)?,
            "request_sha256":digest(&root.join("request.json"))?,"request":{"parent":{"summary_sha256":"c".repeat(64)}},
            "tasks":names,"planned":names[..23],"protocol":summary["protocol"]}),
        )?;
        let review = temp.join("review");
        fs::create_dir(&review)?;
        fs::write(review.join("exit"), "0\n")?;
        fs::write(review.join("commit"), "d".repeat(40))?;
        fs::copy(root.join("summary.json"), review.join("summary.json"))?;
        let request = Request {
            producer_commit: "e".repeat(40),
            suite_sha256: hash(SUITE.as_bytes()),
            parent: Parent {
                binary,
                summary_sha256: digest(&root.join("summary.json"))?,
                inventory_sha256: digest(&root.join("inventory.json"))?,
                root,
                review: Review {
                    exit: review.join("exit"),
                    commit: review.join("commit"),
                    summary: review.join("summary.json"),
                },
            },
            staging: None,
        };
        Ok(Self {
            run: temp.join("run"),
            suite: json!({"tasks":names.iter().map(|n|json!({"name":n})).collect::<Vec<_>>()}),
            temp,
            request,
        })
    }
    pub(super) fn prepare(&self) -> Result<(Value, Value)> {
        let inputs = (0..35)
            .map(|i| Ok(input(i)?.record))
            .collect::<Result<_>>()?;
        inventory::prepare_with(&self.run, self.request.clone(), &self.suite, inputs)?;
        inventory::inventory_with(&self.run, &self.suite, true)
    }
    pub(super) fn successful(&self, i: usize, inv: &Value, parent: &Value) -> Result<()> {
        let task = format!("fixture-{i}");
        execute::run_with(&self.run, &task, inv, parent, input(i)?)?;
        let replay = execute::replay_with(&self.run, &task, inv, parent, input(i)?)?;
        write(&self.run.join(format!("{task}.replay.json")), &replay)?;
        for stage in ["run", "replay"] {
            fs::write(self.run.join(format!("{task}.{stage}.exit")), "0\n")?;
            fs::write(
                self.run.join(format!("{task}.{stage}.resources")),
                "synthetic peak KiB1234\n",
            )?;
        }
        Ok(())
    }
    pub(super) fn failed(&self, i: usize) -> Result<()> {
        let task = format!("fixture-{i}");
        let dir = self.run.join(&task);
        fs::create_dir(&dir)?;
        write(
            &dir.join("partial.json"),
            &json!({"partial_work":1234,"interrupted":true}),
        )?;
        fs::write(self.run.join(format!("{task}.run.exit")), "137\n")?;
        fs::write(
            self.run.join(format!("{task}.run.resources")),
            "synthetic peak KiB5678\n",
        )?;
        fs::write(
            self.run.join(format!("{task}.log")),
            "synthetic process failure\n",
        )?;
        Ok(())
    }
}
