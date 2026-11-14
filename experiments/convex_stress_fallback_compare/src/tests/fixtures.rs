use super::*;
use std::{
    os::unix::fs::PermissionsExt,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
pub(super) fn overwrite<T: Serialize>(path: &Path, value: &T) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
fn example(tokens: Vec<i64>, label: i64) -> data::Example {
    let n = tokens.len();
    let mut targets = vec![-100; n];
    targets[n - 1] = label;
    data::Example {
        task: "categorical".into(),
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
pub(super) fn input(name: &str) -> Result<inputs::Inputs> {
    let mut long = vec![1; 65];
    long[10] = 0;
    long[64] = 5;
    let task = data::Task {
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
    };
    inputs::assemble(task, name, json!({"synthetic":true}))
}
fn counted(arms: Value, parent: Value) -> Value {
    json!({"arms":arms,"complete_parent":parent,"all35_transformer_endpoints_retained":true,"choose_between_initializers":false,"choices_per_arm":105,
        "constructor_accounting":{"charged_once_across_both_arms":true,"all_prior_costs_retained_in_complete_parent":true}})
}
fn evaluated(model: &component::model::Model, x: &inputs::Inputs) -> Result<Value> {
    let mut metrics = serde_json::Map::new();
    for (split, examples) in &x.task.splits {
        metrics.insert(split.clone(), metrics::evaluate(model, examples)?);
    }
    Ok(Value::Object(metrics))
}
pub(super) fn files(root: &Path, task: &str, success: bool, seal: Option<&Value>) -> Result<()> {
    fs::write(
        root.join(format!("{task}.run.exit")),
        if success { "0\n" } else { "9\n" },
    )?;
    for suffix in ["run.resources", "log"] {
        fs::write(
            root.join(format!("{task}.{suffix}")),
            "retained fixture process evidence\n",
        )?;
    }
    if success {
        fs::write(root.join(format!("{task}.replay.exit")), "0\n")?;
        for suffix in ["replay.resources", "replay.log"] {
            fs::write(
                root.join(format!("{task}.{suffix}")),
                "fixture replay evidence\n",
            )?;
        }
        overwrite(&root.join(format!("{task}.replay.json")), seal.unwrap());
    }
    Ok(())
}
pub(super) struct Fixture {
    pub temp: PathBuf,
    pub suite: Value,
    pub all: Vec<String>,
    pub request: Request,
    pub parent_inv: Value,
    pub parent: Value,
    pub stage_inv: Value,
    pub stage: Value,
    pub expected_parent: Value,
    pub expected_stage: Value,
    pub destination: PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.temp).unwrap();
    }
}
impl Fixture {
    pub(super) fn new() -> Result<Self> {
        let temp = std::env::temp_dir().join(format!(
            "fallback-compare-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir(&temp)?;
        let all: Vec<_> = (0..35).map(|i| format!("fixture-{i}")).collect();
        let suite = json!({"tasks":all.iter().map(|n|json!({"name":n})).collect::<Vec<_>>()});
        let binary = temp.join("summary-fixture");
        fs::write(&binary, "#!/bin/sh\ncat \"$2/summary.json\"\n")?;
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o755))?;
        let stage_root = temp.join("stage");
        let parent_root = temp.join("parent");
        let checkpoint_root = temp.join("checkpoints");
        fs::create_dir(&stage_root)?;
        fs::create_dir(&parent_root)?;
        fs::create_dir(&checkpoint_root)?;
        let x = input(&all[0])?;
        let config = x.configuration.as_ref().unwrap();
        let report = binding::run(config.clone())?;
        let mut previous = config.discovery.clone();
        for s in &mut previous.input.observations {
            *s.tokens.last_mut().unwrap() = 5;
        }
        let previous = component::model::initialize(&component::discovery::run(previous)?)?.model;
        let checkpoint = checkpoint_root.join("actual-parent.json");
        write(&checkpoint, &fallback::Parent::Causal(previous.clone()))?;
        let dir = stage_root.join(&all[0]);
        fs::create_dir(&dir)?;
        write(&dir.join("configuration.json"), &x.configuration)?;
        write(
            &dir.join("attempt.json"),
            &json!({"input_refusal":null,"report":report,"error":null,"construction_seconds":0.25}),
        )?;
        let report_sha = digest(&dir.join("attempt.json"))?;
        let records:Vec<_>=all.iter().enumerate().map(|(i,t)|if i==0 {json!({"task":t,"status":"complete","result":{"attempt_sha256":report_sha,"configuration_sha256":digest(&dir.join("configuration.json")).unwrap()}})}else{json!({"task":t,"status":"process_failure","retained_cost":1})}).collect();
        let inputs = all
            .iter()
            .map(|t| Ok(input(t)?.record))
            .collect::<Result<Vec<_>>>()?;
        let producer = "a".repeat(40);
        let suite_sha = hash(SUITE.as_bytes());
        let source = "b".repeat(64);
        let binary_sha = digest(&binary)?;
        let stage_inv = json!({"source_fingerprint":source,"binary_sha256":binary_sha,"producer_commit":producer,"tasks":all,"inputs":inputs,
            "request":{"suite_sha256":suite_sha},"model_protocol":protocol()?,"staging_protocol":{"fixture":true}});
        write(&stage_root.join("request.json"), &stage_inv["request"])?;
        write(&stage_root.join("inventory.json"), &stage_inv)?;
        let stage = json!({"source_fingerprint":source,"producer_commit":producer,"inventory_sha256":digest(&stage_root.join("inventory.json"))?,"tasks":all,"inputs":inputs,
            "records":records,"model_protocol":stage_inv["model_protocol"],"staging_protocol":stage_inv["staging_protocol"],"phase":"training_only","heldout_evaluation":false,"parent_comparison":false});
        write(&stage_root.join("summary.json"), &stage)?;
        let pin = |root: &Path, review_name: &str| -> Result<Reference> {
            let review = temp.join(review_name);
            fs::create_dir(&review)?;
            fs::write(review.join("process.exit"), "0\n")?;
            fs::write(review.join("reviewed.commit"), format!("{producer}\n"))?;
            Ok(Reference {
                root: root.to_path_buf(),
                binary: binary.clone(),
                summary_sha256: digest(&root.join("summary.json"))?,
                inventory_sha256: digest(&root.join("inventory.json"))?,
                review: Review {
                    exit: review.join("process.exit"),
                    exit_sha256: digest(&review.join("process.exit"))?,
                    commit: review.join("reviewed.commit"),
                    commit_sha256: digest(&review.join("reviewed.commit"))?,
                    summary: root.join("summary.json"),
                },
            })
        };
        let staging = pin(&stage_root, "stage-review")?;
        let mut arms = vec![];
        for policy in POLICIES {
            let mut choices = vec![];
            for t in &all {
                for m in MODES {
                    choices.push(json!({"task":t,"training_mode":m,"status":"selected","reference":format!("old:{policy}:{t}:{m}"),"train_ce":1.,
                    "primary_split":"validation","metrics":{"train":{"loss":1.,"sequence_accuracy":1.},"validation":{"sequence_accuracy":1.},"auxiliary":{"sequence_accuracy":1.}},"primary_pass":true,"all_splits_pass":true,
                    "saved_endpoint":{"model_path":"actual-parent.json","model_sha256":digest(&checkpoint)?}}));
                }
            }
            arms.push(json!({"initializer":policy,"choices":choices}));
        }
        let base = json!({"depth_outcomes":(0..336).map(|i|json!({"slot":i,"refused":i%17==0})).collect::<Vec<_>>(),
            "transformer_endpoints":all.iter().map(|t|json!({"task":t,"retained":true})).collect::<Vec<_>>(),"all_parent_costs_charged_once":true});
        let quotient = counted(json!(arms), base);
        let mut causal_arms = arms;
        for arm in &mut causal_arms {
            for row in arm["choices"].as_array_mut().unwrap() {
                if row["task"] != all[0] {
                    continue;
                }
                let m = row["training_mode"].as_str().unwrap();
                let e = report
                    .endpoints
                    .iter()
                    .find(|e| mode(e.policy.mode) == m)
                    .unwrap();
                let model = &e.training.as_ref().unwrap().model;
                let metrics = evaluated(model, &x)?;
                let (primary, aux) = passes(&metrics, "validation")?;
                let before = row.clone();
                *row = json!({"task":all[0],"training_mode":m,"status":"selected","reference":format!("causal-pairs:{}:{m}",all[0]),"train_ce":e.audit.as_ref().unwrap().train_ce,
                "primary_split":"validation","metrics":metrics,"primary_pass":primary,"all_splits_pass":aux,"baseline":before,
                "saved_endpoint":{"model_sha256":fallback::digest(model)?,"report_sha256":report_sha}});
            }
        }
        let mut causal = counted(json!(causal_arms), quotient);
        causal["staged_construction"] = json!({"reference":{"root":staging.root,"summary_sha256":staging.summary_sha256,"inventory_sha256":staging.inventory_sha256},"summary":stage});
        let refined = counted(causal["arms"].clone(), causal);
        let kernel = counted(refined["arms"].clone(), refined);
        let mut parent = counted(kernel["arms"].clone(), kernel);
        parent["source_fingerprint"] = json!(source);
        parent["producer_commit"] = json!(producer);
        parent["tasks"] = json!(35);
        parent["protocol"] = json!({"fixture":true});
        parent["records"] = json!(
            all.iter()
                .map(|t| json!({"task":t,"status":"complete"}))
                .collect::<Vec<_>>()
        );
        let parent_inv = json!({"source_fingerprint":source,"binary_sha256":binary_sha,"producer_commit":producer,"tasks":all,"request":{"suite_sha256":suite_sha},"protocol":parent["protocol"]});
        write(&parent_root.join("request.json"), &parent_inv["request"])?;
        write(&parent_root.join("inventory.json"), &parent_inv)?;
        write(&parent_root.join("summary.json"), &parent)?;
        let parent_pin = pin(&parent_root, "parent-review")?;
        let expected_parent = json!({"source_fingerprint":source,"binary_sha256":binary_sha,"producer_commit":producer});
        let mut expected_stage = expected_parent.clone();
        expected_stage["summary_sha256"] = json!(staging.summary_sha256);
        expected_stage["inventory_sha256"] = json!(staging.inventory_sha256);
        let request = Request {
            producer_commit: producer,
            suite_sha256: suite_sha,
            parent: parent_pin,
            staging,
            checkpoint_root,
        };
        let destination = temp.join("comparison");
        Ok(Self {
            temp,
            suite,
            all,
            request,
            parent_inv,
            parent,
            stage_inv,
            stage,
            expected_parent,
            expected_stage,
            destination,
        })
    }
    pub(super) fn prepare(&self) -> Result<Value> {
        let records = self
            .all
            .iter()
            .map(|t| Ok(input(t)?.record))
            .collect::<Result<Vec<_>>>()?;
        execute::prepare_with(
            &self.destination,
            self.request.clone(),
            &self.suite,
            records,
            &self.parent_inv,
            &self.parent,
            &self.stage_inv,
            &self.stage,
        )?;
        fs::write(
            self.destination.join("prepare.resources"),
            "actual synthetic preparation evidence\n",
        )?;
        execute::inventory_with(
            &self.destination,
            &self.suite,
            &self.parent_inv,
            &self.parent,
            &self.stage_inv,
            &self.stage,
            true,
        )
    }
    pub(super) fn run(&self, inv: &Value, task: &str) -> Result<Value> {
        let x = input(task)?;
        execute::run_with(&self.destination, task, inv, &self.parent, &self.stage, &x)?;
        execute::retained(&self.destination, task, inv, &self.parent, &self.stage, &x)?;
        let seal = execute::seal(&self.destination, task, inv)?;
        files(&self.destination, task, true, Some(&seal))?;
        Ok(seal)
    }
}
