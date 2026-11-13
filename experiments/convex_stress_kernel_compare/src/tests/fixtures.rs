use super::*;
use std::{
    os::unix::fs::PermissionsExt,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
pub(super) fn overwrite<T: Serialize>(path: &Path, value: &T) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
pub(super) fn input(name: &str) -> Result<inputs::Inputs> {
    let mut rows = vec![];
    for cell in 0..36 {
        let (i, j) = (cell % 6, cell / 6);
        let code = (i + 2 * j) % 3;
        let alias = if cell == 35 {
            0
        } else {
            usize::from((i + j) % 4 == 0)
        };
        let label = 20 + 7 * code + 97 * alias;
        rows.push(data::Example {
            task: "categorical".into(),
            tokens: vec![j as i64, i as i64],
            targets: vec![-100, label as i64],
            changes: vec![false, true],
            prompt: vec![],
            answer: vec![],
            limit: 0,
            field: Some((0, (label as i64 - 98).rem_euclid(97))),
            row_index: None,
        });
    }
    let mut diagnostic = rows[35].clone();
    diagnostic.targets[1] = 27;
    diagnostic.field = Some((0, (27_i64 - 98).rem_euclid(97)));
    let mut train = rows[..35].to_vec();
    let mut disconnected = rows[0].clone();
    disconnected.tokens = vec![6, 6];
    train.push(disconnected);
    let task = data::Task {
        vocab: 197,
        selection: "validation".into(),
        splits: BTreeMap::from([
            ("train".into(), train),
            ("validation".into(), rows[35..].to_vec()),
            ("auxiliary".into(), vec![diagnostic]),
        ]),
    };
    inputs::assemble(
        task,
        name,
        name == "fixture-0" || name == "fixture-1",
        13,
        json!({"synthetic":true}),
    )
}
pub(super) fn parent(all: &[String]) -> Value {
    let arms:Vec<_>=POLICIES.iter().map(|policy|{let mut choices=vec![];
        for (i,task) in all.iter().enumerate(){for mode in MODES{
            choices.push(json!({"task":task,"training_mode":mode,"status":"selected","reference":format!("prior:{policy}:{task}:{mode}"),"train_ce":1.,"primary_split":"validation","metrics":{"train":{"loss":1.,"sequence_accuracy":0.5},"validation":{"sequence_accuracy":if i==0{0.}else{1.}},"auxiliary":{"sequence_accuracy":if i==0{0.}else{1.}}},"primary_pass":i!=0,"all_splits_pass":i!=0}));
        }}json!({"initializer":policy,"choices":choices})}).collect();
    let base = json!({"depth_outcomes":(0..336).map(|i|json!({"slot":i,"refused":i%17==0})).collect::<Vec<_>>(),
        "transformer_endpoints":all.iter().map(|n|json!({"task":n,"retained":true})).collect::<Vec<_>>(),"all_parent_costs_charged_once":true});
    let quotient = json!({"complete_parent":base,"all35_transformer_endpoints_retained":true,"choose_between_initializers":false,"choices_per_arm":105,"constructor_accounting":{"charged_once_across_both_arms":true,"all_prior_costs_retained_in_complete_parent":true}});
    let causal = json!({"complete_parent":quotient,"all35_transformer_endpoints_retained":true,"choose_between_initializers":false,"choices_per_arm":105,"constructor_accounting":{"charged_once_across_both_arms":true,"all_prior_costs_retained_in_complete_parent":true}});
    json!({"arms":arms,"complete_parent":causal,"all35_transformer_endpoints_retained":true,"choose_between_initializers":false,"choices_per_arm":105,
        "constructor_accounting":{"charged_once_across_both_arms":true,"all_prior_costs_retained_in_complete_parent":true}})
}
pub(super) fn files(root: &Path, task: &str, success: bool, seal: Option<&Value>) -> Result<()> {
    fs::write(
        root.join(format!("{task}.run.exit")),
        if success { "0\n" } else { "9\n" },
    )?;
    for suffix in ["run.resources", "log"] {
        fs::write(
            root.join(format!("{task}.{suffix}")),
            "retained actual fixture work\n",
        )?;
    }
    if success {
        fs::write(root.join(format!("{task}.replay.exit")), "0\n")?;
        for suffix in ["replay.resources", "replay.log"] {
            fs::write(
                root.join(format!("{task}.{suffix}")),
                "retained fixture replay work\n",
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
    pub planned: Vec<String>,
    pub request: Request,
    pub parent_inv: Value,
    pub parent: Value,
    pub stage_inv: Value,
    pub stage: Value,
    pub destination: PathBuf,
    pub expected_parent: Value,
    pub expected_stage: Value,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.temp).unwrap();
    }
}
impl Fixture {
    pub(super) fn new() -> Result<Self> {
        let temp = std::env::temp_dir().join(format!(
            "kernel-compare-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir(&temp)?;
        let all: Vec<_> = (0..35).map(|i| format!("fixture-{i}")).collect();
        let planned = all[..2].to_vec();
        let suite = json!({"tasks":all.iter().map(|n|json!({"name":n})).collect::<Vec<_>>()});
        let binary = temp.join("summary-fixture");
        fs::write(&binary, "#!/bin/sh\ncat \"$2/summary.json\"\n")?;
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o755))?;
        let producer = "166bebc3d1d1d206b1169e8c33b02bde60549afd";
        let source_fp = "a".repeat(64);
        let expected = json!({"source_fingerprint":source_fp,"producer_commit":producer,"binary_sha256":digest(&binary)?});
        let x = input("fixture-0")?;
        let raw = x.input.as_ref().unwrap();
        let stage_root = temp.join("stage");
        fs::create_dir(&stage_root)?;
        fs::create_dir(stage_root.join("fixture-0"))?;
        let mut stages = vec![];
        for (index, prime) in catalogue().into_iter().enumerate() {
            let file = format!("stage-{index:02}-p{prime}.json");
            let discovery = discovery::run(discovery::Request {
                input: raw.clone(),
                prime,
            })?;
            let head = if prime == 3 {
                assert_eq!(discovery.code.as_ref().unwrap().dimension, 2);
                assert!(outcomes::compatible(
                    &discovery,
                    x.readout.as_ref().unwrap()
                ));
                Some(binding::run(&discovery, x.policies.clone().unwrap())?)
            } else {
                None
            };
            // Other stages explicitly simulate execution failure. The real p3
            // tensors and every prime's actual basis are independently checked.
            let a = json!({"discovery":discovery,"discovery_error":null,"head":head,
                "head_error":if prime==3{None}else{Some("simulated head execution failure")},
                "discovery_seconds":0.125,"head_seconds":0.125});
            write(&stage_root.join("fixture-0").join(&file), &a)?;
            stages.push(json!({"file":file,"sha256":digest(&stage_root.join("fixture-0").join(&file))?,"bytes":fs::metadata(stage_root.join("fixture-0").join(&file))?.len(),
                "prime":prime,"retained":{"discovery_error":a["discovery_error"],"head_error":a["head_error"],"discovery_seconds":0.125,"head_seconds":0.125}}));
        }
        write(&stage_root.join("fixture-0/input.json"), raw)?;
        write(
            &stage_root.join("fixture-0/policies.json"),
            x.policies.as_ref().unwrap(),
        )?;
        let inputs: Vec<_> = planned
            .iter()
            .map(|n| input(n).map(|x| x.record))
            .collect::<Result<_>>()?;
        let result = json!({"metadata":{"input":inputs[0]},"input_sha256":digest(&stage_root.join("fixture-0/input.json"))?,"policies_sha256":digest(&stage_root.join("fixture-0/policies.json"))?,"stages":stages,"heldout_evaluation":false,"parent_comparison":false,"raw_label_identification":false,"target_readout_compatibility_proved":false});
        write(&stage_root.join("fixture-0/result.json"), &result)?;
        let mut artifacts = vec![];
        parents::tree(&stage_root.join("fixture-0"), &mut artifacts)?;
        fs::create_dir(stage_root.join("fixture-1"))?;
        write(
            &stage_root.join("fixture-1/partial.json"),
            &json!({"retained_work":99}),
        )?;
        let partial = parents::evidence(&stage_root.join("fixture-1/partial.json"))?;
        let stage_inv = json!({"source_fingerprint":source_fp,"binary_sha256":digest(&binary)?,"request":{"producer_commit":producer,"suite_sha256":hash(SUITE.as_bytes())},"tasks":all,"planned":planned,"protocol":{"synthetic":true},"inputs":inputs});
        write(&stage_root.join("inventory.json"), &stage_inv)?;
        write(&stage_root.join("request.json"), &stage_inv["request"])?;
        let stage = json!({"source_fingerprint":source_fp,"inventory_sha256":digest(&stage_root.join("inventory.json"))?,"request":stage_inv["request"],"protocol":stage_inv["protocol"],"phase":"training_only","tasks":all,"planned":planned,"inputs":inputs,"heldout_evaluation":false,"parent_comparison":false,"raw_label_identification":false,"target_readout_compatibility_proved":false,
            "records":[{"task":"fixture-0","status":"complete","result":result,"artifacts":artifacts},{"task":"fixture-1","status":"process_failure","artifacts":[partial]}],
            "constructor_accounting":{"all_discovery_work_in_stage_records":true,"charged_once_across_both_arms":true,"complete_discovery_seconds":6.875,"complete_head_seconds":6.875,"failed_work_retained":true}});
        write(&stage_root.join("summary.json"), &stage)?;
        let parent_root = temp.join("parent");
        fs::create_dir(&parent_root)?;
        let parent_inv = json!({"producer_commit":producer,"source_fingerprint":source_fp,"binary_sha256":digest(&binary)?,"protocol":{"synthetic":true},"tasks":all,"request":{"suite_sha256":hash(SUITE.as_bytes())}});
        let mut parent = parent(&all);
        parent["source_fingerprint"] = json!(source_fp);
        parent["producer_commit"] = json!(producer);
        parent["protocol"] = parent_inv["protocol"].clone();
        parent["tasks"] = json!(35);
        parent["staged_construction"] = json!({"preserved237":true});
        parent["records"] = json!(
            all.iter()
                .map(|t| json!({"task":t,"status":"complete"}))
                .collect::<Vec<_>>()
        );
        write(&parent_root.join("inventory.json"), &parent_inv)?;
        write(&parent_root.join("request.json"), &parent_inv["request"])?;
        write(&parent_root.join("summary.json"), &parent)?;
        let make_pin = |root: PathBuf, label: &str| -> Result<Reference> {
            let review = temp.join(label);
            fs::create_dir(&review)?;
            fs::write(review.join("process.exit"), "0\n")?;
            fs::write(review.join("reviewed.commit"), producer)?;
            fs::copy(root.join("summary.json"), review.join("summary.json"))?;
            Ok(Reference {
                summary_sha256: digest(&root.join("summary.json"))?,
                inventory_sha256: digest(&root.join("inventory.json"))?,
                root,
                binary: binary.clone(),
                review: Review {
                    exit_sha256: digest(&review.join("process.exit"))?,
                    exit: review.join("process.exit"),
                    commit_sha256: digest(&review.join("reviewed.commit"))?,
                    commit: review.join("reviewed.commit"),
                    summary: review.join("summary.json"),
                },
            })
        };
        let request = Request {
            producer_commit: producer.into(),
            suite_sha256: hash(SUITE.as_bytes()),
            parent: make_pin(parent_root, "parent-review")?,
            staging: make_pin(stage_root, "stage-review")?,
        };
        let destination = temp.join("comparison");
        Ok(Self {
            temp,
            suite,
            all,
            planned,
            request,
            parent_inv,
            parent,
            stage_inv,
            stage,
            destination,
            expected_parent: expected.clone(),
            expected_stage: expected,
        })
    }
    pub(super) fn prepare(&self) -> Result<Value> {
        parents::load_with(
            &self.request.parent,
            parents::Kind::Parent,
            &self.expected_parent,
            &self.all,
            &self.planned,
            true,
        )?;
        parents::load_with(
            &self.request.staging,
            parents::Kind::Kernel,
            &self.expected_stage,
            &self.all,
            &self.planned,
            true,
        )?;
        execute::prepare_with(
            &self.destination,
            self.request.clone(),
            &self.suite,
            &self.planned,
            self.all
                .iter()
                .map(|t| Ok(input(t)?.record))
                .collect::<Result<_>>()?,
            self.parent_inv.clone(),
            self.parent.clone(),
            self.stage_inv.clone(),
            self.stage.clone(),
        )?;
        fs::write(
            self.destination.join("prepare.resources"),
            "fixture preparation cost\n",
        )?;
        let (inv, _, _) = execute::inventory_with(
            &self.destination,
            &self.suite,
            &self.planned,
            &self.parent_inv,
            self.parent.clone(),
            &self.stage_inv,
            self.stage.clone(),
            true,
        )?;
        Ok(inv)
    }
    pub(super) fn run(&self, inv: &Value, task: &str) -> Result<()> {
        execute::run_with(
            &self.destination,
            task,
            inv,
            &self.parent,
            &self.stage,
            input(task)?,
        )?;
        let seal = execute::replay_with(
            &self.destination,
            task,
            inv,
            &self.parent,
            &self.stage,
            input(task)?,
        )?;
        files(&self.destination, task, true, Some(&seal))
    }
}
