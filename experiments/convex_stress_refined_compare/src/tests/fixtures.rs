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
            1
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
    let task = data::Task {
        vocab: 197,
        selection: "validation".into(),
        splits: BTreeMap::from([
            ("train".into(), rows[..35].to_vec()),
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
    json!({"arms":arms,"complete_parent":quotient,"all35_transformer_endpoints_retained":true,"choose_between_initializers":false,"choices_per_arm":105,
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
            "refined-compare-{}-{}",
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
        let producer = "4f210282986efaa7ce2a9d2c491588ba35a2255f";
        let source_fp = "a".repeat(64);
        let expected = json!({"source_fingerprint":source_fp,"producer_commit":producer,"binary_sha256":digest(&binary)?});
        let source = temp.join("old-source");
        fs::create_dir(&source)?;
        let original_dir = source.join("fixture-0");
        fs::create_dir(&original_dir)?;
        let x = input("fixture-0")?;
        let raw = x.input.as_ref().unwrap();
        let legacy = quotient::binding::run(quotient::binding::Request {
            discovery: quotient::Request {
                input: raw.clone(),
                prime: 3,
            },
            training: [
                quotient::training::Policy {
                    mode: quotient::training::Mode::Full,
                    updates: 1,
                    batch: 13,
                    learning_rate: 1.,
                },
                quotient::training::Policy {
                    mode: quotient::training::Mode::Mini,
                    updates: 1,
                    batch: 13,
                    learning_rate: 0.1,
                },
                quotient::training::Policy {
                    mode: quotient::training::Mode::Forward,
                    updates: 0,
                    batch: 13,
                    learning_rate: 1.,
                },
            ],
            prefix_policy: quotient::row_program::Policy::default(),
            certificate_policy: quotient::certificate::Policy::default(),
            readout: x.readout.clone(),
        })?;
        assert!(legacy.discovery.code.is_some());
        assert_eq!(legacy.candidate_readout_compatible, Some(true));
        let original_file = original_dir.join("stage-01-p3.json");
        write(
            &original_file,
            &json!({"request":legacy.request,"report":legacy,"returned_error":null}),
        )?;
        let source_artifact = parents::evidence(&original_file)?;
        let report = binding::run(&legacy.discovery, x.policies.clone().unwrap())?;
        let stage_root = temp.join("stage");
        fs::create_dir(&stage_root)?;
        fs::create_dir(stage_root.join("fixture-0"))?;
        let mut stages = vec![];
        for (index, prime) in quotient::catalogue().into_iter().enumerate() {
            let file = format!("stage-{index:02}-p{prime}.json");
            let source_sha = if prime == 3 {
                digest(&original_file)?
            } else {
                "b".repeat(64)
            };
            let a = json!({"source_sha256":source_sha,"report":if prime==3{Some(&report)}else{None},"refusal":if prime==3{None}else{Some("fixture no admitted code")},"seconds":0.25});
            write(&stage_root.join("fixture-0").join(&file), &a)?;
            stages.push(json!({"file":file,"sha256":digest(&stage_root.join("fixture-0").join(&file))?,"bytes":fs::metadata(stage_root.join("fixture-0").join(&file))?.len(),
                "source_stage":{"prime":prime,"sha256":source_sha},"retained":{"source_sha256":source_sha,"refusal":a["refusal"],"seconds":0.25}}));
        }
        write(&stage_root.join("fixture-0/input.json"), raw)?;
        write(
            &stage_root.join("fixture-0/policies.json"),
            x.policies.as_ref().unwrap(),
        )?;
        let result = json!({"metadata":{"input":x.record,"source_case":{"artifacts":[source_artifact]}},"input_sha256":digest(&stage_root.join("fixture-0/input.json"))?,"policies_sha256":digest(&stage_root.join("fixture-0/policies.json"))?,"source_refusal":null,"stages":stages,"heldout_evaluation":false,"parent_comparison":false,"unseen_identification":false});
        write(&stage_root.join("fixture-0/result.json"), &result)?;
        let mut artifacts = vec![];
        parents::tree(&stage_root.join("fixture-0"), &mut artifacts)?;
        fs::create_dir(stage_root.join("fixture-1"))?;
        write(
            &stage_root.join("fixture-1/partial.json"),
            &json!({"retained_work":99}),
        )?;
        let partial = parents::evidence(&stage_root.join("fixture-1/partial.json"))?;
        let stage_inv = json!({"source_fingerprint":source_fp,"binary_sha256":digest(&binary)?,"request":{"producer_commit":producer,"discovery":{"root":source,"inventory_sha256":"c".repeat(64)}},"tasks":all,"planned":planned,"protocol":{"synthetic":true}});
        write(&stage_root.join("inventory.json"), &stage_inv)?;
        write(&stage_root.join("request.json"), &stage_inv["request"])?;
        let stage = json!({"source_fingerprint":source_fp,"inventory_sha256":digest(&stage_root.join("inventory.json"))?,"request":stage_inv["request"],"protocol":stage_inv["protocol"],"phase":"training_only","tasks":all,"planned":planned,"heldout_evaluation":false,"parent_comparison":false,"unseen_identification":false,
            "records":[{"task":"fixture-0","status":"complete","result":result,"artifacts":artifacts},{"task":"fixture-1","status":"process_failure","artifacts":[partial]}],
            "constructor_accounting":{"new_discovery_work":0,"charged_once_across_both_arms":true,"complete_constructor_seconds":13.75,"failed_work_retained":true}});
        write(&stage_root.join("summary.json"), &stage)?;
        let parent_root = temp.join("parent");
        fs::create_dir(&parent_root)?;
        let parent_inv = json!({"producer_commit":producer,"source_fingerprint":source_fp,"binary_sha256":digest(&binary)?,"protocol":{"synthetic":true},"tasks":all,"request":{"parent":{"root":source,"inventory_sha256":"c".repeat(64)}}});
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
            parents::Kind::Causal,
            &self.expected_parent,
            &self.all,
            &self.planned,
            true,
        )?;
        parents::load_with(
            &self.request.staging,
            parents::Kind::Refined,
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
