//! Synthetic evidence only; never reads a benchmark run or its answers.
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
static SERIAL: AtomicU64 = AtomicU64::new(0);
pub(super) struct Temp(pub PathBuf);
impl Temp {
    pub fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "quotient-original-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
pub(super) fn overwrite<T: Serialize>(path: &Path, value: &T) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
pub(super) fn task(aliases: bool) -> data::Task {
    let n = if aliases { 6 } else { 3 };
    let rows: Vec<_> = (0..n * n)
        .map(|cell| {
            let (i, j) = (cell % n, cell / n);
            let label = if aliases {
                3 + 2 * ((i + 2 * j) % 3) + (i + j) % 2
            } else {
                1 + 2 * ((i + 2 * j) % 3)
            };
            data::Example {
                task: "categorical-fixture".into(),
                tokens: vec![j as i64, i as i64],
                targets: vec![-100, label as i64],
                changes: vec![false, true],
                prompt: vec![],
                answer: vec![],
                limit: 0,
                field: None,
                row_index: None,
            }
        })
        .collect();
    data::Task {
        vocab: if aliases { 11 } else { 7 },
        selection: "validation".into(),
        splits: BTreeMap::from([
            ("train".into(), rows[..n * n - 1].to_vec()),
            ("validation".into(), rows[n * n - 1..].to_vec()),
            ("test".into(), rows),
        ]),
    }
}
pub(super) fn input(name: &str, aliases: bool) -> Result<inputs::Inputs> {
    inputs::assemble(task(aliases), name, json!({"synthetic":true}))
}
fn artifact(path: &Path) -> Result<Value> {
    Ok(json!({"path":path,"sha256":digest(path)?,"bytes":fs::metadata(path)?.len()}))
}
pub(super) struct Fixture {
    pub temp: Temp,
    pub suite: Value,
    pub request: Request,
    pub planned: Vec<String>,
    pub destination: PathBuf,
}
impl Fixture {
    pub fn new() -> Result<Self> {
        let temp = Temp::new();
        let root = temp.0.join("parent");
        fs::create_dir(&root)?;
        let names: Vec<_> = (0..35).map(|i| format!("fixture-{i}")).collect();
        let suite = json!({"tasks":names.iter().map(|n|json!({"name":n})).collect::<Vec<_>>()});
        let mut snapshots = serde_json::Map::new();
        for i in 0..8 {
            let source = temp.0.join(format!("ancestor-{i}.json"));
            write(
                &source,
                &json!({"ancestor":i,"cost":10+i,"failure_retained":i==0}),
            )?;
            let file = format!("ancestor-{i}.json");
            fs::copy(&source, root.join(&file))?;
            snapshots.insert(file, artifact(&source)?);
        }
        let arms:Vec<_>=["minimum_ce","first_small_ce"].iter().map(|policy| {
            let mut choices=vec![];
            for (i,task) in names.iter().enumerate() {
                for mode in ["full_batch","minibatch","forward_only"] {
                    choices.push(if i==1 && mode=="forward_only" {
                        json!({"task":task,"training_mode":mode,"status":"no_model","reference":null,"train_ce":null,
                            "primary_pass":false,"all_splits_pass":false,"reason":"retained fixture refusal"})
                    }else{json!({"task":task,"training_mode":mode,"status":"selected",
                        "reference":format!("prior:{policy}:{task}:{mode}"),"train_ce":1.0,"primary_split":"validation",
                        "metrics":{"train":{"loss":1.0,"sequence_accuracy":0.5},
                            "validation":{"sequence_accuracy":if i==0{0.0}else{1.0}},"test":{"sequence_accuracy":if i==0{0.0}else{1.0}}},
                        "primary_pass":i!=0,"all_splits_pass":i!=0})});
                }
            }
            json!({"initializer":policy,"choices":choices})
        }).collect();
        let p = protocol()?;
        let summary = json!({"source_fingerprint":p["parent_source_fingerprint"],"arms":arms,
            "depth_outcomes":(0..336).map(|i|json!({"slot":i,"refused":i%17==0})).collect::<Vec<_>>(),
            "transformer_endpoints":names.iter().enumerate().map(|(i,n)|json!({"task":n,"capped":i==1,"updates":10+i})).collect::<Vec<_>>(),
            "all_parent_costs_charged_once":true,"choose_between_initializers":false,
            "complete_parent_snapshots":snapshots,"constructor_accounting":{"old_cost":321}});
        write(&root.join("summary.json"), &summary)?;
        write(&root.join("request.json"), &json!({"frozen":true}))?;
        write(&root.join("selection.json"), &json!({"prior":true}))?;
        let binary = temp.0.join("parent-binary");
        fs::write(&binary, b"synthetic frozen binary")?;
        let model = temp.0.join("old-model.json");
        write(&model, &json!({"actual_fixture_state":[1,2,3]}))?;
        let manifest = json!({"source_fingerprint":p["parent_source_fingerprint"],"producer_commit":p["parent_producer_commit"],
            "binary_sha256":digest(&binary)?,"summary_sha256":digest(&root.join("summary.json"))?,
            "request_sha256":digest(&root.join("request.json"))?,"selection_sha256":digest(&root.join("selection.json"))?,
            "complete_parent_snapshots":snapshots,"artifacts":[artifact(&model)?]});
        write(&root.join("manifest.json"), &manifest)?;
        let replay = json!({"source_fingerprint":p["parent_source_fingerprint"],"manifest_sha256":digest(&root.join("manifest.json"))?,
            "summary_sha256":digest(&root.join("summary.json"))?,"replayed":true,"initializer_arms":2,
            "choices_per_arm":105,"depth_outcomes":336,"transformer_endpoints":35});
        write(&root.join("replay.json"), &replay)?;
        let review = temp.0.join("review");
        fs::create_dir(&review)?;
        fs::write(review.join("exit"), "0\n")?;
        fs::write(review.join("commit"), "a".repeat(40))?;
        fs::copy(root.join("replay.json"), review.join("replay.json"))?;
        let request = Request {
            producer_commit: "b".repeat(40),
            suite_sha256: hash(SUITE.as_bytes()),
            parent: Parent {
                summary_sha256: digest(&root.join("summary.json"))?,
                manifest_sha256: digest(&root.join("manifest.json"))?,
                replay_sha256: digest(&root.join("replay.json"))?,
                root,
                binary,
                review: Review {
                    exit: review.join("exit"),
                    commit: review.join("commit"),
                    replay: review.join("replay.json"),
                },
            },
        };
        let destination = temp.0.join("run");
        Ok(Self {
            temp,
            suite,
            request,
            planned: names[..2].to_vec(),
            destination,
        })
    }
    pub fn prepare(&self, aliases: bool) -> Result<(Value, Value)> {
        let records = self
            .planned
            .iter()
            .map(|n| Ok(input(n, aliases)?.record))
            .collect::<Result<_>>()?;
        inventory::prepare_with(
            &self.destination,
            self.request.clone(),
            &self.suite,
            &self.planned,
            records,
        )?;
        inventory::inventory_with(&self.destination, &self.suite, &self.planned, true)
    }
    pub fn successful(&self, inv: &Value, parent: &Value, aliases: bool) -> Result<()> {
        let name = &self.planned[0];
        execute::run_with(
            &self.destination,
            name,
            inv,
            parent,
            input(name, aliases)?,
            std::time::Instant::now(),
        )?;
        let replay =
            execute::replay_with(&self.destination, name, inv, parent, input(name, aliases)?)?;
        write(
            &self.destination.join(format!("{name}.replay.json")),
            &replay,
        )?;
        for suffix in ["run", "replay"] {
            fs::write(
                self.destination.join(format!("{name}.{suffix}.exit")),
                "0\n",
            )?;
            fs::write(
                self.destination.join(format!("{name}.{suffix}.resources")),
                "synthetic resource record: peak KiB1234\n",
            )?;
        }
        Ok(())
    }
    pub fn failed(&self) -> Result<()> {
        let name = &self.planned[1];
        let dir = self.destination.join(name);
        fs::create_dir(&dir)?;
        write(
            &dir.join("partial.json"),
            &json!({"stage":2,"partial_work":12345}),
        )?;
        fs::write(self.destination.join(format!("{name}.run.exit")), "137\n")?;
        fs::write(
            self.destination.join(format!("{name}.run.resources")),
            "peak KiB12345\n",
        )?;
        fs::write(
            self.destination.join(format!("{name}.log")),
            "synthetic allocation failure\n",
        )?;
        Ok(())
    }
}
