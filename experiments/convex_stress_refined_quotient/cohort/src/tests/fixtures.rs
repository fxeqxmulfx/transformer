use super::*;
use convex_stress_relabel::data;
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
pub(super) struct Fixture {
    pub temp: PathBuf,
    pub suite: Value,
    pub planned: Vec<String>,
    pub request: Request,
    pub destination: PathBuf,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.temp).unwrap();
    }
}
pub(super) fn overwrite<T: Serialize>(path: &Path, value: &T) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
pub(super) fn task() -> data::Task {
    let rows: Vec<_> = (0..36)
        .map(|cell| {
            let (i, j) = (cell % 6, cell / 6);
            data::Example {
                task: "categorical-fixture".into(),
                tokens: vec![j as i64, i as i64],
                targets: vec![-100, (3 + 2 * ((i + 2 * j) % 3) + (i + j) % 2) as i64],
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
        vocab: 11,
        selection: "validation".into(),
        splits: BTreeMap::from([
            ("train".into(), rows[..35].to_vec()),
            ("validation".into(), rows[35..].to_vec()),
        ]),
    }
}
pub(super) fn input(name: &str) -> Result<inputs::Inputs> {
    inputs::assemble(task(), name, 13, json!({"synthetic":true}))
}
pub(super) fn process_files(
    root: &Path,
    task: &str,
    success: bool,
    seal: Option<&Value>,
) -> Result<()> {
    fs::write(
        root.join(format!("{task}.run.exit")),
        if success { "0\n" } else { "9\n" },
    )?;
    for suffix in ["run.resources", "log"] {
        fs::write(
            root.join(format!("{task}.{suffix}")),
            "retained fixture cost\n",
        )?;
    }
    if success {
        fs::write(root.join(format!("{task}.replay.exit")), "0\n")?;
        for suffix in ["replay.resources", "replay.log"] {
            fs::write(
                root.join(format!("{task}.{suffix}")),
                "retained replay cost\n",
            )?;
        }
        overwrite(&root.join(format!("{task}.replay.json")), seal.unwrap());
    }
    Ok(())
}
impl Fixture {
    pub(super) fn new() -> Result<Self> {
        let temp = std::env::temp_dir().join(format!(
            "refined-stage-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir(&temp)?;
        let root = temp.join("source");
        fs::create_dir(&root)?;
        let names: Vec<_> = (0..35).map(|i| format!("fixture-{i}")).collect();
        let planned = names[..2].to_vec();
        let suite = json!({"tasks":names.iter().map(|n|json!({"name":n})).collect::<Vec<_>>()});
        let binary = std::env::current_exe()?;
        let producer = "49105730c6b86570119c36531df1457d7e353933";
        write(
            &root.join("inventory.json"),
            &json!({"source_fingerprint":parent::fingerprint(),
            "binary_sha256":digest(&binary)?,"producer_commit":producer,"tasks":names,"planned":planned,
            "parent_snapshots":{"all35_transformer":true,"both105_arms":true,"depth336":true}}),
        )?;
        let reference = Reference {
            root: root.clone(),
            binary: binary.clone(),
            inventory_sha256: digest(&root.join("inventory.json"))?,
            source_fingerprint: parent::fingerprint(),
            binary_sha256: digest(&binary)?,
            producer_commit: producer.into(),
        };
        let first = &planned[0];
        fs::create_dir(root.join(first))?;
        let x = input(first)?;
        let p: Value = serde_json::from_str(include_str!(
            "../../../../convex_stress_quotient/cohort/original_protocol.json"
        ))?;
        let config = parent::Configuration {
            input: x.input,
            training: serde_json::from_value(p["training"].clone())?,
            prefix: serde_json::from_value(p["prefix"].clone())?,
            certificate: serde_json::from_value(p["certificate"].clone())?,
            readout: Some((0..11).collect()),
        };
        parent::run(config, &root.join(first).join("catalogue"), producer)?;
        let catalogue_replay = parent::replay(&root.join(first).join("catalogue"))?;
        write(
            &root.join(first).join("result.json"),
            &json!({"fixture":true,"has_no_original_metric":true}),
        )?;
        process_files(
            &root,
            first,
            true,
            Some(
                &json!({"task":first,"replayed":true,"source_fingerprint":parent::fingerprint(),
            "result_sha256":digest(&root.join(first).join("result.json"))?,"all_actual_models_and_metrics_replayed":true,
            "all_ce_steps_replayed":true,"catalogue":catalogue_replay}),
            ),
        )?;
        fs::create_dir(root.join(&planned[1]))?;
        write(
            &root.join(&planned[1]).join("partial.json"),
            &json!({"failed_work":17}),
        )?;
        process_files(&root, &planned[1], false, None)?;
        let destination = temp.join("refined");
        let request = Request {
            producer_commit: "c55fe2100ecefdc613400d3f26bcb60c4d0be519".into(),
            suite_sha256: hash(SUITE.as_bytes()),
            discovery: reference,
        };
        let out = Self {
            temp,
            suite,
            planned,
            request,
            destination,
        };
        source::validate_reference(&out.request.discovery, &out.planned, &tasks(&out.suite)?)?;
        Ok(out)
    }
    pub(super) fn prepare(&self) -> Result<Value> {
        let records = self
            .planned
            .iter()
            .map(|t| Ok(input(t)?.record))
            .collect::<Result<Vec<_>>>()?;
        let sources = self
            .planned
            .iter()
            .map(|t| source::case_record(&self.request.discovery, t))
            .collect::<Result<Vec<_>>>()?;
        prepare_with(
            &self.destination,
            self.request.clone(),
            &self.suite,
            &self.planned,
            records,
            sources,
        )?;
        fs::write(
            self.destination.join("prepare.resources"),
            "retained fixture preparation\n",
        )?;
        inventory_with(&self.destination, &self.suite, &self.planned)
    }
    pub(super) fn execute(&self, inv: &Value, task: &str) -> Result<Value> {
        stage::run_with(&self.destination, task, inv, input(task)?)?;
        let seal = stage::replay_with(&self.destination, task, inv, input(task)?)?;
        process_files(&self.destination, task, true, Some(&seal))?;
        summary::case_record(&self.destination, task, inv, input(task)?)
    }
}
