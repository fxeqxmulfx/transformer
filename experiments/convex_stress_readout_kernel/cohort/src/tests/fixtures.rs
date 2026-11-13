use super::*;
use convex_stress_relabel::data;
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
pub(super) struct Fixture {
    pub root: PathBuf,
    pub suite: Value,
    pub planned: Vec<String>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}
pub(super) fn overwrite<T: Serialize>(path: &Path, value: &T) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
pub(super) fn task() -> data::Task {
    let rows = [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 2], [2, 2, 1]];
    let examples: Vec<_> = rows
        .into_iter()
        .map(|[r, c, label]| data::Example {
            task: "categorical-fixture".into(),
            tokens: vec![c, r],
            targets: vec![-100, label],
            changes: vec![false, true],
            prompt: vec![],
            answer: vec![],
            limit: 0,
            field: None,
            row_index: None,
        })
        .collect();
    data::Task {
        vocab: 7,
        selection: "validation".into(),
        splits: BTreeMap::from([
            ("train".into(), examples.clone()),
            ("validation".into(), vec![examples[0].clone()]),
        ]),
    }
}
pub(super) fn input(name: &str) -> Result<inputs::Inputs> {
    inputs::assemble(task(), name, 2, json!({"synthetic":true}))
}
pub(super) fn process(root: &Path, task: &str, success: bool, seal: Option<&Value>) -> Result<()> {
    fs::write(
        root.join(format!("{task}.run.exit")),
        if success { "0\n" } else { "9\n" },
    )?;
    for suffix in ["run.resources", "log"] {
        fs::write(root.join(format!("{task}.{suffix}")), "fixture cost\n")?;
    }
    if success {
        fs::write(root.join(format!("{task}.replay.exit")), "0\n")?;
        for suffix in ["replay.resources", "replay.log"] {
            fs::write(
                root.join(format!("{task}.{suffix}")),
                "fixture replay cost\n",
            )?;
        }
        overwrite(&root.join(format!("{task}.replay.json")), seal.unwrap());
    }
    Ok(())
}
impl Fixture {
    pub(super) fn new() -> Result<Self> {
        let root = std::env::temp_dir().join(format!(
            "readout-kernel-cohort-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        let names: Vec<_> = (0..35).map(|i| format!("fixture-{i}")).collect();
        let planned = names[..2].to_vec();
        let suite = json!({"tasks":names.iter().map(|n|json!({"name":n})).collect::<Vec<_>>()});
        let request = Request {
            producer_commit: "4bd038160dc870089d2ba3e94c6605f78554aae1".into(),
            suite_sha256: hash(SUITE.as_bytes()),
        };
        let records = planned
            .iter()
            .map(|t| Ok(input(t)?.record))
            .collect::<Result<Vec<_>>>()?;
        prepare_with(&root, request, &suite, &planned, records)?;
        fs::write(root.join("prepare.resources"), "fixture preparation cost\n")?;
        Ok(Self {
            root,
            suite,
            planned,
        })
    }
    pub(super) fn inventory(&self) -> Result<Value> {
        inventory_with(&self.root, &self.suite, &self.planned)
    }
    pub(super) fn run_first(&self) -> Result<()> {
        stage::run_with(
            &self.root,
            &self.planned[0],
            &self.inventory()?,
            input(&self.planned[0])?,
        )
    }
    pub(super) fn replay_first(&self) -> Result<Value> {
        stage::replay_with(
            &self.root,
            &self.planned[0],
            &self.inventory()?,
            input(&self.planned[0])?,
        )
    }
    pub(super) fn close(&self) -> Result<()> {
        self.run_first()?;
        process(
            &self.root,
            &self.planned[0],
            true,
            Some(&self.replay_first()?),
        )?;
        fs::create_dir(self.root.join(&self.planned[1]))?;
        write(
            &self.root.join(&self.planned[1]).join("partial.json"),
            &json!({"work_done":19,"refusal":"fixture process failure"}),
        )?;
        process(&self.root, &self.planned[1], false, None)
    }
    pub(super) fn summary(&self) -> Result<Value> {
        let inv = self.inventory()?;
        let records = self
            .planned
            .iter()
            .map(|t| summary::case_record(&self.root, t, &inv, input(t)?))
            .collect::<Result<Vec<_>>>()?;
        summary::render(&self.root, &inv, records)
    }
}
