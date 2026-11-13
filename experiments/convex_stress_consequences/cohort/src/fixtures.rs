use super::*;
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
pub(super) fn temporary() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "consequences-cohort-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}
pub(super) fn fixture(name: &str) -> Result<Inputs> {
    let n = 9;
    let mut splits = BTreeMap::new();
    for split in ["train", "validation"] {
        let examples = (0..n * n)
            .filter(|&cell| {
                if split == "train" {
                    cell != 70
                } else {
                    cell == 70
                }
            })
            .map(|cell| {
                let (i, j) = (cell % n, cell / n);
                let label = if i == 8 || j == 8 {
                    0
                } else {
                    1 + (2 * i + j) % 8
                };
                data::Example {
                    task: name.into(),
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
        splits.insert(split.into(), examples);
    }
    inputs::assemble(
        data::Task {
            splits,
            vocab: n,
            selection: "validation".into(),
        },
        name,
        json!({"train":"a","validation":"b"}),
    )
}
pub(super) fn parents(x: &Inputs) -> Result<Vec<Value>> {
    let ce = (std::f64::consts::E + x.task.vocab as f64 - 1.).ln() - 1.;
    Ok(modes()?.iter().map(|mode| json!({"task":x.record["task"],"training_mode":mode,"selection":"validation","tier":"fixture",
        "status":"selected","primary_pass":false,"all_splits_pass":false,"selected":{"id":"old","train_ce":ce,
            "model_sha256":"old-exact-fit","primary_pass":false,"all_splits_pass":false,
            "metrics":{"train":{"loss":ce},"validation":{"sequence_accuracy":0.}}}})).collect())
}
pub(super) fn inv(x: &Inputs) -> Value {
    json!({"inputs":[x.record],"producer_commit":"0000000000000000000000000000000000000000",
        "binary_sha256":"fixture","parent_sha256":"fixture-parent"})
}
pub(super) fn full_parent(x: Option<&Inputs>) -> Result<Value> {
    let original = x.map(parents).transpose()?.unwrap_or_default();
    let suite: Value = serde_json::from_str(SUITE)?;
    let mut choices = vec![];
    for task in suite["tasks"].as_array().unwrap() {
        for mode in modes()? {
            if x.is_some_and(|x| x.record["task"] == task["name"]) {
                choices.push(
                    original
                        .iter()
                        .find(|r| r["training_mode"] == mode)
                        .unwrap()
                        .clone(),
                );
            } else {
                choices.push(json!({"task":task["name"],"training_mode":mode,"status":"no_model","selected":null}));
            }
        }
    }
    let records: Vec<_> = x
        .into_iter()
        .map(|x| json!({"task":x.record["task"],"result":{"metadata":{"input":x.record}}}))
        .collect();
    let parent =
        json!({"choices":choices,"choices_compared":105,"full35_result":true,"records":records});
    parent::validate(&parent)?;
    Ok(parent)
}
pub(super) fn statuses(root: &Path, success: Option<&str>) -> Result<()> {
    for name in tasks()? {
        fs::write(
            root.join(format!("{name}.run.exit")),
            if success == Some(&name) { "0\n" } else { "1\n" },
        )?;
        fs::write(root.join(format!("{name}.log")), "fixture outcome\n")?;
        if success == Some(&name) {
            fs::write(root.join(format!("{name}.replay.exit")), "0\n")?;
        }
    }
    Ok(())
}
