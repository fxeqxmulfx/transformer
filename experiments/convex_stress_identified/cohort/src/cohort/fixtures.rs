use super::*;
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
pub(super) fn temporary() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "identified-cohort-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}
pub(super) fn fixture(name: &str) -> Result<Inputs> {
    let n = 8;
    let mut splits = BTreeMap::new();
    for split in ["train", "validation"] {
        let examples = (0..n * n)
            .filter(|&cell| {
                if split == "train" {
                    cell != 11 && cell != 27
                } else {
                    cell == 11 || cell == 27
                }
            })
            .map(|cell| {
                let (i, j) = (cell % n, cell / n);
                let label = if i == 0 || j == 0 {
                    0
                } else {
                    1 + (i + j - 2) % 6
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
pub(super) fn parent_fixture(x: &Inputs, raw: &[u8]) -> Result<Value> {
    let report: search::Report = serde_json::from_slice(raw)?;
    let ce = report.candidates[0].audit["train_ce"].clone();
    Ok(
        json!({"choices":modes()?.iter().map(|mode|json!({"task":x.record["task"],"training_mode":mode,"selection":"validation","tier":"fixture",
        "status":"selected","primary_pass":false,"all_splits_pass":false,"selected":{"id":"old","train_ce":ce,"model_sha256":"wrong-exact-fit",
            "primary_pass":false,"all_splits_pass":false,"metrics":{"train":{"loss":ce},"validation":{"sequence_accuracy":0.}}}})).collect::<Vec<_>>(),
        "records":[{"task":x.record["task"],"result":{"selection":{"report_sha256":hash(raw),"input_sha256":x.record["training_input_sha256"]}}}],
        "complete_parent":{"parents":{"complete_general_comparison":{"parents":{"hull_inventory":{"tasks":[{
            "task":x.record["task"],"vocab":x.task.vocab,"scale":1.0,"data_fingerprints":x.record["data_fingerprints"]}]}}}}}}),
    )
}
pub(super) fn inv(x: &Inputs) -> Value {
    json!({"inputs":[x.record],"producer_commit":"0000000000000000000000000000000000000000",
        "binary_sha256":"fixture","parent_sha256":"fixture-parent"})
}
pub(super) fn full_parent(name: &str, mut partial: Value) -> Result<Value> {
    let original = partial["choices"].as_array().cloned().unwrap_or_default();
    let suite: Value = serde_json::from_str(SUITE)?;
    let mut choices = vec![];
    for task in suite["tasks"].as_array().unwrap() {
        for mode in modes()? {
            if task["name"] == name {
                choices.push(
                    original
                        .iter()
                        .find(|r| r["training_mode"] == mode)
                        .ok_or("missing fixture choice")?
                        .clone(),
                );
            } else {
                choices.push(json!({"task":task["name"],"training_mode":mode,"status":"no_model","selected":null}));
            }
        }
    }
    partial["choices"] = choices.into();
    partial["choices_compared"] = 105.into();
    partial["full35_result"] = true.into();
    parent::validate(&partial)?;
    Ok(partial)
}
