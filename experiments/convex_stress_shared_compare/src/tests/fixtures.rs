use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT: AtomicU64 = AtomicU64::new(0);
pub(super) struct Temp(pub PathBuf);
impl Temp {
    pub fn new() -> Result<Self> {
        let path = std::env::temp_dir().join(format!(
            "shared-compare-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
pub(super) fn overwrite<T: Serialize>(path: &Path, value: &T) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
pub(super) fn reference(root: PathBuf) -> Reference {
    Reference {
        binary: root.join("binary"),
        summary_sha256: "3".repeat(64),
        inventory_sha256: "4".repeat(64),
        review: Review {
            exit: root.join("review.exit"),
            exit_sha256: "5".repeat(64),
            commit: root.join("review.commit"),
            commit_sha256: "6".repeat(64),
            summary: root.join("review.json"),
        },
        root,
    }
}
pub(super) fn parent(all: &[String]) -> Value {
    let arms=POLICIES.iter().map(|policy|{
        let rows=all.iter().flat_map(|task|MODES.iter().map(move|mode|json!({"task":task,"training_mode":mode,"status":"selected",
          "reference":format!("fixture:{task}:{mode}"),"train_ce":4.,"primary_split":"validation",
          "metrics":{"train":{"loss":4.,"sequence_accuracy":0.2},"validation":{"loss":4.,"sequence_accuracy":1.}},
          "primary_pass":true,"all_splits_pass":true}))).collect::<Vec<_>>();
        json!({"initializer":policy,"choices":rows})
    }).collect::<Vec<_>>();
    let layer = json!({"arms":arms,"all35_transformer_endpoints_retained":true,"choose_between_initializers":false,"choices_per_arm":105,
      "constructor_accounting":{"charged_once_across_both_arms":true,"all_prior_costs_retained_in_complete_parent":true}});
    let mut quotient = layer.clone();
    quotient["complete_parent"] = json!({"depth_outcomes":vec![Value::Null;336],"transformer_endpoints":vec![Value::Null;35],"all_parent_costs_charged_once":true});
    let mut causal = layer.clone();
    causal["complete_parent"] = quotient;
    let mut refined = layer;
    refined["complete_parent"] = causal;
    refined
}
pub(super) fn inherited_cases(parent: &Value, all: &[String]) -> Result<Vec<Value>> {
    all.iter().map(|task|{
        let mut rows=vec![];
        for policy in POLICIES{for mode in MODES{
            let old=cell(parent,policy,task,mode)?;
            rows.push(json!({"initializer":policy,"task":task,"training_mode":mode,"prior_reference":old["reference"],
              "selected":old["reference"],"train_ce":old["train_ce"],"transformation":{"status":"unchanged"}}));
        }}
        Ok(json!({"task":task,"status":"complete","result":{"seal":{"choices":{"choices":rows}},"outcomes":[]}}))
    }).collect()
}
pub(super) fn example(row: usize, column: usize, label: usize) -> data::Example {
    data::Example {
        task: "synthetic".into(),
        tokens: vec![column as i64, row as i64],
        targets: vec![-100, label as i64],
        changes: vec![false, true],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    }
}
pub(super) fn task() -> data::Task {
    let train = [
        [1, 1, 2],
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [2, 1, 0],
        [1, 2, 0],
        [3, 3, 3],
    ]
    .into_iter()
    .map(|[r, c, y]| example(r, c, y))
    .collect::<Vec<_>>();
    data::Task {
        vocab: 4,
        selection: "validation".into(),
        splits: BTreeMap::from([
            ("train".into(), train),
            ("validation".into(), vec![example(3, 0, 0)]),
        ]),
    }
}
