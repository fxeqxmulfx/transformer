//! Post-training CM13 interpretation of the frozen CM12 cohort.
//! No result here changes a coding, endpoint, stopping decision or selection.
use convex_stress_budgets::{Result, data, hash};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{fs, path::Path};
mod batch;
pub use batch::audit_all;
pub mod graph;
const PRODUCER: &str = "260d22864ac06e44f945791ac766340a96254b917dffca5a713ea396ddc5a6ba";
const PROTOCOL: &str = include_str!("../../protocol.json");
const SUITE: &str = include_str!("../../../convex_stress/suite.json");
pub fn fingerprint() -> String {
    let mut bytes = vec![];
    for source in [
        include_bytes!("lib.rs").as_slice(),
        include_bytes!("main.rs"),
        include_bytes!("graph.rs"),
        include_bytes!("batch.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        include_bytes!("../../../convex_stress/src/vendor/data.rs"),
        include_bytes!("../../../convex_stress/src/vendor/arithmetic_data.rs"),
        include_bytes!("../../../convex_stress/src/vendor/random.rs"),
        include_bytes!("../../../convex_stress/src/vendor/pcg.rs"),
        include_bytes!("../../../convex_stress/src/vendor/blake.rs"),
        include_bytes!("../../../convex_stress_minibatch/src/lib.rs"),
        PROTOCOL.as_bytes(),
        SUITE.as_bytes(),
    ] {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    hash(&bytes)
}

#[derive(Deserialize)]
struct Coding {
    encode: Vec<usize>,
}
#[derive(Deserialize)]
struct Table {
    vocab: usize,
    radius: f64,
    nuclear_budget: f64,
    probabilities: Vec<f64>,
}
#[derive(Deserialize)]
struct Saved {
    coding: Coding,
    table: Table,
}

pub fn audit(root: &Path, id: &str) -> Result<Value> {
    let rbytes = fs::read(root.join(id).join("result.json"))?;
    let r: Value = serde_json::from_slice(&rbytes)?;
    let seal: Value = serde_json::from_slice(&fs::read(root.join(format!("{id}.replay.json")))?)?;
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let case = &r["case"];
    let task_name = case["task"].as_str().ok_or("missing task")?;
    let mut parts = vec![];
    for (field, allowed) in [
        ("arm", "arms"),
        ("task", "tasks"),
        ("coding", "codings"),
        ("budget", "budgets"),
    ] {
        let value = case[field].as_str().ok_or("missing case field")?;
        if !p[allowed].as_array().unwrap().iter().any(|x| x == value) {
            return Err("unplanned graph audit cell".into());
        }
        parts.push(value);
    }
    if id != parts.join("--")
        || r["source_fingerprint"] != PRODUCER
        || r["protocol"] != p
        || seal["replayed"] != true
        || seal["id"] != id
        || seal["source_fingerprint"] != PRODUCER
        || seal["result_sha256"] != hash(&rbytes)
        || seal["status"] != r["status"]
        || seal["model_sha256"] != r["model_sha256"]
    {
        return Err("unsealed or changed producer record".into());
    }
    let mut result = json!({"id":id,"auditor_fingerprint":fingerprint(),"producer_fingerprint":PRODUCER,"result_sha256":hash(&rbytes),
        "proof_commit":"dade2361","status":r["status"],"changes_training_or_selection":false,"exact_real_arithmetic_certificate":false});
    if r["status"] == "proposal_ineligible" {
        if seal["audit"]["rejected_proposal_reproduced"] != true
            || root.join(id).join("model.json").exists()
        {
            return Err("invalid rejected proposal evidence".into());
        }
        result["reason"] = "no_saved_model_for_rejected_coding".into();
        return Ok(result);
    }
    if r["status"] != "complete" || seal["audit"]["raw_coordinate_audit"] != true {
        return Err("endpoint lacks independent raw audit".into());
    }
    let suite: Value = serde_json::from_str(SUITE)?;
    let row = suite["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == task_name)
        .ok_or("unknown original task")?;
    let task = data::build(row)?;
    let n = task.vocab;
    if !(2..=193).contains(&n)
        || r["vocab"] != n
        || r["data_fingerprints"] != row["data_fingerprints"]
    {
        return Err("data or vocabulary mismatch".into());
    }
    let path = root.join(id).join("model.json");
    if fs::metadata(&path)?.len() > 32 * (n as u64).pow(3) + 1024 * 1024 {
        return Err("saved JSON exceeds graph audit memory forecast".into());
    }
    let bytes = fs::read(path)?;
    if r["model_sha256"] != hash(&bytes) {
        return Err("changed model".into());
    }
    let saved: Saved = serde_json::from_slice(&bytes)?;
    let table = &saved.table;
    if table.vocab != n
        || table.probabilities.len() != n * n * n
        || !table.radius.is_finite()
        || !table.nuclear_budget.is_finite()
        || table
            .probabilities
            .iter()
            .any(|p| !p.is_finite() || *p < -1e-10 || *p > 1. + 1e-10)
        || table
            .probabilities
            .chunks_exact(n)
            .any(|p| (p.iter().sum::<f64>() - 1.).abs() > 1e-10)
        || serde_json::to_value(&saved.coding.encode)? != r["coding"]["encode"]
    {
        return Err("invalid saved coordinates".into());
    }
    let training = pairs(n, &task.splits["train"])?;
    let structure = graph::analyze(n, &training, &saved.coding.encode)?;
    result["training_graph_and_completion_conditions"] = (structure.connected
        && structure.perfect_matching.is_some()
        && structure.observed_phase_consistent)
        .into();
    result["declared_nuclear_bound_le_dimension"] = (table.nuclear_budget <= n as f64).into();
    result["observed_ideal_ce_excess"] = r["ideal_ce_excess"].clone();
    result["saved_numerical_norm_checks"] = r["moment_norms"].clone();
    if let Some(labels) = &structure.completion_labels {
        result["completion_comparison"] = compare(&saved, &task, labels)?;
    }
    result["structure"] = serde_json::to_value(structure)?;
    Ok(result)
}

fn compare(saved: &Saved, task: &data::Task, labels: &[usize]) -> Result<Value> {
    let n = task.vocab;
    let mut max_defect = 0_f64;
    let mut agreeing_cells = 0;
    for (cell, &target) in labels.iter().enumerate() {
        let coordinates = &saved.table.probabilities[cell * n..(cell + 1) * n];
        let mut best = 0;
        for c in 0..n {
            let value = coordinates[saved.coding.encode[c]];
            max_defect = max_defect.max((value - f64::from(c == target)).abs());
            if value
                .total_cmp(&coordinates[saved.coding.encode[best]])
                .is_gt()
            {
                best = c;
            }
        }
        agreeing_cells += usize::from(best == target);
    }
    let mut truth = vec![None; n * n];
    let mut conflicts = 0;
    let mut splits = serde_json::Map::new();
    for (name, examples) in &task.splits {
        let pairs = pairs(n, examples)?;
        if pairs.is_empty() {
            return Err("empty graph comparison split".into());
        }
        let mut correct = 0;
        for pair in &pairs {
            let cell = pair.row + n * pair.column;
            correct += usize::from(labels[cell] == pair.label);
            if truth[cell].is_some_and(|prior| prior != pair.label) {
                conflicts += 1;
            }
            truth[cell] = Some(pair.label);
        }
        splits.insert(name.clone(),json!({"examples":pairs.len(),"completion_correct":correct,"completion_accuracy":correct as f64/pairs.len() as f64}));
    }
    let covered = truth.iter().filter(|label| label.is_some()).count();
    let matches = truth
        .iter()
        .zip(labels)
        .filter(|(label, expected)| label.is_some_and(|label| label == **expected))
        .count();
    Ok(
        json!({"model_vs_completion":{"all_table_cells":n*n,"same_argmax_cells":agreeing_cells,"maximum_probability_defect":max_defect},
        "completion_vs_original_labels":splits,"available_ground_truth_cells":covered,"ground_truth_conflicts":conflicts,"matching_ground_truth_cells":matches,
        "complete_ground_truth_code_structure_checked":covered==n*n && conflicts==0 && matches==covered}),
    )
}

fn pairs(n: usize, examples: &[data::Example]) -> Result<Vec<graph::Pair>> {
    examples
        .iter()
        .map(|e| {
            if e.tokens.len() != 2
                || e.targets.len() != 2
                || e.targets[0] != -100
                || e.targets[1] < 0
                || e.targets[1] as usize >= n
                || !e.prompt.is_empty()
                || e.tokens.iter().any(|t| *t < 0 || *t as usize >= n)
            {
                return Err("unexpected graph audit example".into());
            }
            Ok(graph::Pair {
                row: e.tokens[1] as usize,
                column: e.tokens[0] as usize,
                label: e.targets[1] as usize,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_unique_model_completion_can_disagree_with_unseen_ground_truth() -> Result<()> {
        let example = |row, column, label| data::Example {
            task: "categorical".into(),
            tokens: vec![column, row],
            targets: vec![-100, label],
            changes: vec![false, true],
            prompt: vec![],
            answer: vec![],
            limit: 0,
            field: None,
            row_index: None,
        };
        let mut task = data::Task {
            vocab: 2,
            selection: "validation".into(),
            splits: std::collections::BTreeMap::from([
                (
                    "train".into(),
                    vec![example(0, 0, 1), example(0, 1, 0), example(1, 1, 1)],
                ),
                ("validation".into(), vec![example(1, 0, 1)]),
            ]),
        };
        let saved = Saved {
            coding: Coding { encode: vec![1, 0] },
            table: Table {
                vocab: 2,
                radius: 2.,
                nuclear_budget: 2.,
                probabilities: vec![1., 0., 0., 1., 0., 1., 1., 0.],
            },
        };
        let structure = graph::analyze(2, &pairs(2, &task.splits["train"])?, &saved.coding.encode)?;
        assert!(
            structure.connected
                && structure.perfect_matching.is_some()
                && structure.observed_phase_consistent
        );
        let labels = structure.completion_labels.unwrap();
        let report = compare(&saved, &task, &labels)?;
        assert_eq!(
            report["model_vs_completion"]["maximum_probability_defect"],
            0.
        );
        assert_eq!(report["model_vs_completion"]["same_argmax_cells"], 4);
        assert_eq!(
            report["complete_ground_truth_code_structure_checked"],
            false
        );
        assert_eq!(
            report["completion_vs_original_labels"]["validation"]["completion_correct"],
            0
        );
        task.splits.get_mut("validation").unwrap()[0].targets[1] = 0;
        let report = compare(&saved, &task, &labels)?;
        assert_eq!(report["complete_ground_truth_code_structure_checked"], true);
        Ok(())
    }
}
