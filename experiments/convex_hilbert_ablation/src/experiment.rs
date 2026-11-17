//! No fitting, selection or target changes: all declared ablations of all endpoints.
use crate::{Result, ablation, source};
use bilinear::{experiment as original, guard};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const KINDS: [&str; 4] = [
    "intact",
    "zero-state",
    "training-mean-state",
    "new-block-only",
];
pub fn fingerprint() -> String {
    let mut h = Sha256::new();
    for b in [
        include_bytes!("lib.rs").as_slice(),
        include_bytes!("ablation.rs"),
        include_bytes!("source.rs"),
        include_bytes!("experiment.rs"),
        include_bytes!("main.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        source::INPUTS.as_bytes(),
        PROTOCOL.as_bytes(),
        hilbert::experiment::fingerprint().as_bytes(),
    ] {
        h.update((b.len() as u64).to_le_bytes());
        h.update(b);
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
pub fn run(workspace: &Path, storage: &Path, cohort: &Path, output: &Path) -> Result<()> {
    let pins = source::verify(workspace, storage, cohort)?;
    fs::create_dir(output)?;
    for input in pins {
        let arm = input["arm"].as_str().ok_or("arm")?;
        let variant = input["variant"].as_str().ok_or("variant")?;
        let mode = input["mode"].as_str().ok_or("mode")?;
        let path = hilbert::experiment::cell(cohort, arm, variant, mode);
        let (model, row, task) = hilbert::score::load(workspace, storage, &path)?;
        let indices: Vec<_> = (0..task.splits["train"].len()).collect();
        let records = original::records(&task.splits["train"], &indices)?;
        let mean = ablation::training_mean(&model, &records)?;
        let reference = original::score::read(&path.join("scores.json"))?;
        let metric = row["case"]["metric"].as_str().ok_or("metric")?;
        let threshold = row["case"]["target"].as_f64().ok_or("threshold")?;
        let mut reports = serde_json::Map::new();
        for kind in KINDS {
            let ablated = ablation::make(&model, &mean, kind)?;
            let mut splits = serde_json::Map::new();
            for (name, examples) in &task.splits {
                guard::admit(64 * 1024 * 1024, workspace, storage)?;
                let mut r = hilbert::score::evaluate(&ablated, examples)?;
                r["raw_and_exact_99"] = json!(
                    r["metrics"]["token_accuracy"].as_f64().ok_or("raw")? >= 0.99
                        && r["metrics"]["sequence_accuracy"].as_f64().ok_or("exact")? >= 0.99
                );
                if kind == "intact" && r != reference["splits"][name] {
                    return Err("intact original raw replay differs".into());
                }
                println!(
                    "{arm}/{variant}/{mode} {kind} {name} raw={} ce={}",
                    r["metrics"]["token_accuracy"], r["metrics"]["loss"]
                );
                splits.insert(name.clone(), r);
            }
            reports.insert(kind.into(),json!({"original_success":splits[&task.selection]["metrics"][metric].as_f64().ok_or("primary")?>=threshold,"raw_and_exact_99":splits[&task.selection]["raw_and_exact_99"],"maximum_norm_excess":ablated.validate()?,"splits":splits}));
        }
        let protocol: Value = serde_json::from_str(PROTOCOL)?;
        original::write_json(
            &output.join(format!("{arm}-{variant}-{mode}.json")),
            &json!({"source_fingerprint":fingerprint(),"binary_sha256":original::hash(&fs::read(std::env::current_exe()?)?),"protocol":protocol,"input":input,"arm":arm,"variant":variant,"mode":mode,"selection":task.selection,"original_metric":metric,"original_threshold":threshold,"training_mean":mean,"ablations":reports,"intact_raw_replay":"identical","updates":0,"claim":"post-hoc diagnostic only; no fitted or selected endpoint"}),
        )?;
    }
    Ok(())
}
