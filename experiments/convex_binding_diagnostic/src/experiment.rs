//! Fixed read-only report for every declared input; no optimization or selection.
use crate::{Result, diagnose, source};
use bilinear::{experiment as original, guard};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub fn fingerprint() -> String {
    let mut h = Sha256::new();
    for b in [
        include_bytes!("lib.rs").as_slice(),
        include_bytes!("counts.rs"),
        include_bytes!("diagnose.rs"),
        include_bytes!("experiment.rs"),
        include_bytes!("source.rs"),
        include_bytes!("main.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        source::INPUTS.as_bytes(),
        PROTOCOL.as_bytes(),
        growth::experiment::fingerprint().as_bytes(),
    ] {
        h.update((b.len() as u64).to_le_bytes());
        h.update(b);
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
pub fn run(workspace: &Path, storage: &Path, cohort: &Path, output: &Path) -> Result<()> {
    let pins = source::verify(workspace, storage, cohort)?;
    fs::create_dir(output)?;
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    for pin in pins {
        guard::admit(256 * 1024 * 1024, workspace, storage)?;
        let (model, row, task, reference) = source::load(workspace, storage, cohort, &pin)?;
        let mut splits = serde_json::Map::new();
        for (name, examples) in &task.splits {
            guard::admit(64 * 1024 * 1024, workspace, storage)?;
            let report = diagnose::split(&model, examples)?;
            if report["original"] != reference["splits"][name] {
                return Err("binding original raw replay differs".into());
            }
            println!(
                "{} {name}: raw={} stored={} pair-choice={}",
                source::label(&pin)?,
                report["overall"]["original_accuracy"],
                report["overall"]["stored_value_prediction_rate"],
                report["overall"]["two_value_choice_accuracy"]
            );
            splits.insert(name.clone(), report);
        }
        let metric = row["case"]["metric"].as_str().ok_or("metric")?;
        let threshold = row["case"]["target"].as_f64().ok_or("threshold")?;
        let success = splits[&task.selection]["original"]["metrics"][metric]
            .as_f64()
            .ok_or("selection metric")?
            >= threshold;
        if reference["original_success"] != success
            || reference["raw_and_exact_99"]
                != splits[&task.selection]["original"]["raw_and_exact_99"]
        {
            return Err("original binding success flag changed".into());
        }
        original::write_json(
            &output.join(format!("{}.json", source::label(&pin)?)),
            &json!({"source_fingerprint":fingerprint(),"binary_sha256":original::hash(&fs::read(std::env::current_exe()?)?),"protocol":protocol,"input":pin,"selection":task.selection,"original_success":success,"raw_and_exact_99":reference["raw_and_exact_99"],"splits":splits,"updates":0,"intact_raw_replay":"identical","claim":"post-hoc original-example error localization; auxiliary choice is not original accuracy or a new success"}),
        )?;
    }
    Ok(())
}
