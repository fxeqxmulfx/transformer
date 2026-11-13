use super::*;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static SERIAL: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "quotient-catalogue-{}-{}",
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
fn config() -> Configuration {
    // Raw categorical aliases in a six-by-six table; one cell held out.
    let n = 6;
    Configuration {
        input: component::Input {
            n,
            vocabulary: 11,
            observations: (0..n * n - 1)
                .map(|cell| {
                    let (i, j) = (cell % n, cell / n);
                    [cell, 3 + 2 * ((i + 2 * j) % 3) + (i + j) % 2]
                })
                .collect(),
            policy: component::Policy::default(),
        },
        training: [
            training::Policy {
                mode: training::Mode::Full,
                updates: 2,
                batch: 7,
                learning_rate: 1.,
            },
            training::Policy {
                mode: training::Mode::Mini,
                updates: 3,
                batch: 7,
                learning_rate: 0.1,
            },
            training::Policy {
                mode: training::Mode::Forward,
                updates: 0,
                batch: 7,
                learning_rate: 1.,
            },
        ],
        prefix: row_program::Policy::default(),
        certificate: certificate::Policy::default(),
        readout: None,
    }
}
const COMMIT: &str = "3c11a69b7fe452bea9276e4f6de9ebdabe153384";
fn overwrite<T: Serialize>(path: &Path, value: &T) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
#[test]
fn all55_stages_preserve_success_refusals_modes_and_exact_replay() -> Result<()> {
    let temp = Temp::new();
    let root = temp.0.join("run");
    run(config(), &root, COMMIT)?;
    let replayed = replay(&root)?;
    assert_eq!(replayed["stages"], 55);
    // The aliases also carry a two-state quotient. Both proposals are retained;
    // training CE chooses the more informative three-state feature table.
    assert_eq!(replayed["admitted_codes"], 2);
    assert_eq!(replayed["returned_errors"], 0);
    let ledger: Ledger = read(&root.join("ledger.json"))?;
    assert_eq!(ledger.selected_stages, [Some(1); 3]); // prime3, discovered without a task name.
    assert!(ledger.stages.iter().all(|s| s.bytes > 0));
    let accepted: Attempt = read(&root.join(&ledger.stages[1].file))?;
    let report = accepted.report.unwrap();
    assert_eq!(
        report.discovery.code.as_ref().unwrap().names.classes.len(),
        6
    );
    assert_eq!(report.endpoints.len(), 3);
    assert!(report.endpoints.iter().all(|e| e.certificate.certified));
    assert_eq!(report.endpoints[2].training.work.gradient_examples, 0);
    assert!(run(config(), &root, COMMIT).is_err());
    Ok(())
}
#[test]
fn all_returned_errors_and_elimination_caps_are_retained() -> Result<()> {
    let temp = Temp::new();
    let root = temp.0.join("errors");
    let mut bad = config();
    bad.training[2].updates = 1;
    run(bad, &root, COMMIT)?;
    let out = replay(&root)?;
    assert_eq!(out["returned_errors"], 55);
    assert_eq!(out["selected_stages"], json!([null, null, null]));
    let mut capped = config();
    capped.input.policy.elimination.scalar_updates = 0;
    let root = temp.0.join("capped");
    run(capped, &root, COMMIT)?;
    assert_eq!(replay(&root)?["admitted_codes"], 0);
    let ledger: Ledger = read(&root.join("ledger.json"))?;
    assert!(
        ledger
            .stages
            .iter()
            .all(|s| s.retained["discovery_reason"].is_string())
    );
    Ok(())
}
#[test]
fn rehashed_model_changes_missing_stages_source_and_work_are_rejected() -> Result<()> {
    let temp = Temp::new();
    let root = temp.0.join("run");
    run(config(), &root, COMMIT)?;
    let original: Ledger = read(&root.join("ledger.json"))?;
    for slot in 0..6 {
        let mut bad = original.clone();
        match slot {
            0 => {
                bad.stages.pop();
            }
            1 => bad.selected_stages[0] = Some(0),
            2 => bad.metadata["source_fingerprint"] = json!("changed"),
            3 => {
                bad.stages[1].retained["endpoints"][0]["training_work"]["gradient_batches"] =
                    json!(0)
            }
            4 => bad.stages.swap(0, 1),
            _ => bad.seconds_per_stage[0] = -1.,
        }
        overwrite(&root.join("ledger.json"), &bad);
        assert!(replay(&root).is_err(), "mutation{slot}");
    }
    overwrite(&root.join("ledger.json"), &original);
    let path = root.join(&original.stages[1].file);
    let mut attempt: Attempt = read(&path)?;
    attempt.report.as_mut().unwrap().endpoints[0]
        .training
        .model
        .coefficients[0] += 0.1;
    overwrite(&path, &attempt);
    let mut forged = original.clone();
    forged.stages[1] = ledger::stage(&root, 1, &attempt)?;
    overwrite(&root.join("ledger.json"), &forged);
    assert!(replay(&root).is_err()); // Defeats altered state even after rehashing.
    Ok(())
}
#[test]
fn training_ce_ties_ignore_certificates_readouts_and_heldout_fields() -> Result<()> {
    let temp = Temp::new();
    let root = temp.0.join("run");
    run(config(), &root, COMMIT)?;
    let saved: Ledger = read(&root.join("ledger.json"))?;
    let first = saved.stages[1].clone();
    let mut second = first.clone();
    second.retained["candidate_readout_compatible"] = json!(true);
    for endpoint in second.retained["endpoints"].as_array_mut().unwrap() {
        endpoint["code_certified"] = json!(true);
        endpoint["heldout_accuracy"] = json!(1.0);
    }
    assert_eq!(
        ledger::select(&[first.clone(), second.clone()])?,
        [Some(0); 3]
    );
    for endpoint in second.retained["endpoints"].as_array_mut().unwrap() {
        endpoint["train_ce"] = json!(endpoint["train_ce"].as_f64().unwrap() - 0.01);
    }
    assert_eq!(ledger::select(&[first, second])?, [Some(1); 3]);
    Ok(())
}
#[test]
fn serialized_request_hash_and_file_limits_reject_changed_inputs() -> Result<()> {
    let temp = Temp::new();
    let root = temp.0.join("run");
    run(config(), &root, COMMIT)?;
    let mut modified = config();
    modified.input.observations[0][1] = 0;
    overwrite(&root.join("configuration.json"), &modified);
    assert!(replay(&root).is_err());
    let path = temp.0.join("sparse-too-large");
    fs::File::create(&path)?.set_len(8 * 1024 * 1024 * 1024 + 1)?;
    assert!(digest(&path).is_err());
    assert!(read::<Value>(&path).is_err());
    assert!(run(config(), &temp.0.join("bad-commit"), "bad").is_err());
    Ok(())
}
