//! A fixed twelve-endpoint read-only diagnostic; all labels come from training.
use crate::{
    Result, Scan,
    probes::{self, ScoreScan},
    source,
};
use bilinear::{experiment as original, guard};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
pub const PROTOCOL: &str = include_str!("../protocol.json");
pub fn fingerprint() -> String {
    let mut h = Sha256::new();
    for bytes in [
        include_bytes!("lib.rs").as_slice(),
        include_bytes!("source.rs"),
        include_bytes!("probes.rs"),
        include_bytes!("experiment.rs"),
        include_bytes!("main.rs"),
        include_bytes!("../Cargo.toml"),
        include_bytes!("../Cargo.lock"),
        PROTOCOL.as_bytes(),
        source::INPUTS.as_bytes(),
        readouts::experiment::fingerprint().as_bytes(),
    ] {
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
pub fn run(workspace: &Path, storage: &Path, sources: &Path, out: &Path) -> Result<()> {
    guard::admit(0, workspace, storage)?;
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
    let batch = protocol["batch"].as_u64().ok_or("diagnostic batch")? as usize;
    if batch == 0 {
        return Err("zero diagnostic batch".into());
    }
    let etas: Vec<f64> = serde_json::from_value(protocol["etas"].clone())?;
    fs::create_dir(out)?;
    let binary = original::hash(&fs::read(std::env::current_exe()?)?);
    for cohort in ["original-v1", "long-v1", "projected-v1"] {
        for variant in ["cautious", "refined"] {
            for mode in ["mini", "forward"] {
                guard::admit(0, workspace, storage)?;
                let (model, input, task) =
                    source::load(workspace, storage, sources, cohort, variant, mode)?;
                let original_bytes = serde_json::to_vec(&model)?;
                let train = &task.splits["train"];
                if train.len() != 2048 {
                    return Err("original training length changed".into());
                }
                let order: Vec<_> = (0..train.len()).collect();
                let mut scan = Scan::new(&model)?;
                let mut record_hash = Sha256::new();
                let mut batches = Vec::new();
                for indices in order.chunks(batch) {
                    guard::admit(
                        model
                            .reservation(batch)?
                            .checked_mul(4)
                            .ok_or("scan reservation")?,
                        workspace,
                        storage,
                    )?;
                    let records = original::records(train, indices)?;
                    batches.push(records.len());
                    for r in &records {
                        record_hash.update((r.prefix.len() as u64).to_le_bytes());
                        for &o in r.prefix {
                            record_hash.update(o.to_le_bytes());
                        }
                        record_hash.update((r.target as u64).to_le_bytes());
                    }
                    scan.observe(&records)?;
                }
                let analysis = scan.finish()?;
                if analysis.summary.records != 2048 {
                    return Err("supervised record count changed".into());
                }
                let mut trials = Vec::new();
                for &eta in &etas {
                    guard::admit(
                        model
                            .reservation(batch)?
                            .checked_mul(4)
                            .ok_or("probe reservation")?,
                        workspace,
                        storage,
                    )?;
                    let trial = probes::mix(&model, &analysis.best, eta)?;
                    let mut score = ScoreScan::default();
                    for indices in order.chunks(batch) {
                        score.observe(&trial, &original::records(train, indices)?)?;
                    }
                    let score = score.finish()?;
                    if score.records != analysis.summary.records
                        || score.ce + 1e-10 < analysis.summary.approximate_ce_lower_bound
                        || (eta == 0.
                            && (score.ce != analysis.summary.mean_ce
                                || score.correct != analysis.summary.correct))
                    {
                        return Err("actual feasible probe disagrees with diagnostic".into());
                    }
                    trials.push(json!({"eta":eta,"actual":score}));
                }
                if serde_json::to_vec(&model)? != original_bytes
                    || source::load(workspace, storage, sources, cohort, variant, mode)?.1 != input
                {
                    return Err("immutable model or source changed".into());
                }
                let report = json!({"source_fingerprint":fingerprint(),"binary_sha256":binary,
                    "protocol":protocol,"input":input,"training_examples":train.len(),
                    "record_sha256":record_hash.finalize().iter().map(|b|format!("{b:02x}")).collect::<String>(),
                    "batch_record_counts":batches,"summary":analysis.summary,
                    "coefficient_sha256":original::hash(&serde_json::to_vec(&analysis.coefficient)?),
                    "minimizer_sha256":original::hash(&serde_json::to_vec(&analysis.best)?),
                    "probes":trials,"total_ce_evaluations":analysis.summary.ce_evaluations+
                        analysis.summary.records*etas.len(),"accepted_updates":0,
                    "heldout_labels_used":false,"rounding_certificate":false});
                original::write_json(
                    &out.join(format!("{cohort}-{variant}-{mode}.json")),
                    &report,
                )?;
                println!(
                    "{cohort}/{variant}/{mode}: train_ce={:.9} gap={:.9} lower={:.9}",
                    analysis.summary.mean_ce,
                    analysis.summary.approximate_gap,
                    analysis.summary.approximate_ce_lower_bound
                );
            }
        }
    }
    Ok(())
}
