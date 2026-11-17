//! Compare conditional domains on every immutable reviewed endpoint, train only.
use crate::{Result, capacity, diagnostic::Scan, experiment};
use bilinear::{experiment as original, guard};
use optimality::{probes::ScoreScan, source};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

pub fn run(workspace: &Path, storage: &Path, sources: &Path, out: &Path) -> Result<()> {
    guard::admit(0, workspace, storage)?;
    let p: Value = serde_json::from_str(experiment::PROTOCOL)?;
    let batch = p["batch"].as_u64().ok_or("batch")? as usize;
    let etas: Vec<f64> = serde_json::from_value(p["diagnostic_etas"].clone())?;
    fs::create_dir(out)?;
    let binary = original::hash(&fs::read(std::env::current_exe()?)?);
    for cohort in ["original-v1", "long-v1", "projected-v1"] {
        for variant in experiment::VARIANTS {
            for mode in experiment::MODES {
                guard::admit(0, workspace, storage)?;
                let (model, input, task) =
                    source::load(workspace, storage, sources, cohort, variant, mode)?;
                let before = serde_json::to_vec(&model)?;
                let train = &task.splits["train"];
                if train.len() != 2048 {
                    return Err("original training length changed".into());
                }
                let order: Vec<_> = (0..train.len()).collect();
                let mut scan = Scan::new(&model)?;
                let mut hash = Sha256::new();
                let mut batches = Vec::new();
                for indices in order.chunks(batch) {
                    guard::admit(
                        model
                            .reservation(batch)?
                            .checked_mul(6)
                            .ok_or("diagnostic reservation")?,
                        workspace,
                        storage,
                    )?;
                    let records = original::records(train, indices)?;
                    batches.push(records.len());
                    for r in &records {
                        hash.update((r.prefix.len() as u64).to_le_bytes());
                        for &o in r.prefix {
                            hash.update(o.to_le_bytes());
                        }
                        hash.update((r.target as u64).to_le_bytes());
                    }
                    scan.observe(&records)?;
                }
                let a = scan.finish()?;
                let lower = a.fixed.mean_ce - a.support.approximate_gap;
                let mut probes = Vec::new();
                for &eta in &etas {
                    guard::admit(
                        model
                            .reservation(batch)?
                            .checked_mul(6)
                            .ok_or("probe reservation")?,
                        workspace,
                        storage,
                    )?;
                    let trial = capacity::mix(&model, &a.best, eta)?;
                    let mut s = ScoreScan::default();
                    for indices in order.chunks(batch) {
                        s.observe(&trial, &original::records(train, indices)?)?;
                    }
                    let s = s.finish()?;
                    if s.records != 2048
                        || s.ce + 1e-9 < lower
                        || (eta == 0. && (s.ce != a.fixed.mean_ce || s.correct != a.fixed.correct))
                    {
                        return Err("actual joint probe disagrees with centre or bound".into());
                    }
                    probes.push(json!({"eta":eta,"actual":s}));
                }
                if serde_json::to_vec(&model)? != before
                    || source::load(workspace, storage, sources, cohort, variant, mode)?.1 != input
                {
                    return Err("immutable diagnostic input changed".into());
                }
                let report = json!({"source_fingerprint":experiment::fingerprint(),"binary_sha256":binary,
                    "protocol":p,"input":input,"fixed_domain":a.fixed,"joint_support":a.support,
                    "approximate_joint_ce_lower_bound":lower,"maximum_actual_logit_error":a.maximum_actual_logit_error,
                    "coefficient_sha256":original::hash(&serde_json::to_vec(&a.coefficient)?),
                    "minimizer_sha256":original::hash(&serde_json::to_vec(&a.best)?),
                    "record_sha256":hash.finalize().iter().map(|b|format!("{b:02x}")).collect::<String>(),
                    "batch_record_counts":batches,"probes":probes,"accepted_updates":0,
                    "heldout_labels_used":false,"rounding_certificate":false,
                    "total_ce_evaluations":(4+etas.len())*a.fixed.records});
                original::write_json(
                    &out.join(format!("{cohort}-{variant}-{mode}.json")),
                    &report,
                )?;
                println!(
                    "{cohort}/{variant}/{mode}: ce={:.9} fixed_gap={:.9} joint_gap={:.9} selected={}",
                    a.fixed.mean_ce,
                    a.fixed.approximate_gap,
                    a.support.approximate_gap,
                    a.support.selected
                );
            }
        }
    }
    Ok(())
}
