//! Read-only current finite estimates and historical replay after cohort sealing.
//! The dot product below compares numerical estimates, not certified derivatives.
use crate::{
    Result,
    cache::Cache,
    estimate, experiment,
    history::{self, Entry},
    score,
};
use bilinear::{experiment as original, guard};
use serde_json::json;
use std::{fs, path::Path};
pub fn run(workspace: &Path, storage: &Path, cohort: &Path, output: &Path) -> Result<()> {
    guard::admit(512 * 1024 * 1024, workspace, storage)?;
    score::sealed(cohort)?;
    fs::create_dir(output)?;
    for arm in experiment::ARMS {
        for variant in experiment::VARIANTS {
            for mode in experiment::MODES {
                let (p, _, mode_value) = experiment::settings(arm, variant, mode)?;
                let path = experiment::cell(cohort, arm, variant, mode);
                let (model, _, task) = score::load(workspace, storage, &path)?;
                let cache: Cache = serde_json::from_slice(&fs::read(path.join("cache.json"))?)?;
                let history: Vec<Entry> = fs::read_to_string(path.join("history.jsonl"))?
                    .lines()
                    .map(serde_json::from_str)
                    .collect::<std::result::Result<_, _>>()?;
                let initial = experiment::initial(model.parent.clone(), model.block.geometry, &p)?;
                if history.len() != p["updates"].as_u64().ok_or("updates")? as usize + 1
                    || cache.mean.len() != model.block.dimensions()
                {
                    return Err("incomplete historical shape".into());
                }
                history::validate(&history, &initial.block, &model.block)?;
                cache.validate()?;
                let training = &task.splits["train"];
                let indices: Vec<_> = (0..training.len()).collect();
                let records = original::records(training, &indices)?;
                if cache.identities.len() != records.len() || cache.coverage() != records.len() {
                    return Err("incomplete original occurrence coverage".into());
                }
                let h = p["query_step"].as_f64().ok_or("query step")?;
                let mut current = vec![0.; cache.mean.len()];
                let (mut ce, mut min_norm, mut mean_norm) = (0., f64::INFINITY, 0.);
                let (mut historical_queries, mut current_queries, mut offset) =
                    (0usize, 0usize, 0usize);
                for batch in records.chunks(p["batch"].as_u64().ok_or("batch")? as usize) {
                    guard::admit(64 * 1024 * 1024, workspace, storage)?;
                    let bases: Vec<_> = batch
                        .iter()
                        .map(|r| model.parent_logits(r.prefix))
                        .collect::<Result<_>>()?;
                    let estimates =
                        estimate::batch(&model.block, &bases, batch, model.mass(), mode_value, h)?;
                    ce += estimates.mean_ce * batch.len() as f64;
                    current_queries += estimates.ce_evaluations;
                    for (i, (r, base)) in batch.iter().zip(&bases).enumerate() {
                        historical_queries += history::replay_record(
                            &cache,
                            &history,
                            offset + i,
                            r,
                            base,
                            (model.mass(), mode_value, h),
                        )?;
                        for (s, &g) in current.iter_mut().zip(&estimates.values[i]) {
                            *s += g;
                        }
                        let norm = model
                            .block
                            .geometry
                            .state_norm(model.block.atom.state(r.prefix)?);
                        min_norm = min_norm.min(norm);
                        mean_norm += norm;
                    }
                    offset += batch.len();
                }
                for g in &mut current {
                    *g /= records.len() as f64;
                }
                let max_error = current
                    .iter()
                    .zip(&cache.mean)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f64::max);
                let error_norm = current
                    .iter()
                    .zip(&cache.mean)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let dot = current
                    .iter()
                    .zip(&cache.mean)
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                let recompute = cache.error()?;
                if recompute > 1e-10 {
                    return Err("retained finite estimate cache drift".into());
                }
                original::write_json(
                    &output.join(format!("{arm}-{variant}-{mode}.json")),
                    &json!({"source_fingerprint":experiment::fingerprint(),"arm":arm,"variant":variant,"mode":mode,"endpoint_sha256":original::hash(&fs::read(path.join("endpoint.json"))?),"historical_records_replayed":offset,"history_versions":history.len(),"cache_recomputation_error":recompute,"mean_training_ce":ce/records.len() as f64,"current_estimate_sha256":original::hash(&serde_json::to_vec(&current)?),"current_estimate_norm":current.iter().map(|x|x*x).sum::<f64>().sqrt(),"retained_estimate_norm":cache.mean.iter().map(|x|x*x).sum::<f64>().sqrt(),"current_retained_dot":dot,"estimate_error_norm":error_norm,"maximum_estimate_error":max_error,"initial_state_norm":model.block.geometry.state_norm(model.block.atom.initial),"minimum_executed_state_norm":min_norm,"mean_executed_state_norm":mean_norm/records.len() as f64,"current_ce_evaluations":current_queries,"historical_ce_evaluations":historical_queries,"updates_accepted":0,"claim":"finite numerical estimates and history consistency; no stochastic convergence, analytic directional certificate or unseen-answer guarantee"}),
                )?;
                println!(
                    "diagnose {arm}/{variant}/{mode}: {} historical records; estimate error={error_norm:.8}; dot={dot:.8}",
                    offset
                );
            }
        }
    }
    Ok(())
}
