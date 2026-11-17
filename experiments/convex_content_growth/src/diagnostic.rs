//! Replay every retained occurrence at its actual stage and pre-update version.
use crate::{Result, estimate, experiment, model::Model, seal};
use bilinear::{
    experiment as original, guard,
    train::{Mode, Record},
};
use hilbert::{
    cache::Cache,
    history::{self, Entry},
};
use serde_json::json;
use std::{fs, path::Path};
pub fn replay_record(
    cache: &Cache,
    history: &[Entry],
    id: usize,
    r: &Record<'_>,
    base: &[f64],
    query: (f64, Mode, f64),
) -> Result<usize> {
    if cache.identities.get(id) != Some(&retained::index::identity(r)) {
        return Err("historical occurrence changed".into());
    }
    let version = cache.origins[id].ok_or("unobserved history record")?;
    if version + 1 >= history.len() {
        return Err("estimate origin is not a pre-update version".into());
    }
    let e = estimate::batch(
        &history[version].block,
        &[base.to_vec()],
        std::slice::from_ref(r),
        query.0,
        query.1,
        query.2,
        false,
    )?;
    if e.values[0].len() != cache.values[id].len()
        || e.values[0]
            .iter()
            .zip(&cache.values[id])
            .any(|(x, y)| x.to_bits() != y.to_bits())
    {
        return Err("cached estimate disagrees with its actual recorded origin".into());
    }
    Ok(e.full_ce_evaluations + e.cached_ce_evaluations)
}
fn current(
    model: &Model,
    active: usize,
    records: &[Record<'_>],
    cache: &Cache,
    mode: Mode,
    h: f64,
) -> Result<serde_json::Value> {
    let mut mean = vec![0.; cache.mean.len()];
    let mut ce = 0.;
    let mut queries = 0;
    for batch in records.chunks(32) {
        let bases: Vec<_> = batch
            .iter()
            .map(|r| model.base_logits(Some(active), r.prefix))
            .collect::<Result<_>>()?;
        let e = estimate::batch(
            &model.blocks[active].block,
            &bases,
            batch,
            model.blocks[active].weight,
            mode,
            h,
            false,
        )?;
        ce += e.mean_ce * batch.len() as f64;
        queries += e.full_ce_evaluations + e.cached_ce_evaluations;
        for v in e.values {
            for (m, x) in mean.iter_mut().zip(v) {
                *m += x;
            }
        }
    }
    for x in &mut mean {
        *x /= records.len() as f64;
    }
    Ok(
        json!({"mean_training_ce":ce/records.len() as f64,"current_estimate_sha256":original::hash(&serde_json::to_vec(&mean)?),"current_estimate_norm":mean.iter().map(|x|x*x).sum::<f64>().sqrt(),"retained_estimate_norm":cache.mean.iter().map(|x|x*x).sum::<f64>().sqrt(),"current_retained_dot":mean.iter().zip(&cache.mean).map(|(a,b)|a*b).sum::<f64>(),"estimate_error_norm":mean.iter().zip(&cache.mean).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt(),"ce_evaluations":queries,"updates_accepted":0}),
    )
}
pub fn run(workspace: &Path, storage: &Path, cohort: &Path, output: &Path) -> Result<()> {
    seal::all(workspace, storage, cohort)?;
    fs::create_dir(output)?;
    let p = experiment::protocol()?;
    for cell in experiment::cells() {
        let (mut model, _, task) = seal::initial(workspace, storage, cohort, &cell)?;
        let path = cell.path(cohort);
        let training = &task.splits["train"];
        let ids: Vec<_> = (0..training.len()).collect();
        let records = original::records(training, &ids)?;
        let mode = original::mode(&cell.mode)?;
        let h = p["query_step"].as_f64().ok_or("query step")?;
        for n in 1..=p["stages"].as_u64().ok_or("stages")? as usize {
            guard::admit(512 * 1024 * 1024, workspace, storage)?;
            let active = experiment::prepare(&mut model, &cell, n, &p)?;
            let stage = experiment::stage(&path, n);
            let endpoint = seal::transition(&model, &stage, active)?;
            let cache: Cache = serde_json::from_slice(&fs::read(stage.join("cache.json"))?)?;
            let entries: Vec<Entry> = fs::read_to_string(stage.join("history.jsonl"))?
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
            if entries.len() != p["updates_per_stage"].as_u64().ok_or("updates")? as usize + 1
                || cache.mean.len() != model.blocks[active].block.dimensions()
                || cache.identities.len() != records.len()
                || cache.coverage() != records.len()
            {
                return Err("incomplete growth history or occurrence coverage".into());
            }
            history::validate(
                &entries,
                &model.blocks[active].block,
                &endpoint.blocks[active].block,
            )?;
            cache.validate()?;
            let mut queries = 0;
            for (i, r) in records.iter().enumerate() {
                if i.is_multiple_of(32) {
                    guard::admit(64 * 1024 * 1024, workspace, storage)?;
                }
                let base = model.base_logits(Some(active), r.prefix)?;
                queries += replay_record(
                    &cache,
                    &entries,
                    i,
                    r,
                    &base,
                    (model.blocks[active].weight, mode, h),
                )?;
            }
            let error = cache.error()?;
            if error > 1e-10 {
                return Err("retained finite estimate cache drift".into());
            }
            let final_estimate = if n == p["stages"].as_u64().ok_or("stages")? as usize {
                current(&endpoint, active, &records, &cache, mode, h)?
            } else {
                serde_json::Value::Null
            };
            original::write_json(
                &output.join(format!("{}-{n}.json", cell.label())),
                &json!({"source_fingerprint":experiment::fingerprint(),"cell":cell,"stage":n,"endpoint_sha256":original::hash(&fs::read(stage.join("endpoint.json"))?),"historical_records_replayed":records.len(),"history_versions":entries.len(),"cache_recomputation_error":error,"historical_ce_evaluations":queries,"final_current_estimate":final_estimate,"claim":"finite history consistency, not a convergence or unseen-label certificate"}),
            )?;
            model = endpoint;
        }
        println!("{} all eight historical caches replayed", cell.label());
    }
    Ok(())
}
