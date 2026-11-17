//! Fresh per-occurrence vector cache for one complete active block.
use crate::{
    Result, estimate,
    experiment::{self, Cell},
    model::Model,
};
use baseline::{data::Example, random::Random};
use bilinear::{experiment as original, guard, model::cross_entropy};
use hilbert::{cache::Cache, history::Entry, model::combine};
use retained::index::Index;
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path, time::Instant};
pub struct Context<'a> {
    pub workspace: &'a Path,
    pub storage: &'a Path,
    pub path: &'a Path,
    pub cell: &'a Cell,
    pub p: &'a Value,
    pub stage: usize,
    pub active: usize,
    pub fingerprint: &'a str,
}
pub fn run(model: &mut Model, training: &[Example], ctx: &Context<'_>) -> Result<()> {
    let p = ctx.p;
    let path = experiment::stage(ctx.path, ctx.stage);
    fs::create_dir(&path)?;
    let base_hash = model.base_hash(ctx.active)?;
    let initial_hash = original::hash(&serde_json::to_vec(model)?);
    let mass = model.blocks[ctx.active].weight;
    let index = Index::new(training)?;
    let mut order: Vec<_> = (0..training.len()).collect();
    let (_, all) = index.batch(training, &order)?;
    guard::admit(
        (all.len() * model.blocks[ctx.active].block.dimensions() * 32) as u64,
        ctx.workspace,
        ctx.storage,
    )?;
    let mut cache = Cache::new(&all, model.blocks[ctx.active].block.dimensions())?;
    drop(all);
    let mut trace = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path.join("training.jsonl"))?;
    let mut history = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path.join("history.jsonl"))?;
    writeln!(
        history,
        "{}",
        serde_json::to_string(&Entry::new(0, &model.blocks[ctx.active].block)?)?
    )?;
    let updates = p["updates_per_stage"].as_u64().ok_or("updates")? as usize;
    let batch = p["batch"].as_u64().ok_or("batch")? as usize;
    let h = p["query_step"].as_f64().ok_or("query step")?;
    let rate = p["learning_rate"].as_f64().ok_or("rate")?;
    let mode = original::mode(&ctx.cell.mode)?;
    let mut rng = Random::new(
        (p["shuffle_seed_base"].as_u64().ok_or("shuffle seed")? + ctx.stage as u64).try_into()?,
    );
    let (
        mut cursor,
        mut examples,
        mut targets,
        mut full,
        mut cached,
        mut queries,
        mut rises,
        mut audits,
    ) = (
        order.len(),
        0usize,
        0usize,
        0usize,
        0usize,
        0usize,
        0usize,
        0usize,
    );
    let (mut pairs, mut changed) = ([0usize; 3], [0usize; 3]);
    let (mut max_excess, mut max_ce_error, mut max_logit_error, mut max_rss) =
        (0f64, 0f64, 0f64, 0u64);
    let start = Instant::now();
    for n in 0..updates {
        if cursor == order.len() {
            rng.shuffle(&mut order);
            cursor = 0;
        }
        let end = (cursor + batch).min(order.len());
        let (ids, records) = index.batch(training, &order[cursor..end])?;
        max_rss = max_rss.max(guard::admit(64 * 1024 * 1024, ctx.workspace, ctx.storage)?);
        let bases: Vec<_> = records
            .iter()
            .map(|r| model.base_logits(Some(ctx.active), r.prefix))
            .collect::<Result<_>>()?;
        let estimates = estimate::batch(
            &model.blocks[ctx.active].block,
            &bases,
            &records,
            mass,
            mode,
            h,
            true,
        )?;
        cache.observe(&ids, &records, &estimates.values, n)?;
        let before = model.blocks[ctx.active].block.coordinates();
        model.blocks[ctx.active].block.update(&cache.mean, rate)?;
        let block = &model.blocks[ctx.active].block;
        for (k, (&a, b)) in before.iter().zip(block.coordinates()).enumerate() {
            changed[block.category(k)?] += usize::from(a.to_bits() != b.to_bits());
        }
        let mut after_ce = 0.;
        let mut error = 0f64;
        for (r, base) in records.iter().zip(&bases) {
            let actual = model.logits(r.prefix)?;
            let combined = combine(base, block, block.atom.state(r.prefix)?, mass)?;
            for (a, b) in actual.iter().zip(combined) {
                error = error.max((a - b).abs());
            }
            after_ce += cross_entropy(&actual, r.target)?;
        }
        after_ce /= records.len() as f64;
        if error > 2e-12 {
            return Err("actual complete mixed logits disagree".into());
        }
        max_logit_error = max_logit_error.max(error);
        max_ce_error = max_ce_error.max(estimates.maximum_cached_ce_error);
        max_excess = max_excess.max(model.validate()?);
        rises += usize::from(after_ce > estimates.mean_ce + 1e-12);
        examples += end - cursor;
        targets += records.len();
        full += estimates.full_ce_evaluations + records.len();
        cached += estimates.cached_ce_evaluations;
        queries += estimates.finite_logit_queries;
        audits += estimates.audit_comparisons;
        for (a, b) in pairs.iter_mut().zip(estimates.coordinate_pairs) {
            *a += b;
        }
        let indices: Vec<_> = order[cursor..end]
            .iter()
            .flat_map(|&i| (i as u64).to_le_bytes())
            .collect();
        let record_ids: Vec<_> = ids.iter().flat_map(|&i| (i as u64).to_le_bytes()).collect();
        cursor = end;
        writeln!(
            history,
            "{}",
            serde_json::to_string(&Entry::new(n + 1, block)?)?
        )?;
        writeln!(
            trace,
            "{}",
            json!({"update":n+1,"batch_indices_sha256":original::hash(&indices),"record_indices_sha256":original::hash(&record_ids),"centre_ce":estimates.mean_ce,"after_ce":after_ce,"coordinate_pairs":estimates.coordinate_pairs,"cache_coverage":cache.coverage(),"maximum_actual_logit_error":error,"maximum_cached_ce_error":estimates.maximum_cached_ce_error,"audit_comparisons":estimates.audit_comparisons})
        )?;
    }
    trace.sync_all()?;
    history.sync_all()?;
    let error = cache.error()?;
    if error > 1e-10 || model.base_hash(ctx.active)? != base_hash {
        return Err("retained estimate drift or held mixture mutation".into());
    }
    original::write_json(&path.join("endpoint.json"), model)?;
    original::write_json(&path.join("cache.json"), &cache)?;
    original::write_json(
        &path.join("training_receipt.json"),
        &json!({"source_fingerprint":ctx.fingerprint,"cell":ctx.cell,"stage":ctx.stage,"active":ctx.active,"active_mass":mass,"initial_sha256":initial_hash,"base_sha256":base_hash,"final_sha256":original::hash(&serde_json::to_vec(model)?),"updates":updates,"examples_seen":examples,"targets_seen":targets,"full_ce_evaluations":full,"cached_ce_evaluations":cached,"finite_logit_queries":queries,"coordinate_pairs":pairs,"changed_real_coordinates":changed,"batch_ce_rises":rises,"cache_coverage":cache.coverage(),"cache_refreshes":cache.refreshes,"cache_recomputation_error":error,"maximum_operator_norm_excess":max_excess,"maximum_aggregate_rss":max_rss,"maximum_actual_logit_error":max_logit_error,"maximum_cached_ce_error":max_ce_error,"audit_comparisons":audits,"endpoint_sha256":original::hash(&fs::read(path.join("endpoint.json"))?),"cache_sha256":original::hash(&fs::read(path.join("cache.json"))?),"history_sha256":original::hash(&fs::read(path.join("history.jsonl"))?),"trace_sha256":original::hash(&fs::read(path.join("training.jsonl"))?),"elapsed_seconds":start.elapsed().as_secs_f64(),"status":"sealed","heldout_scored":false}),
    )?;
    println!(
        "{} stage={} updates={updates} elapsed={:.2}s query_error={max_ce_error:.3e}",
        ctx.cell.label(),
        ctx.stage,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
