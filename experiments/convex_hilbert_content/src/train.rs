//! Original minibatches only; fixed simultaneous new-block updates without selection.
use crate::{Result, cache::Cache, estimate, experiment, history::Entry, source};
use baseline::random::Random;
use bilinear::{experiment as original, guard, model::cross_entropy};
use retained::index::Index;
use serde_json::json;
use std::{fs, io::Write, path::Path, time::Instant};
pub fn run(
    workspace: &Path,
    storage: &Path,
    cohort: &Path,
    arm: &str,
    variant: &str,
    mode: &str,
) -> Result<()> {
    let (p, geometry, mode_value) = experiment::settings(arm, variant, mode)?;
    let (parent, input, task) = source::load(workspace, storage, variant, mode)?;
    let (_, row, _) = original::case_in(
        readouts::source::CASE,
        rotary::experiment::CAUTIOUS_PROTOCOL,
    )?;
    let mut model = experiment::initial(parent, geometry, &p)?;
    let parent_hash = original::hash(&serde_json::to_vec(&model.parent)?);
    let training = &task.splits["train"];
    let index = Index::new(training)?;
    let mut order: Vec<_> = (0..training.len()).collect();
    let (_, all) = index.batch(training, &order)?;
    guard::admit(
        (all.len() * model.block.dimensions() * 32) as u64,
        workspace,
        storage,
    )?;
    let mut cache = Cache::new(&all, model.block.dimensions())?;
    drop(all);
    let path = experiment::cell(cohort, arm, variant, mode);
    fs::create_dir_all(path.parent().ok_or("parent")?)?;
    fs::create_dir(&path)?;
    original::write_json(
        &path.join("metadata.json"),
        &json!({"source_fingerprint":experiment::fingerprint(),"protocol":p,"input":input,"original":row,"selection":task.selection,"arm":arm,"variant":variant,"mode":mode,"objects":model.parent.objects,"budget":model.parent.budget,"components_stored":model.parent.atoms.len()+1,"active_real_coordinates":model.block.dimensions(),"parent_sha256":parent_hash,"binary_sha256":original::hash(&fs::read(std::env::current_exe()?)?)}),
    )?;
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
        serde_json::to_string(&Entry::new(0, &model.block)?)?
    )?;
    let batch = p["batch"].as_u64().ok_or("batch")? as usize;
    let updates = p["updates"].as_u64().ok_or("updates")? as usize;
    let h = p["query_step"].as_f64().ok_or("query step")?;
    let rate = p["learning_rate"].as_f64().ok_or("learning rate")?;
    if row["case"]["batch"] != batch {
        return Err("original batch changed".into());
    }
    let mut rng = Random::new(
        p["shuffle_seed"]
            .as_u64()
            .ok_or("shuffle seed")?
            .try_into()?,
    );
    let (mut cursor, mut examples, mut targets, mut ce_queries, mut logit_queries, mut rises) =
        (order.len(), 0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut pairs, mut changed) = ([0usize; 3], [0usize; 3]);
    let (mut max_excess, mut max_rss) = (0f64, 0u64);
    let start = Instant::now();
    for n in 0..updates {
        if cursor == order.len() {
            rng.shuffle(&mut order);
            cursor = 0;
        }
        let end = (cursor + batch).min(order.len());
        let (ids, records) = index.batch(training, &order[cursor..end])?;
        let rss = guard::admit(64 * 1024 * 1024, workspace, storage)?;
        max_rss = max_rss.max(rss);
        let bases: Vec<_> = records
            .iter()
            .map(|r| model.parent_logits(r.prefix))
            .collect::<Result<_>>()?;
        let estimates =
            estimate::batch(&model.block, &bases, &records, model.mass(), mode_value, h)?;
        cache.observe(&ids, &records, &estimates.values, n)?;
        let before = model.block.coordinates();
        model.block.update(&cache.mean, rate)?;
        let after = model.block.coordinates();
        for (k, (&a, &b)) in before.iter().zip(&after).enumerate() {
            changed[model.block.category(k)?] += usize::from(a.to_bits() != b.to_bits());
        }
        let mut after_ce = 0.;
        let mut logit_error = 0f64;
        for (r, base) in records.iter().zip(&bases) {
            let actual = model.logits(r.prefix)?;
            let cached = crate::model::combine(
                base,
                &model.block,
                model.block.atom.state(r.prefix)?,
                model.mass(),
            )?;
            for (&a, &b) in actual.iter().zip(&cached) {
                logit_error = logit_error.max((a - b).abs());
            }
            after_ce += cross_entropy(&actual, r.target)?;
        }
        after_ce /= records.len() as f64;
        if logit_error > 2e-12 {
            return Err("actual mixed logits disagree".into());
        }
        rises += usize::from(after_ce > estimates.mean_ce + 1e-12);
        max_excess = max_excess.max(model.validate()?);
        examples += end - cursor;
        targets += records.len();
        ce_queries += estimates.ce_evaluations + records.len();
        logit_queries += estimates.logit_queries;
        for (a, &b) in pairs.iter_mut().zip(&estimates.coordinate_pairs) {
            *a += b;
        }
        let indices: Vec<_> = order[cursor..end]
            .iter()
            .flat_map(|&i| (i as u64).to_le_bytes())
            .collect();
        let record_indices: Vec<_> = ids.iter().flat_map(|&i| (i as u64).to_le_bytes()).collect();
        cursor = end;
        writeln!(
            history,
            "{}",
            serde_json::to_string(&Entry::new(n + 1, &model.block)?)?
        )?;
        writeln!(
            trace,
            "{}",
            json!({"update":n+1,"examples_seen":examples,"targets_seen":targets,"batch_indices_sha256":original::hash(&indices),"record_indices_sha256":original::hash(&record_indices),"centre_ce":estimates.mean_ce,"after_ce":after_ce,"coordinate_pairs":estimates.coordinate_pairs,"cache_coverage":cache.coverage(),"retained_estimate_norm":cache.mean.iter().map(|x|x*x).sum::<f64>().sqrt(),"maximum_actual_logit_error":logit_error,"aggregate_rss_before":rss,"elapsed_seconds":start.elapsed().as_secs_f64()})
        )?;
        history.flush()?;
        trace.flush()?;
        if n == 0 || (n + 1) % 64 == 0 {
            println!(
                "{arm}/{variant}/{mode} update={} centre={:.8} after={:.8} coverage={}",
                n + 1,
                estimates.mean_ce,
                after_ce,
                cache.coverage()
            );
        }
    }
    trace.sync_all()?;
    history.sync_all()?;
    let error = cache.error()?;
    if error > 1e-10 || original::hash(&serde_json::to_vec(&model.parent)?) != parent_hash {
        return Err("retained estimate drift or parent mutation".into());
    }
    original::write_json(&path.join("endpoint.json"), &model)?;
    original::write_json(&path.join("cache.json"), &cache)?;
    original::write_json(
        &path.join("training_receipt.json"),
        &json!({"source_fingerprint":experiment::fingerprint(),"arm":arm,"variant":variant,"mode":mode,"updates":updates,"examples_seen":examples,"targets_seen":targets,"epochs":examples as f64/training.len() as f64,"ce_evaluations":ce_queries,"finite_logit_queries":logit_queries,"coordinate_pairs":pairs,"changed_real_coordinates":changed,"batch_ce_rises":rises,"cache_coverage":cache.coverage(),"cache_refreshes":cache.refreshes,"cache_recomputation_error":error,"maximum_operator_norm_excess":max_excess,"maximum_aggregate_rss":max_rss,"parent_sha256":parent_hash,"endpoint_sha256":original::hash(&fs::read(path.join("endpoint.json"))?),"cache_sha256":original::hash(&fs::read(path.join("cache.json"))?),"history_sha256":original::hash(&fs::read(path.join("history.jsonl"))?),"trace_sha256":original::hash(&fs::read(path.join("training.jsonl"))?),"elapsed_seconds":start.elapsed().as_secs_f64(),"status":"sealed","heldout_scored":false}),
    )?;
    Ok(())
}
