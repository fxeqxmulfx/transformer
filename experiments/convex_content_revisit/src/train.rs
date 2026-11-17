//! Fixed512-update matched continuation, using only the original training split.
use crate::{Result, cache::Cache, content, experiment, history::Entry, source, update};
use baseline::random::Random;
use bilinear::{experiment as original, guard};
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
    let (p, config, content_config, revisit) = experiment::settings(arm, variant, mode)?;
    let (mut model, parent, input, task) = source::load(workspace, storage, variant, mode)?;
    let (_, row, _) = original::case_in(
        readouts::source::CASE,
        rotary::experiment::CAUTIOUS_PROTOCOL,
    )?;
    let initial_content = readouts::source::content_hash(&model)?;
    let training = &task.splits["train"];
    let index = Index::new(training)?;
    let mut order: Vec<_> = (0..training.len()).collect();
    let (_, records) = index.batch(training, &order)?;
    guard::admit(
        model
            .reservation(records.len())?
            .checked_mul(8)
            .ok_or("state cache reservation")?,
        workspace,
        storage,
    )?;
    let mut cache = Cache::from_parent(&model, &records, parent)?;
    drop(records);
    let path = experiment::cell(cohort, arm, variant, mode);
    fs::create_dir_all(path.parent().ok_or("parent")?)?;
    fs::create_dir(&path)?;
    original::write_json(
        &path.join("metadata.json"),
        &json!({"source_fingerprint":experiment::fingerprint(),
        "protocol":p,"input":input,"original":row,"objects":model.objects,"budget":model.budget,
        "arm":arm,"variant":variant,"mode":mode,"selection":task.selection,"initial_content_sha256":initial_content,
        "binary_sha256":original::hash(&fs::read(std::env::current_exe()?)?)}),
    )?;
    let mut trace = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path.join("training.jsonl"))?;
    let mut history = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path.join("history.jsonl"))?;
    let batch = p["batch"].as_u64().ok_or("batch")? as usize;
    let updates = p["updates"].as_u64().ok_or("updates")? as usize;
    if row["case"]["batch"] != batch {
        return Err("original batch size changed".into());
    }
    let mut rng = Random::new(
        p["shuffle_seed"]
            .as_u64()
            .ok_or("shuffle seed")?
            .try_into()?,
    );
    let (mut cursor, mut examples, mut targets, mut accepted, mut content_changes) =
        (order.len(), 0usize, 0usize, 0usize, 0usize);
    let (mut head_queries, mut content_queries, mut bias_queries, mut line_queries) =
        (0usize, 0usize, 0usize, 0usize);
    let mut selected = vec![0usize; model.atoms.len()];
    let mut content_visits = selected.clone();
    let (mut trials, mut accepted_coordinates) = ([0usize; 3], [0usize; 3]);
    let start = Instant::now();
    for n in 0..updates {
        if cursor == order.len() {
            rng.shuffle(&mut order);
            cursor = 0;
        }
        let end = (cursor + batch).min(order.len());
        let (ids, records) = index.batch(training, &order[cursor..end])?;
        let rss = guard::admit(
            model
                .reservation(records.len())?
                .checked_mul(8)
                .ok_or("update reservation")?,
            workspace,
            storage,
        )?;
        let content_receipt = if revisit {
            let j = 1 + n % (model.atoms.len() - 1);
            content_visits[j] += 1;
            Some(content::step(&mut model, &records, j, content_config)?)
        } else {
            None
        };
        let changed = content_receipt
            .as_ref()
            .filter(|r| r.content_changed)
            .map(|r| r.component);
        let entry = Entry {
            version: n + 1,
            component: changed,
            atom: changed.map(|j| model.atoms[j].clone()),
            content_sha256: readouts::source::content_hash(&model)?,
        };
        writeln!(history, "{}", serde_json::to_string(&entry)?)?;
        history.flush()?;
        if let Some(r) = &content_receipt {
            content_queries += r.total_ce_evaluations;
            content_changes += usize::from(r.content_changed);
            for k in 0..3 {
                trials[k] += r.coordinate_trials[k];
                accepted_coordinates[k] += r.accepted_coordinates[k];
            }
        }
        let receipt = update::step(&mut model, &mut cache, &ids, &records, config, n + 1)?;
        examples += end - cursor;
        targets += records.len();
        accepted += usize::from(receipt.eta > 0.);
        head_queries += receipt.total_ce_evaluations;
        bias_queries += receipt.bias_loss_queries;
        line_queries += receipt.line_loss_queries;
        selected[receipt.selected_component] += 1;
        let indices: Vec<_> = order[cursor..end]
            .iter()
            .flat_map(|&i| (i as u64).to_le_bytes())
            .collect();
        let record_indices: Vec<_> = ids.iter().flat_map(|&i| (i as u64).to_le_bytes()).collect();
        cursor = end;
        writeln!(
            trace,
            "{}",
            json!({"update":n+1,"examples_seen":examples,"targets_seen":targets,
            "batch_indices_sha256":original::hash(&indices),"record_indices_sha256":original::hash(&record_indices),
            "cache_coverage":cache.coverage(),"aggregate_rss_before":rss,"elapsed_seconds":start.elapsed().as_secs_f64(),
            "content":content_receipt,"readout":receipt})
        )?;
        trace.flush()?;
        if n == 0 || (n + 1) % 64 == 0 {
            println!(
                "{arm}/{variant}/{mode} update={} centre={:.8} after={:.8} eta={:.6} content_changes={}",
                n + 1,
                receipt.centre_ce,
                receipt.after_ce,
                receipt.eta,
                content_changes
            );
        }
    }
    history.sync_all()?;
    trace.sync_all()?;
    let content = readouts::source::content_hash(&model)?;
    if !revisit && content != initial_content {
        return Err("control changed learned content".into());
    }
    cache.validate(&model)?;
    let drift = cache.recomputation_error(&model)?;
    if drift > 1e-10 {
        return Err("historical product cache drift".into());
    }
    original::write_json(&path.join("endpoint.json"), &model)?;
    original::write_json(&path.join("cache.json"), &cache)?;
    original::write_json(
        &path.join("training_receipt.json"),
        &json!({"source_fingerprint":experiment::fingerprint(),
        "arm":arm,"variant":variant,"mode":mode,"updates":updates,"accepted_readout_updates":accepted,
        "examples_seen":examples,"targets_seen":targets,"epochs":examples as f64/training.len() as f64,
        "total_ce_evaluations":head_queries+content_queries,"readout_ce_evaluations":head_queries,
        "content_ce_evaluations":content_queries,"bias_loss_queries":bias_queries,"line_loss_queries":line_queries,
        "content_changes":content_changes,"content_coordinate_trials":trials,"accepted_content_coordinates":accepted_coordinates,
        "components_stored":model.atoms.len(),"content_component_visits":content_visits,"selected_readout_components":selected,
        "initial_content_sha256":initial_content,"content_sha256":content,"cache_coverage":cache.coverage(),
        "cache_refreshes":cache.refreshes,"starting_refreshes":cache.starting_refreshes,"cache_recomputation_error":drift,
        "elapsed_seconds":start.elapsed().as_secs_f64(),"endpoint_sha256":original::hash(&fs::read(path.join("endpoint.json"))?),
        "cache_sha256":original::hash(&fs::read(path.join("cache.json"))?),
        "history_sha256":original::hash(&fs::read(path.join("history.jsonl"))?),"status":"sealed","heldout_scored":false}),
    )?;
    Ok(())
}
