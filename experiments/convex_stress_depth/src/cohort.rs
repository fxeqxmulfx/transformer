//! Frozen native depth cohort; the original split metrics are used only after
//! training. Depth selection consumes original training CE and a fixed order.
use crate::{
    Result, checkpoint, data,
    features::{Bank, Execution, Transport},
    head::{self, ForwardChoice, Head},
    independent::Independent,
    stream_cache,
};
use convex_stress_hull::{meta, random::Random};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};
mod cache;
pub mod model;
mod replay;
use cache::Observations;
use model::Model;
pub use replay::{replay_cell, summarize};

pub const PROTOCOL: &str = include_str!("../protocol.json");
pub const RETAINED_PROTOCOL: &str = include_str!("../retained_protocol.json");
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RetainedChoice {
    MinimumCe,
    FirstSmallCe,
}
impl RetainedChoice {
    fn label(self) -> &'static str {
        match self {
            Self::MinimumCe => "minimum_ce",
            Self::FirstSmallCe => "first_small_ce",
        }
    }
}
/// CD8 `fixed_margin_ce`: calibrate only proposal choice against a pure
/// margin-24 logit vector. This is not the head family's optimum lower bound.
pub fn pure_proposal_ce(vocab: usize) -> f64 {
    ((vocab - 1) as f64 * (-24f64).exp()).ln_1p()
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub task: String,
    pub depth: usize,
    pub architecture: String,
    pub transport: Transport,
    pub mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retained_initializer: Option<RetainedChoice>,
}
impl Case {
    pub fn initializer_threshold(&self, vocab: usize) -> Option<f64> {
        (self.retained_initializer == Some(RetainedChoice::FirstSmallCe))
            .then(|| pure_proposal_ce(vocab) + 1e-8)
    }
    pub fn id(&self) -> String {
        let base = format!(
            "{}--{}--{}--{:?}--d{}",
            self.mode, self.task, self.architecture, self.transport, self.depth
        );
        match self.retained_initializer {
            Some(choice) => format!("{base}--retained-{}", choice.label()),
            None => base,
        }
    }
    pub fn bank(&self, task: &data::Task) -> Result<Bank> {
        let mut bank = Bank::discover(
            task.vocab,
            self.depth,
            4096,
            self.transport,
            &task.splits["train"],
        )?;
        bank.execution = if self.retained_initializer.is_some() {
            if self.architecture != "joint" {
                return Err("retained comparison only trains combined heads".into());
            }
            Execution::Cumulative
        } else if self.architecture == "additive" {
            Execution::Additive
        } else {
            Execution::Composed
        };
        Ok(bank)
    }
    pub fn allocation(&self, bank: &Bank, n: usize) -> Result<usize> {
        if self.retained_initializer.is_some() {
            // Conservatively allow one example per supervised prefix in the
            // common forecast; the actual streamed index can only be smaller.
            let plan = stream_cache::allocation(bank, n, n)?;
            if plan.cache_bytes_upper_bound > stream_cache::DISK_LIMIT {
                return Err("retained cache exceeds disk cap".into());
            }
            return plan
                .explicit_buffer_bytes
                .checked_add(65536)
                .ok_or_else(|| "retained allocation overflow".into());
        }
        let base = bank.allocation_bytes(n)?;
        // The common initializer temporarily coexists with its copied shared
        // decoder. Count that conversion peak, even though later factors shrink.
        let extra = if self.architecture == "independent" {
            bank.masks.len() * bank.stride() * bank.vocab * 8
        } else {
            0
        };
        base.checked_add(extra)
            .and_then(|b| b.checked_add(65536))
            .ok_or_else(|| "cohort allocation overflow".into())
    }
}
pub fn fingerprint() -> String {
    let sources: &[&[u8]] = &[
        include_bytes!("cohort.rs"),
        include_bytes!("cohort/model.rs"),
        include_bytes!("cohort/cache.rs"),
        include_bytes!("cohort/replay.rs"),
        include_bytes!("../run_cohort.sh"),
        include_bytes!("../run_retained.sh"),
        PROTOCOL.as_bytes(),
        RETAINED_PROTOCOL.as_bytes(),
    ];
    let mut bytes = crate::training::fingerprint().into_bytes();
    for source in sources {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    crate::hash(&bytes)
}
pub fn cases() -> Result<Vec<Case>> {
    cases_for(&serde_json::from_str(PROTOCOL)?)
}
pub fn cases_for(p: &Value) -> Result<Vec<Case>> {
    let retained = *p == serde_json::from_str::<Value>(RETAINED_PROTOCOL)?;
    if !retained && *p != serde_json::from_str::<Value>(PROTOCOL)? {
        return Err("unknown depth comparison protocol".into());
    }
    let mut out = vec![];
    for task in p["tasks"].as_array().unwrap() {
        let depths = if retained {
            vec![p["maximum_depths"][task.as_str().unwrap()].clone()]
        } else {
            p["run_depth_order"].as_array().unwrap().clone()
        };
        let choices: Vec<Option<RetainedChoice>> = if retained {
            serde_json::from_value(p["initializers"].clone())?
        } else {
            vec![None]
        };
        for choice in choices {
            for depth in &depths {
                for architecture in p["architectures"].as_array().unwrap() {
                    for transport in p["transports"].as_array().unwrap() {
                        for mode in p["modes"].as_array().unwrap() {
                            out.push(Case {
                                task: task.as_str().unwrap().into(),
                                depth: depth.as_u64().unwrap() as usize,
                                architecture: architecture.as_str().unwrap().into(),
                                transport: serde_json::from_value(transport.clone())?,
                                mode: mode.as_str().unwrap().into(),
                                retained_initializer: choice,
                            });
                        }
                    }
                }
            }
        }
    }
    if out.len() != if retained { 48 } else { 288 } {
        return Err("depth cohort has an incomplete declared inventory".into());
    }
    Ok(out)
}
pub fn write_json(path: impl AsRef<Path>, value: &Value) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut out, value)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
pub fn read_json(path: impl AsRef<Path>) -> Result<Value> {
    Ok(serde_json::from_reader(File::open(path)?)?)
}
fn commit() -> Result<String> {
    let value = std::env::var("DEPTH_PRODUCER_COMMIT")?;
    if value.len() != 40 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid producer commit".into());
    }
    Ok(value)
}
pub fn prepare(root: &Path) -> Result<()> {
    prepare_with(root, &serde_json::from_str(PROTOCOL)?)
}
pub fn prepare_retained(root: &Path) -> Result<()> {
    prepare_with(root, &serde_json::from_str(RETAINED_PROTOCOL)?)
}
fn prepare_with(root: &Path, protocol: &Value) -> Result<()> {
    let producer = commit()?;
    fs::create_dir(root)?;
    let inventory = cases_for(protocol)?;
    let mut tasks = vec![];
    let mut allocations = vec![];
    let mut projected_disk = 0u64;
    for name in protocol["tasks"].as_array().unwrap() {
        let row = meta::selected(name.as_str().unwrap())?;
        let task = data::build(&row)?;
        let n = task.splits["train"]
            .iter()
            .map(|e| e.targets.iter().filter(|y| **y >= 0).count())
            .sum();
        tasks.push(json!({"name":name,"data_fingerprints":row["data_fingerprints"],"original_train_examples":task.splits["train"].len(),
            "train_prefixes":n,"selection":task.selection,"historical_seed1":row["historical_seed1"]}));
        for case in inventory
            .iter()
            .filter(|c| c.task == name.as_str().unwrap())
        {
            let bank = case.bank(&task)?;
            let bytes = case.allocation(&bank, n)?;
            let disk = if case.retained_initializer.is_some() {
                stream_cache::allocation(&bank, n, task.splits["train"].len())?
                    .cache_bytes_upper_bound
                    + (bank.dimension() * bank.vocab * 8) as u64
                    + 1024 * 1024
            } else {
                0
            };
            projected_disk += disk;
            allocations
                .push(json!({"id":case.id(),"bytes":bytes,"eligible":bytes<=head::BUFFER_LIMIT,"disk_bytes_upper_bound":disk}));
        }
    }
    write_json(
        root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),"producer_commit":producer,
        "protocol":protocol,"cases":inventory,"tasks":tasks,"allocations":allocations,
        "projected_disk_bytes":projected_disk}),
    )
}
pub fn inventory(root: &Path) -> Result<Value> {
    let manifest = read_json(root.join("inventory.json"))?;
    if manifest["source_fingerprint"] != fingerprint()
        || manifest["cases"] != serde_json::to_value(cases_for(&manifest["protocol"])?)?
    {
        return Err("frozen cohort inventory mismatch".into());
    }
    Ok(manifest)
}
pub fn selected_case(root: &Path, id: &str) -> Result<(Value, Case)> {
    let manifest = inventory(root)?;
    let case = cases_for(&manifest["protocol"])?
        .into_iter()
        .find(|c| c.id() == id)
        .ok_or("unknown cohort cell")?;
    Ok((manifest, case))
}

pub fn run_cell(root: &Path, id: &str) -> Result<()> {
    let start = Instant::now();
    let (manifest, case) = selected_case(root, id)?;
    let directory = root.join(id);
    fs::create_dir(&directory)?;
    let row = meta::selected(&case.task)?;
    let task = data::build(&row)?;
    let data_seconds = start.elapsed().as_secs_f64();
    let train = &task.splits["train"];
    let n = train
        .iter()
        .map(|e| e.targets.iter().filter(|y| **y >= 0).count())
        .sum::<usize>();
    let bank = case.bank(&task)?;
    let allocation = case.allocation(&bank, n)?;
    write_json(
        directory.join("metadata.json"),
        &json!({"case":case,"source_fingerprint":fingerprint(),"producer_commit":manifest["producer_commit"],
        "bank":bank,"data_fingerprints":row["data_fingerprints"],"training_prefixes":n,"training_examples":train.len(),
        "explicit_buffer_bytes":allocation,"radius":32.*task.vocab as f64,
        "initializer_ce_threshold":case.initializer_threshold(task.vocab)}),
    )?;
    if allocation > head::BUFFER_LIMIT {
        return write_json(
            directory.join("result.json"),
            &json!({"status":"ineligible_memory_budget","case":case,
            "source_fingerprint":fingerprint(),"explicit_buffer_bytes":allocation,"heldout_scored":false}),
        );
    }
    let stage = Instant::now();
    let cache_path = directory.join("features.bin");
    let mut cache = Observations::build(
        &bank,
        train,
        case.retained_initializer.map(|_| cache_path.as_path()),
    )?;
    let cache_record = if case.retained_initializer.is_some() {
        json!({"bytes":fs::metadata(&cache_path)?.len(),
            "sha256":checkpoint::digest(BufReader::new(File::open(&cache_path)?))?})
    } else {
        Value::Null
    };
    let feature_seconds = stage.elapsed().as_secs_f64();
    let mut tensor = Head::zero(bank, 32. * task.vocab as f64)?;
    let mut scratch = vec![0.; tensor.coefficients.len()];
    let stage = Instant::now();
    let choice = case
        .initializer_threshold(task.vocab)
        .map_or(ForwardChoice::MinimumCe, ForwardChoice::FirstBelow);
    let channel = cache.initialize(&mut tensor, &mut scratch, choice)?;
    let mut model = if case.architecture == "independent" {
        let factor = Independent::from_vertex(&tensor, channel)?;
        drop(tensor);
        drop(scratch);
        scratch = vec![0.; factor.decoder.len()];
        Model::Factor(factor)
    } else {
        Model::Tensor(tensor)
    };
    let initializer_seconds = stage.elapsed().as_secs_f64();
    let order = cache.full_order();
    let initial_ce = model.objective(&mut cache, &order)?;
    let mut ce = initial_ce;
    let mut trace = BufWriter::new(File::create(directory.join("trace.jsonl"))?);
    writeln!(
        trace,
        "{}",
        json!({"update":0,"train_ce":ce,"elapsed_seconds":start.elapsed().as_secs_f64()})
    )?;
    trace.flush()?;
    let mut shuffled = order.clone();
    let mut cursor = order.len();
    let mut rng = Random::new(1);
    let batch_size = row["case"]["batch"]
        .as_u64()
        .ok_or("missing original batch size")? as usize;
    let protocol = &manifest["protocol"];
    let cap = match case.mode.as_str() {
        "full_batch" => protocol["full_batch_updates"].as_u64().unwrap() as usize,
        "minibatch" => protocol["minibatch_updates"].as_u64().unwrap() as usize,
        _ => 0,
    };
    let (
        mut updates,
        mut update_examples,
        mut backward_examples,
        mut backward_prefixes,
        mut line_prefixes,
        mut ce_prefixes,
    ) = (0, 0, 0, 0, 0, n);
    let stage = Instant::now();
    for update in 1..=cap {
        let batch = if case.mode == "full_batch" {
            order.as_slice()
        } else {
            if cursor == shuffled.len() {
                rng.shuffle(&mut shuffled);
                cursor = 0;
            }
            let end = (cursor + batch_size).min(shuffled.len());
            let begin = cursor;
            cursor = end;
            &shuffled[begin..end]
        };
        let observed = cache.prefixes(batch)?;
        let (seen, checks) = model.update(&mut cache, batch, &mut scratch)?;
        updates = update;
        update_examples += batch.len();
        backward_examples += batch.len() * model.gradient_passes();
        backward_prefixes += seen;
        line_prefixes += checks * observed;
        if case.mode == "full_batch" || [1, 16, 64, 256].contains(&update) || update == cap {
            ce = model.objective(&mut cache, &order)?;
            ce_prefixes += n;
            writeln!(
                trace,
                "{}",
                json!({"update":update,"train_ce":ce,"optimizer_backward_prefixes":backward_prefixes,
                "line_search_forward_prefixes":line_prefixes,"elapsed_seconds":start.elapsed().as_secs_f64()})
            )?;
            trace.flush()?;
            if update >= protocol["minimum_gradient_updates"].as_u64().unwrap() as usize
                && ce <= protocol["training_ce_stop"].as_f64().unwrap()
            {
                break;
            }
        }
    }
    let optimization_seconds = stage.elapsed().as_secs_f64();
    let stage = Instant::now();
    let (gap, diagnostic_backward_prefixes) = match &model {
        Model::Tensor(m) if case.mode != "forward_only" => {
            ce = cache.objective(m, &order, Some(&mut scratch))?.0;
            (Some(head::linear_gap(m, &scratch)?), n)
        }
        _ => (None, 0),
    };
    let max_l1 = model.validate()?;
    let diagnostic_seconds = stage.elapsed().as_secs_f64();
    drop(scratch);
    drop(cache);
    let stage = Instant::now();
    let metrics = convex_stress_hull::all_metrics(&model, &task)?;
    let scoring_seconds = stage.elapsed().as_secs_f64();
    if (metrics["train"]["loss"]
        .as_f64()
        .ok_or("missing training metric")?
        - ce)
        .abs()
        > 1e-10
    {
        return Err("cache and original metrics disagree on CE".into());
    }
    let stage = Instant::now();
    model.save(&directory.join("model.bin"))?;
    let hash = checkpoint::digest(BufReader::new(File::open(directory.join("model.bin"))?))?;
    let serialization_seconds = stage.elapsed().as_secs_f64();
    write_json(
        directory.join("result.json"),
        &json!({"status":"complete","case":case,"source_fingerprint":fingerprint(),
        "producer_commit":manifest["producer_commit"],"model_sha256":hash,"feature_cache":cache_record,"initializer_channel":channel,"initializer_train_ce":initial_ce,
        "initializer_ce_threshold":case.initializer_threshold(task.vocab),
        "train_ce":ce,"gradient_updates":updates,"update_examples_seen":update_examples,"optimizer_backward_examples":backward_examples,
        "optimizer_backward_prefixes":backward_prefixes,"line_search_forward_prefixes":line_prefixes,"ce_check_prefixes":ce_prefixes,
        "diagnostic_backward_prefixes":diagnostic_backward_prefixes,"full_gradient_linear_gap":gap,"max_class_l1":max_l1,
        "initializer_examples":train.len(),"initializer_prefixes":n,"parameters":model.parameters(),"paths":model.bank().paths(),
        "primary_split":task.selection,"primary_pass":convex_stress_hull::passes(&metrics[&task.selection]),
        "all_splits_pass":metrics.as_object().unwrap().iter().filter(|(s,_)|s.as_str()!="train").all(|(_,v)|convex_stress_hull::passes(v)),
        "metrics":metrics,"timing_seconds":{"data":data_seconds,"features":feature_seconds,"initializer":initializer_seconds,
        "optimization":optimization_seconds,"final_diagnostic":diagnostic_seconds,"scoring":scoring_seconds,"serialization":serialization_seconds,
        "total":start.elapsed().as_secs_f64()}}),
    )
}
