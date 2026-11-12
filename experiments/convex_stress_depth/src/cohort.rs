//! Frozen native depth cohort; the original split metrics are used only after
//! training. Depth selection consumes original training CE and a fixed order.
use crate::{
    Result, checkpoint, data,
    features::{Bank, Execution, Transport},
    head::{self, Cache, Head},
    independent::Independent,
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
pub mod model;
mod replay;
use model::Model;
pub use replay::{replay_cell, summarize};

pub const PROTOCOL: &str = include_str!("../protocol.json");
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub task: String,
    pub depth: usize,
    pub architecture: String,
    pub transport: Transport,
    pub mode: String,
}
impl Case {
    pub fn id(&self) -> String {
        format!(
            "{}--{}--{}--{:?}--d{}",
            self.mode, self.task, self.architecture, self.transport, self.depth
        )
    }
    pub fn bank(&self, task: &data::Task) -> Result<Bank> {
        let mut bank = Bank::discover(
            task.vocab,
            self.depth,
            4096,
            self.transport,
            &task.splits["train"],
        )?;
        bank.execution = if self.architecture == "additive" {
            Execution::Additive
        } else {
            Execution::Composed
        };
        Ok(bank)
    }
    pub fn allocation(&self, bank: &Bank, n: usize) -> Result<usize> {
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
        include_bytes!("cohort/replay.rs"),
        include_bytes!("../run_cohort.sh"),
        PROTOCOL.as_bytes(),
    ];
    let mut bytes = crate::training::fingerprint().into_bytes();
    for source in sources {
        bytes.extend_from_slice(&(source.len() as u64).to_le_bytes());
        bytes.extend_from_slice(source);
    }
    crate::hash(&bytes)
}
pub fn cases() -> Result<Vec<Case>> {
    let p: Value = serde_json::from_str(PROTOCOL)?;
    let mut out = vec![];
    for task in p["tasks"].as_array().unwrap() {
        for depth in p["run_depth_order"].as_array().unwrap() {
            for architecture in p["architectures"].as_array().unwrap() {
                for transport in p["transports"].as_array().unwrap() {
                    for mode in p["modes"].as_array().unwrap() {
                        out.push(Case {
                            task: task.as_str().unwrap().into(),
                            depth: depth.as_u64().unwrap() as usize,
                            architecture: architecture.as_str().unwrap().into(),
                            transport: serde_json::from_value(transport.clone())?,
                            mode: mode.as_str().unwrap().into(),
                        });
                    }
                }
            }
        }
    }
    if out.len() != 288 {
        return Err("depth cohort must retain all 288 cells".into());
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
    let producer = commit()?;
    fs::create_dir(root)?;
    let inventory = cases()?;
    let mut tasks = vec![];
    let mut allocations = vec![];
    for name in serde_json::from_str::<Value>(PROTOCOL)?["tasks"]
        .as_array()
        .unwrap()
    {
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
            allocations
                .push(json!({"id":case.id(),"bytes":bytes,"eligible":bytes<=head::BUFFER_LIMIT}));
        }
    }
    write_json(
        root.join("inventory.json"),
        &json!({"source_fingerprint":fingerprint(),"producer_commit":producer,
        "protocol":serde_json::from_str::<Value>(PROTOCOL)?,"cases":inventory,"tasks":tasks,"allocations":allocations}),
    )
}
pub fn inventory(root: &Path) -> Result<Value> {
    let manifest = read_json(root.join("inventory.json"))?;
    if manifest["source_fingerprint"] != fingerprint()
        || manifest["protocol"] != serde_json::from_str::<Value>(PROTOCOL)?
        || manifest["cases"] != serde_json::to_value(cases()?)?
    {
        return Err("frozen cohort inventory mismatch".into());
    }
    Ok(manifest)
}
pub fn selected_case(root: &Path, id: &str) -> Result<(Value, Case)> {
    let manifest = inventory(root)?;
    let case = cases()?
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
        "explicit_buffer_bytes":allocation,"radius":32.*task.vocab as f64}),
    )?;
    if allocation > head::BUFFER_LIMIT {
        return write_json(
            directory.join("result.json"),
            &json!({"status":"ineligible_memory_budget","case":case,
            "source_fingerprint":fingerprint(),"explicit_buffer_bytes":allocation,"heldout_scored":false}),
        );
    }
    let stage = Instant::now();
    let cache = Cache::build(&bank, train)?;
    let feature_seconds = stage.elapsed().as_secs_f64();
    let mut tensor = Head::zero(bank, 32. * task.vocab as f64)?;
    let mut scratch = vec![0.; tensor.coefficients.len()];
    let stage = Instant::now();
    let channel = head::forward_initialize(&mut tensor, &cache, &mut scratch)?;
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
    let initial_ce = model.objective(&cache, &order)?;
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
    let protocol: Value = serde_json::from_str(PROTOCOL)?;
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
        let observed: usize = batch
            .iter()
            .map(|e| cache.example_offsets[e + 1] - cache.example_offsets[*e])
            .sum();
        let (seen, checks) = model.update(&cache, batch, &mut scratch)?;
        updates = update;
        update_examples += batch.len();
        backward_examples += batch.len() * model.gradient_passes();
        backward_prefixes += seen;
        line_prefixes += checks * observed;
        if case.mode == "full_batch" || [1, 16, 64, 256].contains(&update) || update == cap {
            ce = model.objective(&cache, &order)?;
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
            ce = head::objective(m, &cache, &order, Some(&mut scratch))?.0;
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
        "producer_commit":manifest["producer_commit"],"model_sha256":hash,"initializer_channel":channel,"initializer_train_ce":initial_ce,
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
