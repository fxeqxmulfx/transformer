//! A run on disk: its frozen configuration, the history of exhaustive evaluations, the held-out
//! predictions and margins of every evaluation, and checkpoints from which it resumes exactly.

use crate::adam::Hyper;
use crate::arith;
use crate::data::{self, Equation};
use crate::layout::Dims;
use crate::mt::Mt;
use crate::train::{Eval, Sampler, Trainer, batch_size, init_seed, initialize, sampler_seed};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The sources the executable was built from, for its fingerprint.
const SOURCES: [(&str, &str); 10] = [
    ("adam.rs", include_str!("adam.rs")),
    ("arith.rs", include_str!("arith.rs")),
    ("data.rs", include_str!("data.rs")),
    ("gemm.rs", include_str!("gemm.rs")),
    ("layout.rs", include_str!("layout.rs")),
    ("main.rs", include_str!("main.rs")),
    ("model.rs", include_str!("model.rs")),
    ("mt.rs", include_str!("mt.rs")),
    ("run.rs", include_str!("run.rs")),
    ("train.rs", include_str!("train.rs")),
];

/// The SHA-256 of the sources, names and contents in order, in hex.
pub fn source_fingerprint() -> String {
    let mut hasher = Sha256::new();
    for (name, text) in SOURCES {
        hasher.update(name.as_bytes());
        hasher.update([0]);
        hasher.update(text.as_bytes());
        hasher.update([0]);
    }
    data::hex(&hasher.finalize())
}

/// What fixes a run.
#[derive(Clone, PartialEq, Debug)]
pub struct Config {
    /// `adam`, the protocol of the paper's Figure 1, or `adamw`, its default.
    pub protocol: String,
    /// The run's seed: of the split, the initialization and the batches.
    pub seed: u32,
    /// The decoupled weight decay.
    pub weight_decay: f64,
    /// Updates to make.
    pub budget: u64,
    /// The supervised positions: 4 and 5 in the source, 4 alone in the paper's text.
    pub sup: Vec<usize>,
    /// Threads of a step, which the result does not depend on: not part of the run's identity.
    pub threads: usize,
    /// Updates between checkpoints.
    pub checkpoint_every: u64,
    /// The update after which every thread of the run flushes subnormal numbers to zero
    /// (`PROTOCOL.md`, Amendment 2); `None` keeps gradual underflow, the source's arithmetic.
    pub flush_from: Option<u64>,
}

impl Config {
    /// The line of the configuration with gradual underflow throughout, as the executable built
    /// from the sources of fingerprint `sources` writes it.
    fn unflushed_line(&self, sources: &str) -> String {
        format!(
            "protocol={} seed={} weight_decay={} budget={} sup={:?} checkpoint_every={} sources={sources}",
            self.protocol, self.seed, self.weight_decay, self.budget, self.sup, self.checkpoint_every
        )
    }

    /// The configuration as one line of text, which `run.json` holds and the checkpoints hash.
    pub fn line(&self) -> String {
        let flush = self.flush_from.map_or(String::new(), |from| format!(" flush_from={from}"));
        self.unflushed_line(&source_fingerprint()) + &flush
    }

    fn hash(&self) -> [u8; 32] {
        Sha256::digest(self.line().as_bytes()).into()
    }

    /// What `run.json` holds for the configuration of line `line` on the split given.
    fn frozen(line: &str, train: &[Equation], heldout: &[Equation]) -> String {
        format!(
            "{{\"config\":\"{line}\",\"train_fingerprint\":\"{}\",\"heldout_fingerprint\":\"{}\",\"train\":{},\
             \"heldout\":{},\"batch\":{},\"parameters\":{}}}\n",
            data::fingerprint(train),
            data::fingerprint(heldout),
            train.len(),
            heldout.len(),
            batch_size(train.len()),
            Dims::PAPER.len()
        )
    }
}

/// Whether the run is evaluated after `step` updates: at the start, after 10, 20 and 50, every
/// 100 up to 10000, every 1000 after, and at the end of the budget.
pub fn evaluated(step: u64, budget: u64) -> bool {
    matches!(step, 0 | 10 | 20 | 50)
        || (step <= 10_000 && step.is_multiple_of(100))
        || step.is_multiple_of(1000)
        || step == budget
}

/// A float as JSON: `null` where it is not finite.
fn num(x: f64) -> String {
    if x.is_finite() { format!("{x}") } else { "null".to_string() }
}

fn eval_fields(name: &str, eval: &Eval) -> String {
    format!(
        "\"{name}_loss\":{},\"{name}_ce_answer\":{},\"{name}_ce_eos\":{},\"{name}_acc\":{},\"{name}_acc_source\":{},\
         \"{name}_errors\":{},\"{name}_margin_min\":{},\"{name}_margin_q05\":{},\"{name}_margin_median\":{}",
        num(eval.loss_source()),
        num(eval.ce_answer),
        num(eval.ce_eos),
        num(eval.accuracy()),
        num(eval.accuracy_source()),
        eval.right.iter().filter(|&&r| !r).count(),
        num(eval.margin_quantile(0.0) as f64),
        num(eval.margin_quantile(0.05) as f64),
        num(eval.margin_quantile(0.5) as f64),
    )
}

/// The bytes of a checkpoint: the configuration's hash, the step, the seconds trained, the
/// parameters and both moments, the sampler, and the SHA-256 of all of it.
pub fn checkpoint_bytes(hash: &[u8; 32], seconds: f64, trainer: &Trainer, sampler: &Sampler) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 * trainer.p.len() + 4 * sampler.perm.len() + 4096);
    out.extend(b"GROKDIV1");
    out.extend(hash);
    out.extend(trainer.step.to_le_bytes());
    out.extend(seconds.to_le_bytes());
    out.extend((trainer.p.len() as u64).to_le_bytes());
    for tensor in [&trainer.p, &trainer.m, &trainer.v] {
        for value in tensor.iter() {
            out.extend(value.to_le_bytes());
        }
    }
    out.extend((sampler.batch as u64).to_le_bytes());
    out.extend((sampler.index as u64).to_le_bytes());
    out.extend((sampler.perm.len() as u64).to_le_bytes());
    for word in sampler.perm.iter().chain(sampler.rng.words().iter()) {
        out.extend(word.to_le_bytes());
    }
    let digest = Sha256::digest(&out);
    out.extend(digest);
    out
}

/// Restores `trainer` and `sampler` from the bytes of a checkpoint of the configuration with
/// `hash`: the seconds trained before it.
pub fn restore(bytes: &[u8], hash: &[u8; 32], trainer: &mut Trainer, sampler: &mut Sampler) -> Result<f64, String> {
    if bytes.len() < 96 || &bytes[..8] != b"GROKDIV1" {
        return Err("not a checkpoint".into());
    }
    let (body, digest) = bytes.split_at(bytes.len() - 32);
    if Sha256::digest(body).as_slice() != digest {
        return Err("the checkpoint's digest does not match its bytes".into());
    }
    if &body[8..40] != hash {
        return Err("the checkpoint is of another configuration or other sources".into());
    }
    let mut at = 40;
    let mut take = |n: usize| {
        let slice = &body[at..at + n];
        at += n;
        slice
    };
    let word = |b: &[u8]| u64::from_le_bytes(b.try_into().expect("eight bytes"));
    trainer.step = word(take(8));
    let seconds = f64::from_le_bytes(take(8).try_into().expect("eight bytes"));
    let len = word(take(8)) as usize;
    if len != trainer.p.len() {
        return Err("the checkpoint is of another model".into());
    }
    for tensor in [&mut trainer.p, &mut trainer.m, &mut trainer.v] {
        for (value, b) in tensor.iter_mut().zip(take(4 * len).chunks_exact(4)) {
            *value = f32::from_le_bytes(b.try_into().expect("four bytes"));
        }
    }
    sampler.batch = word(take(8)) as usize;
    sampler.index = word(take(8)) as usize;
    let n = word(take(8)) as usize;
    let words: Vec<u32> =
        take(4 * (n + 625)).chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().expect("four bytes"))).collect();
    sampler.perm = words[..n].to_vec();
    sampler.rng = Mt::from_words(&words[n..]);
    Ok(seconds)
}

fn checkpoint_path(out: &Path, step: u64) -> PathBuf {
    out.join(format!("checkpoint-{step:07}.bin"))
}

/// The step of the last checkpoint in `out` at or before update `upto`, if it has one.
fn last_checkpoint(out: &Path, upto: u64) -> Option<u64> {
    let entries = fs::read_dir(out).ok()?;
    entries
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|name| name.strip_prefix("checkpoint-")?.strip_suffix(".bin")?.parse().ok())
        .filter(|&step| step <= upto)
        .max()
}

/// The value of the number at `"key":` in a line of the history.
pub fn field(line: &str, key: &str) -> Option<f64> {
    let rest = &line[line.find(&format!("\"{key}\":"))? + key.len() + 3..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn append(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new().create(true).append(true).open(path).map_err(|e| format!("{path:?}: {e}"))?;
    file.write_all(bytes).map_err(|e| format!("{path:?}: {e}"))
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    File::create(&tmp)
        .and_then(|mut f| f.write_all(bytes).and_then(|_| f.sync_all()))
        .map_err(|e| format!("{tmp:?}: {e}"))?;
    fs::rename(&tmp, path).map_err(|e| format!("{path:?}: {e}"))
}

/// The training set and the held-out set of a run, and the optimizer's constants.
pub fn setup(config: &Config) -> (Vec<Equation>, Vec<Equation>) {
    data::split(config.seed, data::train_len(data::P * (data::P - 1), 50.0))
}

/// Starts the run of `config`, which flushes subnormals after update `n = config.flush_from`,
/// in `out` from the last checkpoint at or before `n` of the run in `parent` (`PROTOCOL.md`,
/// Amendment 2). The parent is the run of the same configuration with gradual underflow
/// throughout, by this executable or one built from other sources; up to update `n` the two
/// runs are the same run, so the parent's history and held-out records up to the checkpoint
/// stand for them. The checkpoint is copied under this configuration's hash, and
/// `adopted.json` records where it came from. A run that has begun adopts nothing.
pub fn adopt(config: &Config, parent: &Path, out: &Path) -> Result<(), String> {
    let from = config.flush_from.ok_or("only a run with --flush-from adopts a checkpoint")?;
    if out.join("run.json").exists() {
        return Ok(());
    }
    let step = last_checkpoint(parent, from).ok_or(format!("{parent:?} has no checkpoint up to update {from}"))?;
    let read = |path: PathBuf| fs::read(&path).map_err(|e| format!("{path:?}: {e}"));
    let text = |path: PathBuf| fs::read_to_string(&path).map_err(|e| format!("{path:?}: {e}"));
    let (train, heldout) = setup(config);
    let parent_json = text(parent.join("run.json"))?;
    let sources = parent_json.split("sources=").nth(1).and_then(|rest| rest.get(..64)).unwrap_or("");
    let parent_line = config.unflushed_line(sources);
    if sources.len() != 64 || parent_json != Config::frozen(&parent_line, &train, &heldout) {
        return Err(format!("{parent:?} is not the run of this configuration with gradual underflow:\n{parent_json}"));
    }
    let bytes = read(checkpoint_path(parent, step))?;
    if bytes.len() < 96 || &bytes[..8] != b"GROKDIV1" {
        return Err("the parent's file is not a checkpoint".into());
    }
    let (body, digest) = bytes.split_at(bytes.len() - 32);
    if Sha256::digest(body).as_slice() != digest || body[8..40] != Sha256::digest(parent_line.as_bytes())[..] {
        return Err("the parent's checkpoint is damaged or of another configuration".into());
    }
    let history = text(parent.join("history.jsonl"))?;
    let kept: Vec<&str> = history.lines().filter(|l| field(l, "step").is_some_and(|s| s <= step as f64)).collect();
    if kept.len() != (0..=step).filter(|&s| evaluated(s, config.budget)).count() {
        return Err(format!("the parent's history lacks a record up to update {step}"));
    }
    fs::create_dir_all(out).map_err(|e| format!("{out:?}: {e}"))?;
    write_atomically(&out.join("history.jsonl"), (kept.join("\n") + "\n").as_bytes())?;
    for (name, record) in [("heldout_predictions.bin", heldout.len()), ("heldout_margins.bin", 4 * heldout.len())] {
        let records = read(parent.join(name))?;
        let prefix = records.get(..kept.len() * record).ok_or(format!("the parent's {name} is too short"))?;
        write_atomically(&out.join(name), prefix)?;
    }
    let mut copy = body.to_vec();
    copy[8..40].copy_from_slice(&config.hash());
    let copy_digest = Sha256::digest(&copy);
    copy.extend(copy_digest);
    write_atomically(&checkpoint_path(out, step), &copy)?;
    let record = format!(
        "{{\"parent_config\":\"{parent_line}\",\"step\":{step},\"parent_checkpoint_sha256\":\"{}\"}}\n",
        data::hex(&Sha256::digest(&bytes))
    );
    write_atomically(&out.join("adopted.json"), record.as_bytes())
}

/// Trains the run of `config` in `out` to its budget, from its last checkpoint if it has one.
/// `stop_after` ends the process early, after that many updates of this call, for the tests of
/// resumption.
pub fn run(config: &Config, out: &Path, stop_after: Option<u64>) -> Result<(), String> {
    fs::create_dir_all(out).map_err(|e| format!("{out:?}: {e}"))?;
    let (train, heldout) = setup(config);
    let (dims, hash) = (Dims::PAPER, config.hash());
    let frozen = Config::frozen(&config.line(), &train, &heldout);
    let run_json = out.join("run.json");
    match fs::read_to_string(&run_json) {
        Ok(old) if old != frozen => return Err(format!("{run_json:?} is of another run:\n{old}")),
        Ok(_) => {}
        Err(_) => {
            write_atomically(&run_json, frozen.as_bytes())?;
            let rows: String = heldout.iter().map(|e| format!("{} {} {}\n", e.num, e.den, e.quo)).collect();
            write_atomically(&out.join("heldout.txt"), rows.as_bytes())?;
        }
    }
    let params = initialize(dims, &mut Mt::new(init_seed(config.seed)));
    let hyper = Hyper::source(config.weight_decay);
    let mut trainer = Trainer::new(dims, config.sup.clone(), hyper, params, config.threads);
    let mut sampler = Sampler::new(sampler_seed(config.seed), train.len(), batch_size(train.len()));
    let (history, predictions, margins) =
        (out.join("history.jsonl"), out.join("heldout_predictions.bin"), out.join("heldout_margins.bin"));
    let mut before = 0f64;
    match last_checkpoint(out, u64::MAX) {
        Some(step) => {
            let mut bytes = Vec::new();
            let path = checkpoint_path(out, step);
            File::open(&path).and_then(|mut f| f.read_to_end(&mut bytes)).map_err(|e| format!("{path:?}: {e}"))?;
            before = restore(&bytes, &hash, &mut trainer, &mut sampler)?;
            // What was recorded after the checkpoint is dropped: the run goes on from it alone.
            let text = fs::read_to_string(&history).map_err(|e| format!("{history:?}: {e}"))?;
            let kept: Vec<&str> = text.lines().filter(|l| field(l, "step").is_some_and(|s| s <= step as f64)).collect();
            write_atomically(&history, (kept.join("\n") + "\n").as_bytes())?;
            for (path, record) in [(&predictions, heldout.len()), (&margins, 4 * heldout.len())] {
                let file = OpenOptions::new().write(true).open(path).map_err(|e| format!("{path:?}: {e}"))?;
                file.set_len((kept.len() * record) as u64).map_err(|e| format!("{path:?}: {e}"))?;
            }
            eprintln!("resumed {out:?} from update {step}");
        }
        None => {
            for path in [&history, &predictions, &margins] {
                write_atomically(path, b"")?;
            }
        }
    }
    let started = Instant::now();
    let record = |trainer: &mut Trainer, batch_loss: f64, seconds: f64| -> Result<(), String> {
        let (on_train, on_heldout) = (trainer.evaluate(&train), trainer.evaluate(&heldout));
        let norm = trainer.p.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
        let lr = if trainer.step == 0 { 0.0 } else { trainer.hyper.lr(trainer.step) };
        let line = format!(
            "{{\"step\":{},\"lr\":{},\"batch_loss\":{},{},{},\"param_norm\":{},\"seconds\":{}}}\n",
            trainer.step,
            num(lr),
            num(batch_loss),
            eval_fields("train", &on_train),
            eval_fields("heldout", &on_heldout),
            num(norm),
            num(seconds)
        );
        append(&history, line.as_bytes())?;
        append(&predictions, &on_heldout.predictions)?;
        let raw: Vec<u8> = on_heldout.margins.iter().flat_map(|m| m.to_le_bytes()).collect();
        append(&margins, &raw)?;
        eprintln!(
            "{} {} seed {} update {}: train {:.4} heldout {:.4} loss {:.3e}/{:.3e} ({:.0} s)",
            config.protocol,
            config.weight_decay,
            config.seed,
            trainer.step,
            on_train.accuracy(),
            on_heldout.accuracy(),
            on_train.ce_answer,
            on_heldout.ce_answer,
            seconds
        );
        Ok(())
    };
    if trainer.step == 0 {
        record(&mut trainer, f64::NAN, 0.0)?;
    }
    let (mut made, mut flushing) = (0u64, None);
    while trainer.step < config.budget {
        if stop_after.is_some_and(|limit| made >= limit) {
            return Ok(());
        }
        // The record and the checkpoint of update `flush_from` are made before the switch.
        if flushing.is_none() && config.flush_from.is_some_and(|from| trainer.step >= from) {
            trainer.set_flush(true);
            flushing = Some(arith::Restore);
            if !arith::flushes_subnormals() {
                return Err("the thread did not take the mode that flushes subnormals".into());
            }
            eprintln!("{out:?}: subnormals are flushed after update {}", trainer.step);
        }
        let batch: Vec<Equation> = sampler.next().iter().map(|&i| train[i as usize]).collect();
        let loss = trainer.update(&batch);
        made += 1;
        if !loss.is_finite() {
            return Err(format!("the loss is not finite at update {}", trainer.step));
        }
        let seconds = before + started.elapsed().as_secs_f64();
        if evaluated(trainer.step, config.budget) {
            record(&mut trainer, loss, seconds)?;
        }
        if trainer.step.is_multiple_of(config.checkpoint_every) || trainer.step == config.budget {
            let bytes = checkpoint_bytes(&hash, seconds, &trainer, &sampler);
            write_atomically(&checkpoint_path(out, trainer.step), &bytes)?;
        }
    }
    Ok(())
}

/// The first step of the history at which `test` holds, and the first from which it holds at
/// every later recorded step.
fn onsets(rows: &[(f64, f64)], test: impl Fn(f64) -> bool) -> (Option<f64>, Option<f64>) {
    let first = rows.iter().find(|r| test(r.1)).map(|r| r.0);
    let last_failure = rows.iter().rposition(|r| !test(r.1));
    let sustained = match last_failure {
        None => rows.first().map(|r| r.0),
        Some(i) => rows.get(i + 1).map(|r| r.0),
    };
    (first, sustained)
}

fn opt(x: Option<f64>) -> String {
    x.map_or("null".to_string(), num)
}

/// The summary of the run in `out` as a line of JSON: where training and held-out accuracy
/// first reach 99% and from where they stay there, and the held-out accuracy, errors and
/// margins over the terminal window, the recorded steps of the last tenth of the budget.
pub fn report(out: &Path) -> Result<String, String> {
    let frozen = fs::read_to_string(out.join("run.json")).map_err(|e| format!("{out:?}/run.json: {e}"))?;
    let budget: f64 = frozen
        .split("budget=")
        .nth(1)
        .and_then(|rest| rest.split(' ').next()?.parse().ok())
        .ok_or("run.json has no budget")?;
    let text = fs::read_to_string(out.join("history.jsonl")).map_err(|e| format!("{out:?}/history.jsonl: {e}"))?;
    let column = |key: &str| -> Vec<(f64, f64)> {
        text.lines().filter_map(|l| Some((field(l, "step")?, field(l, key)?))).collect()
    };
    let (train_acc, heldout_acc) = (column("train_acc"), column("heldout_acc"));
    let last = heldout_acc.last().ok_or("the history is empty")?;
    let (train_first, train_sustained) = onsets(&train_acc, |a| a >= 0.99);
    let (heldout_first, heldout_sustained) = onsets(&heldout_acc, |a| a >= 0.99);
    let (_, heldout_exact) = onsets(&heldout_acc, |a| a >= 1.0);
    let window =
        |key: &str| -> Vec<f64> { column(key).into_iter().filter(|r| r.0 >= 0.9 * budget).map(|r| r.1).collect() };
    let (acc, errors, margin) = (window("heldout_acc"), window("heldout_errors"), window("heldout_margin_min"));
    let (ce, train_ce) = (window("heldout_ce_answer"), window("train_ce_answer"));
    let min = |xs: &[f64]| xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = |xs: &[f64]| xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    let seconds = column("seconds").last().map(|r| r.1);
    Ok(format!(
        "{{\"run\":\"{}\",\"config\":\"{}\",\"complete\":{},\"last_step\":{},\"seconds\":{},\
         \"train_acc_99_first\":{},\"train_acc_99_sustained\":{},\"heldout_acc_99_first\":{},\
         \"heldout_acc_99_sustained\":{},\"heldout_acc_100_sustained\":{},\"window_observations\":{},\
         \"window_heldout_acc_min\":{},\"window_heldout_acc_mean\":{},\"window_heldout_acc_max\":{},\
         \"window_heldout_errors_max\":{},\"window_heldout_margin_min\":{},\"window_heldout_ce_max\":{},\
         \"window_train_ce_max\":{},\"final_heldout_acc\":{},\"final_train_acc\":{}}}",
        out.display(),
        frozen.split("\"config\":\"").nth(1).and_then(|r| r.split('"').next()).unwrap_or(""),
        last.0 >= budget,
        last.0,
        opt(seconds),
        opt(train_first),
        opt(train_sustained),
        opt(heldout_first),
        opt(heldout_sustained),
        opt(heldout_exact),
        acc.len(),
        num(min(&acc)),
        num(mean(&acc)),
        num(max(&acc)),
        num(max(&errors)),
        num(min(&margin)),
        num(max(&ce)),
        num(max(&train_ce)),
        num(last.1),
        opt(train_acc.last().map(|r| r.1)),
    ))
}
