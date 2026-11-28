//! Eight-bit modular addition on a 16-by-16 staircase. Only the observed
//! split is passed to training. Held-out logits are scored on the host.
use crate::{
    KERNITE_VERSION, Result,
    engine::{
        Engine,
        adamw::{FIRST_NOISE_STDDEV, RatePlan, wsd},
        page::{Observations, PageRule},
    },
    page_runs::{Optimizer, Rule},
    suite::{Shape, load_weights, publish, save_weights},
};
use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    path::Path,
    time::Instant,
};

const LENGTH: usize = 8;
const RUNS: usize = 8;
const UPDATES: usize = 20_480;
#[derive(Clone, Copy)]
pub(crate) enum Method {
    Page(PageRule),
    AdamW,
    AdamWFirstNoise,
}
impl Method {
    fn protocol_id(self) -> &'static str {
        match self {
            Self::Page(PageRule::Single) => "argand-sum8-adaptive-page-v1",
            Self::Page(PageRule::SquareRoot) => "argand-sum8-adaptive-page-sqrt-v1",
            Self::Page(PageRule::Retry) => "argand-sum8-adaptive-page-retry-v1",
            Self::AdamW => "argand-sum8-adamw-wsd-v1",
            Self::AdamWFirstNoise => "argand-sum8-adamw-first-noise-v1",
        }
    }
    fn rule(self, sampling_seed: u32) -> Rule {
        match self {
            Self::Page(correction) => Rule::Page(correction),
            Self::AdamW => Rule::AdamW(RatePlan::Wsd),
            Self::AdamWFirstNoise => Rule::AdamWFirstNoise {
                seed: sampling_seed,
            },
        }
    }
}

struct Split {
    tokens: Vec<u32>,
    labels: Vec<u32>,
}

impl Split {
    fn new(observed: bool) -> Self {
        let mut split = Self {
            tokens: Vec::new(),
            labels: Vec::new(),
        };
        for word in 0..256u32 {
            let (row, col) = (word >> 4, word & 15);
            if (col == row || col == row + 1) == observed {
                split
                    .tokens
                    .extend((0..LENGTH).rev().map(|bit| (word >> bit) & 1));
                split.labels.extend([u32::MAX; LENGTH - 1]);
                split.labels.push(word.count_ones() % 2);
            }
        }
        split
    }

    fn observations(&self) -> Observations<'_> {
        Observations {
            tokens: &self.tokens,
            labels: &self.labels,
            length: LENGTH,
        }
    }

    fn score(&self, engine: &mut Engine) -> Result<Value> {
        let mut correct = 0;
        let mut ties = 0;
        let mut min_gap = f64::INFINITY;
        let mut ce = 0.0;
        let chunk = LENGTH * engine.config.max_batch;
        let masked = vec![u32::MAX; chunk];
        for (inputs, labels) in self.tokens.chunks(chunk).zip(self.labels.chunks(chunk)) {
            engine.forward(
                inputs,
                &masked[..inputs.len()],
                inputs.len() / LENGTH,
                LENGTH,
            )?;
            let logits = engine
                .logits
                .part(..2 * inputs.len())
                .ok_or("logit capacity")?
                .read()?;
            for (word, y) in logits
                .as_chunks::<{ 2 * LENGTH }>()
                .0
                .iter()
                .zip(labels.as_chunks::<LENGTH>().0)
            {
                let y = y[LENGTH - 1] as usize;
                let z = &word[2 * (LENGTH - 1)..];
                let gap = f64::from(z[y]) - f64::from(z[1 - y]);
                if !gap.is_finite() {
                    return Err("nonfinite evaluation logit".into());
                }
                correct += usize::from(gap > 0.0);
                ties += usize::from(gap == 0.0);
                min_gap = min_gap.min(gap);
                ce += (-gap).max(0.0) + (-gap.abs()).exp().ln_1p();
            }
        }
        let total = self.tokens.len() / LENGTH;
        Ok(
            json!({"correct": correct, "total": total, "ties": ties, "min_gap": min_gap,
            "ce": ce / total as f64, "all_strict": correct == total}),
        )
    }
}

pub(crate) fn new_engine(seed: u32) -> Result<Engine> {
    Engine::new(
        Shape {
            width: 64,
            heads: 4,
            layers: 6,
            vocab: 2,
            max_length: LENGTH,
            max_batch: 32,
        },
        1,
        seed,
    )
}

/// The diagonal and superdiagonal are the only oracle data read by the
/// phase certificate. Each label comes from the actual observed split.
pub(crate) fn observed_phase_labels() -> ([f64; 16], [f64; 15]) {
    let seen = Split::new(true);
    let (mut diagonal, mut edge) = ([0.0; 16], [0.0; 15]);
    for (bits, row) in seen
        .tokens
        .as_chunks::<LENGTH>()
        .0
        .iter()
        .zip(seen.labels.as_chunks::<LENGTH>().0)
    {
        let id = bits.iter().fold(0, |a, b| (a << 1) | *b) as usize;
        let sign = 1.0 - 2.0 * f64::from(row[LENGTH - 1]);
        if id / 16 == id % 16 {
            diagonal[id / 16] = sign;
        } else {
            edge[id / 16] = sign;
        }
    }
    (diagonal, edge)
}

fn protocol(updates: usize, seeds: [[u32; 2]; RUNS], method: Method) -> Value {
    let mut plan = json!({"protocol":method.protocol_id(), "kernite":KERNITE_VERSION, "updates":updates, "seeds":seeds,
        "seed_order":["native_initialization","page_sampling"], "width":64,"heads":4,"ffn_width":256,
        "layers":6,"repeat":1,"vocab":2,"sigma":0.02,"open_words":31,"hidden_words":225,
        "mask":"big-endian nibbles; col == row or col == row + 1, without wrap",
        "success":"positive terminal true-class logit gap on all 225 hidden words"});
    if matches!(method, Method::AdamW | Method::AdamWFirstNoise) {
        let second = if matches!(method, Method::AdamWFirstNoise) {
            plan["gaussian_weight_noise"] = json!({"after_update":1,"stddev":FIRST_NOISE_STDDEV});
            "first_update_noise"
        } else {
            "unused"
        };
        plan["seed_order"] = json!(["native_initialization", second]);
        plan["adamw"] = RatePlan::Wsd.settings(wsd(updates));
    }
    plan
}

fn manifest(
    dir: &Path,
    requested: Option<usize>,
    replay_only: bool,
    method: Method,
) -> Result<(usize, Value)> {
    if requested == Some(0) {
        return Err("expected a positive observation horizon".into());
    }
    let path = dir.join("protocol.json");
    if path.try_exists()? {
        let prior: Value = serde_json::from_slice(&fs::read(path)?)?;
        let updates: usize = serde_json::from_value(prior["updates"].clone())?;
        let seeds = serde_json::from_value(prior["seeds"].clone())?;
        if updates == 0
            || requested.is_some_and(|n| n != updates)
            || prior != protocol(updates, seeds, method)
        {
            return Err("existing directory contains a different experiment".into());
        }
        return Ok((updates, prior));
    }
    if replay_only {
        return Err("missing run to replay".into());
    }
    let updates = requested.unwrap_or(UPDATES);
    let seeds = match method {
        Method::AdamW | Method::AdamWFirstNoise => {
            let paired: Value = serde_json::from_str(include_str!("../sum8-page-protocol.json"))?;
            serde_json::from_value(paired["seeds"].clone())?
        }
        Method::Page(_) => {
            let mut random = fs::File::open("/dev/urandom")?;
            let mut seeds = [[0; 2]; RUNS];
            for pair in &mut seeds {
                let mut bytes = [0; 8];
                random.read_exact(&mut bytes)?;
                *pair = [
                    u32::from_le_bytes(bytes[..4].try_into()?),
                    u32::from_le_bytes(bytes[4..].try_into()?),
                ];
            }
            seeds
        }
    };
    let plan = protocol(updates, seeds, method);
    publish(dir, "protocol.json", &serde_json::to_vec_pretty(&plan)?)?;
    Ok((updates, plan))
}

fn read_seed(pair: &Value, index: usize) -> Result<u32> {
    Ok(u32::try_from(pair[index].as_u64().ok_or("missing seed")?)?)
}

fn train(
    dir: &Path,
    pair: &Value,
    updates: usize,
    observed: &Split,
    hidden: &Split,
    method: Method,
) -> Result<()> {
    let seed = read_seed(pair, 0)?;
    let sampling_seed = read_seed(pair, 1)?;
    let mut model = new_engine(seed)?;
    let initial =
        json!({"observed":observed.score(&mut model)?, "hidden":hidden.score(&mut model)?});
    let mut optimizer = Optimizer::new(
        &mut model,
        method.rule(sampling_seed),
        observed.observations(),
        sampling_seed,
        wsd(updates),
    )?;
    let mut history = fs::File::create(dir.join("training.jsonl"))?;
    writeln!(history, "{}", optimizer.metrics())?;
    let start = Instant::now();
    let mut reason = "budget";
    let mut milestones = Vec::new();
    for t in 1..=updates {
        if !optimizer.step(&mut model, observed.observations(), false)? {
            reason = "no_resolvable_full_step";
            break;
        }
        if t.is_power_of_two() || t.is_multiple_of(256) || t == updates {
            writeln!(history, "{}", optimizer.metrics())?;
            history.flush()?;
        }
        // Capture weights on the same uninterrupted trajectory. Intermediate
        // predictions are evaluated only after the terminal weights are saved.
        if t < updates && ((t <= 512 && t.is_power_of_two()) || t.is_multiple_of(1024)) {
            let (_, digest) = save_weights(dir, &format!("weights-{t}.f32"), &model)?;
            milestones.push(json!({"step":t,"weights_sha256":digest}));
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    let (parameters, digest) = save_weights(dir, "weights.f32", &model)?;
    let terminal_train = observed.score(&mut model)?;
    let terminal_hidden = hidden.score(&mut model)?;
    for point in &mut milestones {
        load_weights(
            dir,
            &format!("weights-{}.f32", point["step"]),
            &point["weights_sha256"],
            &mut model,
        )?;
        point["observed"] = observed.score(&mut model)?;
        point["hidden"] = hidden.score(&mut model)?;
    }
    let record = json!({"protocol":method.protocol_id(),"seeds":pair,"updates":updates,"gpu":model.cuda.name(),
        "parameters":parameters,"seconds":seconds,"stop":reason,"initial":initial,"milestones":milestones,
        "optimizer":optimizer.metrics(),"observed":terminal_train,"hidden":terminal_hidden,"weights_sha256":digest});
    publish(dir, "result.json", &serde_json::to_vec_pretty(&record)?)
}

fn replay(
    dir: &Path,
    pair: &Value,
    updates: usize,
    observed: &Split,
    hidden: &Split,
    method: Method,
) -> Result<Value> {
    let record: Value = serde_json::from_slice(&fs::read(dir.join("result.json"))?)?;
    if record["protocol"] != method.protocol_id()
        || record["seeds"] != *pair
        || record["updates"] != updates
    {
        return Err("invalid experiment checkpoint".into());
    }
    let mut model = new_engine(read_seed(pair, 0)?)?;
    let mut verify = |file: &str, point: &Value| -> Result<()> {
        load_weights(dir, file, &point["weights_sha256"], &mut model)?;
        for (name, split) in [("observed", observed), ("hidden", hidden)] {
            if split.score(&mut model)? != point[name] {
                return Err(format!("replay of {file} disagrees on the {name} split").into());
            }
        }
        Ok(())
    };
    verify("weights.f32", &record)?;
    for point in record["milestones"].as_array().into_iter().flatten() {
        let step = point["step"].as_u64().ok_or("invalid saved step")?;
        verify(&format!("weights-{step}.f32"), point)?;
    }
    Ok(record)
}

pub fn run(dir: &Path, requested: Option<usize>, replay_only: bool, method: Method) -> Result<()> {
    fs::create_dir_all(dir)?;
    let lock = fs::File::options()
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join("run.lock"))?;
    lock.lock()?;
    let (updates, plan) = manifest(dir, requested, replay_only, method)?;
    let (observed, hidden) = (Split::new(true), Split::new(false));
    let mut records = Vec::new();
    for (i, pair) in plan["seeds"]
        .as_array()
        .ok_or("missing run seeds")?
        .iter()
        .enumerate()
    {
        let cell = dir.join(format!("draw-{i}"));
        fs::create_dir_all(&cell)?;
        if !replay_only && !cell.join("result.json").try_exists()? {
            train(&cell, pair, updates, &observed, &hidden, method)?;
        }
        let record = replay(&cell, pair, updates, &observed, &hidden, method)?;
        println!(
            "draw {i}: {} updates, {}/31 observed, {}/225 hidden, {:.2} s",
            record["optimizer"]["step"],
            record["observed"]["correct"],
            record["hidden"]["correct"],
            record["seconds"].as_f64().ok_or("missing runtime")?
        );
        records.push(record);
    }
    let successes = records
        .iter()
        .filter(|r| r["hidden"]["all_strict"] == true)
        .count();
    let correct: usize = records
        .iter()
        .map(|r| r["hidden"]["correct"].as_u64().unwrap() as usize)
        .sum();
    let summary = json!({"protocol":plan,"runs":records.len(),"joint_successes":successes,
        "mean_hidden_accuracy":correct as f64 / (225*records.len()) as f64,"replayed":true,"records":records});
    publish(dir, "summary.json", &serde_json::to_vec_pretty(&summary)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn paired_adamw_uses_the_registered_page_initializations_and_its_own_wsd_budget() -> Result<()>
    {
        let page: Value = serde_json::from_str(include_str!("../sum8-page-protocol.json"))?;
        let pairs = serde_json::from_value(page["seeds"].clone())?;
        assert_eq!(
            protocol(UPDATES, pairs, Method::Page(PageRule::Single)),
            page
        );
        let adamw = protocol(UPDATES, pairs, Method::AdamW);
        assert_eq!(adamw["seeds"], page["seeds"]);
        assert_eq!(adamw["adamw"]["weight_decay"], 0.1);
        assert_eq!(adamw["adamw"]["warmup"], 409);
        assert_eq!(adamw["adamw"]["cooldown_start"], 16384);
        assert_eq!(adamw["adamw"]["budget"], UPDATES);
        let noisy = protocol(UPDATES, pairs, Method::AdamWFirstNoise);
        assert_eq!(noisy["seeds"], adamw["seeds"]);
        assert_eq!(noisy["adamw"], adamw["adamw"]);
        assert_eq!(noisy["seed_order"][1], "first_update_noise");
        assert_eq!(noisy["gaussian_weight_noise"]["stddev"], FIRST_NOISE_STDDEV);
        Ok(())
    }

    #[test]
    fn split_is_the_nonwrapping_staircase_and_its_complement() {
        let mut all = BTreeSet::new();
        for (observed, rows, ones) in [(true, 31, 10), (false, 225, 118)] {
            let split = Split::new(observed);
            assert_eq!(split.tokens.len(), LENGTH * rows);
            assert_eq!(split.labels.iter().filter(|&&y| y == 1).count(), ones);
            for (bits, targets) in split
                .tokens
                .as_chunks::<LENGTH>()
                .0
                .iter()
                .zip(split.labels.as_chunks::<LENGTH>().0)
            {
                let word = bits.iter().fold(0, |n, b| (n << 1) | b);
                assert!(all.insert(word));
                assert_eq!(
                    observed,
                    (word & 15) == (word >> 4) || (word & 15) == (word >> 4) + 1
                );
                assert!(targets[..7].iter().all(|&y| y == u32::MAX));
                assert_eq!(targets[7], bits.iter().sum::<u32>() % 2);
            }
        }
        assert_eq!(all.len(), 256);
    }
}
