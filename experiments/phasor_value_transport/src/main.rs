//! Matched AdamW training on the original half-table split of NV11.
#![recursion_limit = "256"]

use arithmetic::{CASES, dataset};
use arithmetic::{hash, reference as arithmetic_data};
use serde_json::{Value, json};
use std::{env, fs, io::Write, path::Path};

mod addresses;
mod causal_values;
mod frequency_read;
mod pair_checks;
mod pair_memory;
mod pair_ops;
mod relation_paths;
include!(concat!(env!("OUT_DIR"), "/upstream.rs"));
use engine::{Config, Engine, Result, ValueLayer};
mod checks;
mod snapshot;
mod stress;

const SEED: u32 = 1;
const BUDGET: usize = 100_000;
const BATCH: usize = 256;
const OBSERVE: usize = 250;
const WINDOW: usize = 2000;
const KERNITE_VERSION: &str = env!("KERNITE_CUDA_VERSION");

fn config(batch: usize) -> Result<Config> {
    Ok(Config {
        width: 128,
        vocab: 97,
        layers: 1,
        heads: 4,
        max_length: 2,
        max_batch: batch,
        normalized: true,
        ssmax: true,
        gain: 4. / 128.,
        lr: schedule::PEAK,
        decay: 0.1,
        warmup: schedule::Schedule::new(BUDGET)?.warmup,
        architecture: "complex-phase-score-lerope".into(),
        phase_optimizer: false,
        shared_values: false,
    })
}

fn cubic_data() -> Result<dataset::Dataset> {
    let cases: Vec<Value> = serde_json::from_str(CASES)?;
    let row = cases
        .iter()
        .find(|v| v["case"]["name"] == "arithmetic-cubic-product")
        .ok_or("missing arithmetic-cubic-product case")?;
    dataset::build(row)?.ok_or_else(|| "missing NV11 split".into())
}

fn forward(e: &mut Engine, rows: &[[u32; 3]]) -> Result<()> {
    let tokens: Vec<_> = rows.iter().flat_map(|&[a, b, _]| [a, b]).collect();
    let labels: Vec<_> = rows.iter().flat_map(|&[_, _, y]| [u32::MAX, y]).collect();
    e.forward(&tokens, &labels, rows.len(), 2)
}

fn evaluate(
    e: &mut Engine,
    layer: Option<(&ValueLayer, bool)>,
    rows: &[[u32; 3]],
) -> Result<Value> {
    let mut correct = 0;
    let mut loss = 0.;
    let mut first = [None; 97];
    for chunk in rows.chunks(BATCH) {
        forward(e, chunk)?;
        if let Some((l, product)) = layer {
            let addresses = e.predictions.read()?;
            for (i, &[a, _, _]) in chunk.iter().enumerate() {
                let address = addresses[2 * i];
                if first[a as usize]
                    .replace(address)
                    .is_some_and(|old| old != address)
                {
                    return Err("first causal token changed address across second operands".into());
                }
            }
            l.forward(e, chunk.len(), product)?;
        }
        let answers = e.predictions.read()?;
        let losses = e.losses(2 * chunk.len())?;
        for (i, &[_, _, target]) in chunk.iter().enumerate() {
            correct += usize::from(answers[2 * i + 1] == target);
            loss += f64::from(losses[2 * i + 1]);
        }
    }
    if !loss.is_finite() {
        return Err("nonfinite evaluation loss".into());
    }
    // The cube is inspected only after inference to diagnose the learned address.
    let cubic_rows = layer.map(|_| {
        first
            .iter()
            .enumerate()
            .filter(|&(a, s)| *s == Some(((a * a * a) % 97) as u32))
            .count()
    });
    Ok(
        json!({"correct":correct,"examples":rows.len(),"mean_ce":loss/rows.len() as f64,
        "cubic_address_rows":cubic_rows}),
    )
}

fn cell(arm: &str, out: &Path, seed: u32, window: usize) -> Result<()> {
    if window == 0 || window > BUDGET || !window.is_multiple_of(OBSERVE) {
        return Err(
            "NV11 observation window must be a positive multiple of 250 up to 100000".into(),
        );
    }
    let data = cubic_data()?;
    let mut cfg = config(BATCH)?;
    cfg.shared_values = arm == "shared-values";
    let initial = layout::initialize_model(&cfg, seed);
    let digest = arithmetic::hash(
        &initial
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let mut e = Engine::new(cfg, &initial)?;
    let layer = (arm == "transport")
        .then(|| ValueLayer::new(&e))
        .transpose()?;
    let mut opt = optimizer::Optimizer::new(&e, BUDGET)?;
    let mut rng = random::Random::new(seed.checked_add(10_000).ok_or("invalid order seed")?);
    let mut epoch = Vec::new();
    let mut offset = 0;
    fs::create_dir_all(out)?;
    let mut log = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join(format!("{arm}.jsonl")))?;
    let mut saved_exact = false;
    for completed in 0..=window {
        if completed % OBSERVE == 0 {
            let read = layer.as_ref().map(|l| (l, true));
            let train = evaluate(&mut e, read, &data.training.rows)?;
            let held = evaluate(&mut e, read, &data.validation)?;
            let ablated = layer
                .as_ref()
                .map(|l| evaluate(&mut e, Some((l, false)), &data.validation))
                .transpose()?;
            let row = json!({"arm":arm,"step":completed,"seed":seed,"initial_sha256":digest,
                "train":train,"held_out":held,"zero_offset_held_out":ablated,
                "fp32_affines":precision::FP32_AFFINE_REFERENCE,"gpu":e.cuda.name(),
                "kernite_version":KERNITE_VERSION});
            writeln!(log, "{row}")?;
            log.flush()?;
            println!("{row}");
            if (layer.is_some() || e.config.shared_values)
                && (completed == OBSERVE || completed == window)
            {
                snapshot::save(&e, &out.join(format!("{arm}-{completed}.f32")))?;
            }
            if e.config.shared_values
                && !saved_exact
                && row["train"]["correct"] == data.training.rows.len()
                && row["held_out"]["correct"] == data.validation.len()
            {
                snapshot::save(&e, &out.join(format!("{arm}-first-exact.f32")))?;
                saved_exact = true;
            }
        }
        if completed == window {
            break;
        }
        if offset == epoch.len() {
            epoch = (0..data.training.rows.len()).collect();
            rng.shuffle(&mut epoch);
            offset = 0;
        }
        let end = (offset + BATCH).min(epoch.len());
        let batch: Vec<_> = epoch[offset..end]
            .iter()
            .map(|&i| data.training.rows[i])
            .collect();
        offset = end;
        forward(&mut e, &batch)?;
        if let Some(l) = &layer {
            l.forward(&e, batch.len(), true)?;
            l.backward(&e, batch.len(), true)?;
        } else {
            e.backward(batch.len(), 2)?;
        }
        opt.step(&e, completed)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    match args.as_slice() {
        [op] if op == "check" => checks::run(),
        [op] if matches!(op.as_str(), "shared-check" | "pair-check") => pair_checks::run(None),
        [op, width] if op == "shared-check" => pair_checks::run(Some(width.parse()?)),
        [op] if op == "routes" => stress::routes(),
        [op] if op == "extensions" => stress::extensions(),
        [op, name] if op == "successor-alignment" => stress::alignment::run(name, 1),
        [op, name, hops] if op == "successor-alignment" => stress::alignment::run(name, hops.parse()?),
        [op, dest] if matches!(op.as_str(), "report" | "extended-report") =>
            stress::report::run(Path::new(dest), op == "extended-report"),
        [op, dest, names @ ..] if op == "catalog-report" && !names.is_empty() =>
            stress::report::catalog(Path::new(dest), names),
        [op, name, seed, out] if op == "cell" => stress::run(name, seed.parse()?, Path::new(out), None),
        [op, name, seed, out, steps] if op == "screen" =>
            stress::run(name, seed.parse()?, Path::new(out), Some(steps.parse()?)),
        [op, name, seed, out] if op == "replay" => stress::replay(name, seed.parse()?, Path::new(out)),
        [op, name, seed, out] if op == "probes" =>
            stress::score_checkpoint(name, seed.parse()?, Path::new(out), true),
        [arm, dest] if matches!(arm.as_str(), "transport" | "phasor" | "shared-values") => {
            cell(arm, Path::new(dest), SEED, WINDOW)
        }
        [arm, dest, seed] if matches!(arm.as_str(), "transport" | "phasor" | "shared-values") => {
            cell(arm, Path::new(dest), seed.parse()?, WINDOW)
        }
        [arm, dest, seed, steps] if arm == "shared-values" =>
            cell(arm, Path::new(dest), seed.parse()?, steps.parse()?),
        [op, file] if op == "inspect" => snapshot::inspect(Path::new(file)),
        [op, file] if op == "nv11-score" => snapshot::nv11_scores(Path::new(file)),
        [op, file, a, b] if op == "predict" => snapshot::predict(Path::new(file), None, a.parse()?, b.parse()?),
        [op, file, p, a, b] if op == "predict" => snapshot::predict(Path::new(file), Some(p.parse()?), a.parse()?, b.parse()?),
        _ => Err("usage: phasor-value-transport {check|routes|extensions} | shared-check [WIDTH] | successor-alignment CASE [HOPS] | {report|extended-report} RUN_DIR | catalog-report RUN_DIR CASE... | {cell|replay|probes} CASE SEED RUN_DIR | screen CASE SEED RUN_DIR LAST_OBSERVATION | {transport|phasor} RUN_DIR [SEED] | shared-values RUN_DIR [SEED [STEPS]] | {inspect|nv11-score} FILE | predict FILE [VOCAB] A B".into()),
    }
}
