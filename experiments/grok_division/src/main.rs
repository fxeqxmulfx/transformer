//! Division mod 97 of arXiv:2201.02177 under the author's Adam and AdamW protocols, trained
//! natively. `PROTOCOL.md` pins the model, the data and the optimizer to the paper and to
//! openai/grok at 3d64b1d8; `README.md` has the commands and the findings.
//!
//! ```text
//! grok-division check                          # the pins and the numerical preflight
//! grok-division bench [threads]                # the time of an update
//! grok-division run adam|adamw SEED DIR [--threads N] [--budget N] [--answer-only]
//! grok-division report DIR...                  # a line of JSON for each run
//! ```

// Loops over an index that reads several rows at once stay loops over the index.
#![allow(clippy::needless_range_loop, clippy::chunks_exact_to_as_chunks, clippy::type_complexity)]

mod adam;
mod data;
mod gemm;
mod layout;
mod model;
mod mt;
mod reference;
mod run;
#[cfg(test)]
mod tests;
mod train;

use adam::Hyper;
use data::{ANSWER_POS, EOS_POS};
use layout::Dims;
use mt::Mt;
use reference::Example;
use std::path::Path;
use std::process::{Command, ExitCode};
use std::time::Instant;
use train::{Trainer, init_seed, initialize};

/// The commit of Kernite the kernel is taken from and built with.
const KERNITE: &str = "a1b6bc2c956e1a5f35960e26b209e1b6c213096c";
/// Where the manifest finds Kernite.
const KERNITE_DIR: &str = "/home/misha/rust_projects/kgen";
/// Threads of a step unless `--threads` says otherwise.
const THREADS: usize = 8;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let done = match args.first().map(String::as_str) {
        Some("check") => check(),
        Some("bench") => bench(args.get(1).and_then(|a| a.parse().ok()).unwrap_or(THREADS)),
        Some("kernel") => {
            kernel_speed();
            Ok(())
        }
        Some("run") => launch(&args[1..]),
        Some("report") => {
            args[1..].iter().try_for_each(|dir| run::report(Path::new(dir)).map(|line| println!("{line}")))
        }
        _ => Err("usage: grok-division check | bench [threads] | run adam|adamw SEED DIR [--threads N] [--budget N] \
                  [--answer-only] | report DIR..."
            .to_string()),
    };
    match done {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

/// The configuration of `run PROTOCOL SEED DIR [flags]`, and the directory.
fn launch(args: &[String]) -> Result<(), String> {
    let [protocol, seed, dir, flags @ ..] = args else {
        return Err("run takes a protocol, a seed and a directory".into());
    };
    let (weight_decay, mut budget, checkpoint_every) = match protocol.as_str() {
        "adam" => (0.0, 1_000_000, 50_000),
        "adamw" => (1.0, 100_000, 10_000),
        other => return Err(format!("the protocol is adam or adamw, not {other}")),
    };
    let seed: u32 = seed.parse().map_err(|_| format!("the seed {seed} is not a number"))?;
    let (mut threads, mut sup, mut stop_after) = (THREADS, vec![ANSWER_POS, EOS_POS], None);
    let mut flags = flags.iter();
    while let Some(flag) = flags.next() {
        let mut value = || flags.next().and_then(|v| v.parse::<u64>().ok()).ok_or(format!("{flag} takes a number"));
        match flag.as_str() {
            "--threads" => threads = value()? as usize,
            "--budget" => budget = value()?,
            "--stop-after" => stop_after = Some(value()?),
            "--answer-only" => sup = vec![ANSWER_POS],
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let config = run::Config { protocol: protocol.clone(), seed, weight_decay, budget, sup, threads, checkpoint_every };
    run::run(&config, Path::new(dir), stop_after)
}

/// Equations as the reference model's examples.
fn examples(equations: &[data::Equation]) -> Vec<Example> {
    equations
        .iter()
        .map(|eq| Example { tokens: eq.tokens()[..6].to_vec(), targets: (0..6).map(|pos| eq.target(pos)).collect() })
        .collect()
}

/// The preflight: the pins, the split's fingerprints, and the fast model against the reference
/// at the paper's sizes, at initialization and after updates.
fn check() -> Result<(), String> {
    if !gemm::supported() {
        return Err("the kernel needs AVX2 and FMA".into());
    }
    let git = |args: &[&str]| -> Result<String, String> {
        let out =
            Command::new("git").arg("-C").arg(KERNITE_DIR).args(args).output().map_err(|e| format!("git: {e}"))?;
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let head = git(&["rev-parse", "HEAD"])?;
    let dirty = git(&["status", "--porcelain", "--", "crates", "examples/matmul"])?;
    println!("kernite {head} {}", if dirty.is_empty() { "clean" } else { "modified" });
    if head != KERNITE || !dirty.is_empty() {
        return Err(format!("Kernite is not the clean checkout of {KERNITE}"));
    }
    println!("sources {}", run::source_fingerprint());
    let dims = Dims::PAPER;
    for seed in 0..3 {
        let (train, heldout) = data::split(seed, 4656);
        println!("split {seed}: train {} heldout {}", data::fingerprint(&train), data::fingerprint(&heldout));
    }
    let (train, _) = data::split(0, 4656);
    let mut failures = 0;
    for (label, sup, updates) in [
        ("source", vec![ANSWER_POS, EOS_POS], 0),
        ("source", vec![ANSWER_POS, EOS_POS], 30),
        ("answer", vec![ANSWER_POS], 30),
    ] {
        let params = initialize(dims, &mut Mt::new(init_seed(0)));
        let mut trainer = Trainer::new(dims, sup.clone(), Hyper::source(1.0), params, 3);
        for step in 0..updates {
            trainer.update(&train[step * 64..(step + 1) * 64]);
        }
        let batch = &train[2000..2011];
        let fast_loss = trainer.gradient(batch);
        let p64: Vec<f64> = trainer.p.iter().map(|&v| v as f64).collect();
        let (loss, grad) = reference::loss_and_gradient(dims, &p64, &examples(batch), &sup);
        let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        let error = grad.iter().zip(&trainer.grad).map(|(a, &b)| (a - b as f64).powi(2)).sum::<f64>().sqrt();
        let eval = trainer.evaluate(batch);
        let mut logit_error = 0f64;
        for (b, example) in examples(batch).iter().enumerate() {
            let all = reference::logits(dims, &p64, &example.tokens);
            let row = &all[ANSWER_POS * dims.vocab..(ANSWER_POS + 1) * dims.vocab];
            let target = example.targets[ANSWER_POS];
            let rival = (0..dims.vocab).filter(|&c| c != target).map(|c| row[c]).fold(f64::NEG_INFINITY, f64::max);
            logit_error = logit_error.max((row[target] - rival - eval.margins[b] as f64).abs());
        }
        let ok = (fast_loss - loss).abs() < 1e-5 * loss.max(1.0) && error < 1e-4 * norm && logit_error < 1e-4;
        failures += usize::from(!ok);
        println!(
            "{label} after {updates} updates: loss {fast_loss:.9} against {loss:.9}; gradient error {:.3e} of norm {norm:.3e}; \
             margin error {logit_error:.3e}: {}",
            error,
            if ok { "pass" } else { "FAIL" }
        );
    }
    if failures == 0 { Ok(()) } else { Err(format!("{failures} checks failed")) }
}

/// The time of an update on `threads` threads, and of an exhaustive evaluation.
fn bench(threads: usize) -> Result<(), String> {
    let dims = Dims::PAPER;
    let (train, heldout) = data::split(0, 4656);
    let params = initialize(dims, &mut Mt::new(init_seed(0)));
    let mut trainer = Trainer::new(dims, vec![ANSWER_POS, EOS_POS], Hyper::source(0.0), params, threads);
    for _ in 0..3 {
        trainer.update(&train[..512]);
    }
    let start = Instant::now();
    let count = 30;
    for i in 0..count {
        trainer.update(&train[i * 100..i * 100 + 512]);
    }
    let update = start.elapsed().as_secs_f64() / count as f64;
    let start = Instant::now();
    let eval = trainer.evaluate(&heldout);
    println!(
        "{threads} threads: {:.2} ms an update, {:.0} ms an evaluation of {} equations (accuracy {:.4})",
        1e3 * update,
        1e3 * start.elapsed().as_secs_f64(),
        heldout.len(),
        eval.accuracy()
    );
    Ok(())
}

/// The kernel's single-thread speed at the shapes of a chunk's products, in GFLOPS.
fn kernel_speed() {
    let mut gemm = gemm::Gemm::default();
    let mut rng = Mt::new(1);
    for (m, n, k) in [
        (192, 128, 128),
        (192, 256, 128),
        (192, 512, 128),
        (192, 128, 512),
        (128, 512, 192),
        (512, 128, 192),
        (64, 239, 128),
        (128, 256, 192),
    ] {
        let a: Vec<f32> = (0..m * k).map(|_| rng.normal() as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.normal() as f32).collect();
        let mut c = vec![0f32; m * n];
        let start = Instant::now();
        let mut count = 0u64;
        while start.elapsed().as_secs_f64() < 0.3 {
            gemm.mul(&mut c, &a, &b, (m, n, k));
            count += 1;
        }
        let flops = 2.0 * (m * n * k) as f64 * count as f64;
        println!("{m} by {n} by {k}: {:.1} GFLOPS", flops / start.elapsed().as_secs_f64() / 1e9);
    }
}
