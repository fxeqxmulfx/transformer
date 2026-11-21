//! Fit, save and evaluate a Phasor for the original NV11 arithmetic pairs.
use arithmetic::{
    CASES,
    dataset::{self, Dataset},
};
use rayon::prelude::*;
use serde_json::Value;
use std::{env, fs, io::Write, path::Path};

// The upstream CPU reference imports these host types from its CUDA executable.
mod engine {
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    pub struct Config {
        pub width: usize,
        pub vocab: usize,
        pub heads: usize,
        pub layers: usize,
        pub gain: f32,
        pub architecture: String,
        pub phase_optimizer: bool,
    }
}
use engine::{Config, Result};
#[derive(Clone, Copy, PartialEq)]
enum AttentionMode {
    Xsa,
    Ordinary,
    Centered,
}
impl AttentionMode {
    fn from_ordinary(ordinary: bool) -> Self {
        if ordinary { Self::Ordinary } else { Self::Xsa }
    }
}

include!(concat!(env!("OUT_DIR"), "/upstream.rs"));
mod circuit;
mod shifts;
use circuit::{P, config};
use oracle::Jet;

fn data(name: &str) -> Result<Dataset> {
    let cases: Vec<Value> = serde_json::from_str(CASES)?;
    let case = cases
        .iter()
        .find(|v| v["case"]["name"] == name)
        .ok_or("missing case")?;
    dataset::build(case)?.ok_or_else(|| "unavailable case".into())
}

fn save(path: &Path, weights: &[f32]) -> Result<()> {
    let bytes: Vec<_> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    file.write_all(&bytes)?;
    file.persist(path)?;
    println!(
        "{} FP32 parameters, sha256 {}",
        weights.len(),
        arithmetic::hash(&bytes)
    );
    Ok(())
}

fn load(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    if bytes.len() != 4 * layout::total_count(&config()) {
        return Err("wrong Phasor parameter block size".into());
    }
    let weights: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect();
    layout::validate(&config(), &weights)?;
    Ok(weights)
}

fn forward_weights(weights: &[f32]) -> Vec<Jet> {
    weights.iter().map(|&v| Jet::from(f64::from(v))).collect()
}

fn answer_logits(output: &[Jet]) -> impl Iterator<Item = &[Jet]> {
    output
        .as_chunks::<{ 2 * P }>()
        .0
        .iter()
        .map(|pair| &pair[P..])
}

fn forward_pairs(
    cfg: &Config,
    weights: &[Jet],
    rows: &[[u32; 3]],
) -> std::result::Result<Vec<Jet>, &'static str> {
    let tokens: Vec<_> = rows.iter().flat_map(|&[a, b, _]| [a, b]).collect();
    let output = oracle::main_model(cfg, weights, &tokens, rows.len(), 2);
    if answer_logits(&output).flatten().any(|s| !s.v.is_finite()) {
        return Err("nonfinite Phasor logit");
    }
    Ok(output)
}

#[derive(Clone, Copy)]
struct Score {
    correct: usize,
    min_margin: f64,
    loss: f64,
}

impl Score {
    fn empty() -> Self {
        Self {
            correct: 0,
            min_margin: f64::INFINITY,
            loss: 0.0,
        }
    }

    fn observe(&mut self, logits: &[Jet], label: u32) {
        let label = label as usize;
        let rival = logits
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != label)
            .map(|(_, s)| s.v)
            .fold(f64::NEG_INFINITY, f64::max);
        let margin = logits[label].v - rival;
        let peak = logits[label].v.max(rival);
        let log_mass = logits.iter().map(|s| (s.v - peak).exp()).sum::<f64>().ln();
        self.correct += usize::from(margin > 0.0);
        self.min_margin = self.min_margin.min(margin);
        self.loss += (peak - logits[label].v) + log_mass;
    }

    fn merge(self, other: Self) -> Self {
        Self {
            correct: self.correct + other.correct,
            min_margin: self.min_margin.min(other.min_margin),
            loss: self.loss + other.loss,
        }
    }
}

fn score_split(name: &str, weights: &[Jet], cubic: &[[u32; 3]], mul: &[[u32; 3]]) -> Result<()> {
    if cubic.len() != mul.len() || !cubic.iter().zip(mul).all(|(c, m)| c[..2] == m[..2]) {
        return Err("control has different operand pairs".into());
    }
    let cfg = config();
    let batches: Vec<_> = cubic
        .par_chunks(16)
        .zip(mul.par_chunks(16))
        .map(|(rows, control)| {
            let output = forward_pairs(&cfg, weights, rows)?;
            let mut scores = [Score::empty(); 2];
            for ((logits, row), control) in answer_logits(&output).zip(rows).zip(control) {
                scores[0].observe(logits, row[2]);
                scores[1].observe(logits, control[2]);
            }
            Ok::<_, &'static str>(scores)
        })
        .collect::<std::result::Result<_, _>>()?;
    for (i, target) in ["NV11", "multiplication"].iter().enumerate() {
        let score = batches
            .iter()
            .fold(Score::empty(), |s, batch| s.merge(batch[i]));
        println!(
            "{name} {target}: {}/{}, min margin {:.6}, mean CE {:.6}",
            score.correct,
            cubic.len(),
            score.min_margin,
            score.loss / cubic.len() as f64
        );
    }
    Ok(())
}

fn predict(file: &str, a: &str, b: &str) -> Result<()> {
    let tokens: [u32; 2] = [a.parse()?, b.parse()?];
    if tokens.iter().any(|&t| t >= P as u32) {
        return Err("operands must be residues 0 through 96".into());
    }
    let weights = forward_weights(&load(Path::new(file))?);
    let output = oracle::main_model(&config(), &weights, &tokens, 1, 2);
    let logits = &output[P..];
    if logits.iter().any(|s| !s.v.is_finite()) {
        return Err("nonfinite Phasor logit".into());
    }
    let winner = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.v.total_cmp(&b.v))
        .unwrap()
        .0;
    println!("{winner}");
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    match args.as_slice() {
        [action, file] if action == "fit" || action == "fit-class" => {
            let training = data("arithmetic-cubic-product")?.training;
            let weights = if action == "fit" {
                shifts::fit(&training)?
            } else {
                circuit::fit_class(&training)?
            };
            save(Path::new(file), &weights)
        }
        [action, file, rest @ ..]
            if action == "score" && (rest.is_empty() || rest == ["--ablate-shifts"]) =>
        {
            let mut weights = load(Path::new(file))?;
            if !rest.is_empty() {
                circuit::set_shifts(&mut weights, &[0; P], &[0; P]);
            }
            let weights = forward_weights(&weights);
            let cubic = data("arithmetic-cubic-product")?;
            let mul = data("arithmetic-mul")?;
            score_split("train", &weights, &cubic.training.rows, &mul.training.rows)?;
            score_split("held-out", &weights, &cubic.validation, &mul.validation)
        }
        [action, file, a, b] if action == "predict" => predict(file, a, b),
        _ => Err(concat!(
            "usage: phasor-nv11 fit WEIGHTS | fit-class WEIGHTS | ",
            "score WEIGHTS [--ablate-shifts] | predict WEIGHTS A B"
        )
        .into()),
    }
}

fn main() -> Result<()> {
    run()
}

#[cfg(test)]
mod tests;
