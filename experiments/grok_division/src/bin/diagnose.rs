//! Reads checkpoints of a run and prints, for each, what the model does at the answer's
//! position on a sample of its training equations: accuracy and cross-entropy, how many hidden
//! units fire, how spread the attention is, what the normalizations divide by, and the norms of
//! the parameter groups.
//!
//!     diagnose RUN_DIR SEED STEP...
//!
//! It shares the layout, the data and the reference model in doubles with the trainer and
//! nothing of the trainer's fast path; the trainer's sources are not touched by it.

// The shared modules have items that only the trainer or its tests use.
#![allow(dead_code, clippy::chunks_exact_to_as_chunks)]

#[path = "../data.rs"]
mod data;
#[path = "../layout.rs"]
mod layout;
#[path = "../mt.rs"]
mod mt;
#[path = "../reference.rs"]
mod reference;

use std::path::Path;
use std::process::ExitCode;

use layout::Dims;
use sha2::{Digest, Sha256};

/// Training equations read: the first of the run's training set.
const SAMPLE: usize = 512;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("diagnose: {message}");
            ExitCode::FAILURE
        }
    }
}

/// The parameters of a checkpoint, in doubles, after its digest is checked.
fn parameters(path: &Path, dims: Dims) -> Result<Vec<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 96 || &bytes[..8] != b"GROKDIV1" {
        return Err(format!("{}: not a checkpoint", path.display()));
    }
    let (body, digest) = bytes.split_at(bytes.len() - 32);
    if Sha256::digest(body).as_slice() != digest {
        return Err(format!("{}: the digest does not match the bytes", path.display()));
    }
    let len = u64::from_le_bytes(body[56..64].try_into().expect("eight bytes")) as usize;
    if len != dims.len() {
        return Err(format!("{}: another model", path.display()));
    }
    Ok(body[64..64 + 4 * len]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")) as f64)
        .collect())
}

fn norm(p: &[f64]) -> f64 {
    p.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn mean(p: &[f64]) -> f64 {
    p.iter().sum::<f64>() / p.len() as f64
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("usage: diagnose RUN_DIR SEED STEP...".into());
    }
    let dir = Path::new(&args[0]);
    let seed: u32 = args[1].parse().map_err(|_| "SEED is not a number".to_string())?;
    let dims = Dims::PAPER;
    let (train, _) = data::split(seed, 4656);
    let (d, dff, vocab, pos) = (dims.d, dims.dff, dims.vocab, data::ANSWER_POS);
    for step in &args[2..] {
        let step: u64 = step.parse().map_err(|_| "STEP is not a number".to_string())?;
        let p = parameters(&dir.join(format!("checkpoint-{step:07}.bin")), dims)?;
        let (mut right, mut ce) = (0usize, 0.0);
        let mut alive = vec![vec![false; dff]; dims.layers];
        let mut firing = vec![0.0; dims.layers];
        let mut entropy = vec![0.0; dims.layers];
        let mut sigma = vec![(0.0, 0.0); dims.layers];
        for equation in &train[..SAMPLE] {
            let tokens = &equation.tokens()[..dims.ctx];
            let logits = reference::logits(dims, &p, tokens);
            let row = &logits[pos * vocab..(pos + 1) * vocab];
            let target = equation.target(pos);
            let top = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            right += (row.iter().position(|&v| v == top) == Some(target)) as usize;
            ce += reference::cross_entropy(row, target).0;
            let probe = reference::probe(dims, &p, tokens, pos);
            for l in 0..dims.layers {
                for (flag, &active) in alive[l].iter_mut().zip(&probe.active[l]) {
                    *flag |= active;
                }
                firing[l] += probe.active[l].iter().filter(|&&a| a).count() as f64 / dff as f64;
                let weights = &probe.attention[l];
                entropy[l] -= weights.iter().filter(|&&a| a > 0.0).map(|a| a * a.ln()).sum::<f64>() / dims.heads as f64;
                sigma[l].0 += probe.sigma[l].0;
                sigma[l].1 += probe.sigma[l].1;
            }
        }
        let n = SAMPLE as f64;
        println!(
            "step {step}: accuracy {:.4}, cross-entropy {:.4} on {SAMPLE} training equations",
            right as f64 / n,
            ce / n
        );
        println!(
            "  norms: all {:.2}, embedding {:.2}, readout {:.2}",
            norm(&p),
            norm(&p[dims.emb()..dims.emb() + vocab * d]),
            norm(&p[dims.wout()..dims.wout() + d * vocab])
        );
        for l in 0..dims.layers {
            let at = dims.block(l);
            println!(
                "  block {l}: hidden units that fire on some equation {}/{dff}, mean share firing {:.3}; attention entropy {:.3} of {:.3}; sigma {:.3} then {:.3}",
                alive[l].iter().filter(|&&a| a).count(),
                firing[l] / n,
                entropy[l] / n,
                ((pos + 1) as f64).ln(),
                sigma[l].0 / n,
                sigma[l].1 / n,
            );
            println!(
                "    norms: Wq {:.2}, WkWv {:.2}, Wo {:.2}, W1 {:.2}, W2 {:.2}; gains: first mean {:.3} norm {:.2}, second mean {:.3} norm {:.2}; biases {:.2}, {:.2}",
                norm(&p[at.wq..at.wq + d * d]),
                norm(&p[at.wkv..at.wkv + 2 * d * d]),
                norm(&p[at.wo..at.wo + d * d]),
                norm(&p[at.w1..at.w1 + d * dff]),
                norm(&p[at.w2..at.w2 + dff * d]),
                mean(&p[at.g1..at.g1 + d]),
                norm(&p[at.g1..at.g1 + d]),
                mean(&p[at.g2..at.g2 + d]),
                norm(&p[at.g2..at.g2 + d]),
                norm(&p[at.b1..at.b1 + d]),
                norm(&p[at.b2..at.b2 + d]),
            );
        }
    }
    Ok(())
}
