//! ReRoPE and Leaky ReRoPE in gpt's main research model, against its LeRoPE
//! cohort. The control architecture `complex-phase-score-lerope` at gpt 5a37c51
//! keeps SSMax, Peri-RMS, the squared-imaginary score, phase-mHC, the complex
//! FFN, precision, AdamW/WSD and every parameter, LeRoPE's learned frequencies
//! included; only the position map of the rotation changes.
//! Every module not in this directory is the pinned control's own file.
#![recursion_limit = "256"]
mod analysis;
#[path = "../../../.lake/reproduction/gpt/research/complex_geometry_suite/native/src/arithmetic_data.rs"]
#[allow(dead_code)]
mod arithmetic_data;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/attention.rs"]
mod attention;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/base_ops.rs"]
mod base_ops;
#[path = "../../../.lake/reproduction/gpt/research/complex_geometry_suite/native/src/blake.rs"]
mod blake;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/centered_attention.rs"]
mod centered_attention;
#[path = "../../../.lake/reproduction/gpt/research/complex_geometry_suite/native/src/complex.rs"]
mod complex;
#[path = "../../../.lake/reproduction/gpt/research/complex_geometry_suite/native/src/data.rs"]
#[allow(dead_code)]
mod data;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/embedding_backward.rs"]
mod embedding_backward;
#[allow(dead_code)]
mod engine;
mod far_rotary;
#[path = "../../../.lake/reproduction/gpt/research/complex_geometry_suite/native/src/gemm.rs"]
mod gemm;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/layout.rs"]
mod layout;
#[path = "../../../.lake/reproduction/gpt/research/complex_geometry_suite/native/src/metrics.rs"]
#[allow(dead_code)]
mod metrics;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/mhc.rs"]
mod mhc;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/optimizer.rs"]
// The checkpoint's optimizer state is saved and restored by `main` alone.
#[cfg_attr(test, allow(dead_code))]
mod optimizer;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/optimizer_kernels.rs"]
mod optimizer_kernels;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/ordinary_attention.rs"]
mod ordinary_attention;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/ordinary_forward.rs"]
mod ordinary_forward;
#[path = "../../../.lake/reproduction/gpt/research/complex_geometry_suite/native/src/pcg.rs"]
#[allow(dead_code)]
mod pcg;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/phase_mhc.rs"]
mod phase_mhc;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/phase_score.rs"]
mod phase_score;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/phase_score_forward.rs"]
mod phase_score_forward;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/pipeline.rs"]
mod pipeline;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/pipeline_host.rs"]
#[allow(dead_code)]
mod pipeline_host;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/placement.rs"]
#[allow(dead_code)]
mod placement;
#[path = "../../../.lake/reproduction/gpt/research/complex_geometry_suite/diagnostic/src/plain_product.rs"]
mod plain_product;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/precision.rs"]
#[allow(dead_code)]
mod precision;
#[path = "../../../.lake/reproduction/gpt/research/complex_geometry_suite/native/src/random.rs"]
#[allow(dead_code)]
mod random;
mod rerope_backward;
mod rerope_forward;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/rotary.rs"]
#[allow(dead_code)]
mod rotary;
#[allow(dead_code)]
mod runner;
#[path = "../../../.lake/reproduction/gpt/research/main_model/wsd_suite_native/src/schedule.rs"]
#[allow(dead_code)]
mod schedule;
#[path = "../../../.lake/reproduction/gpt/research/complex_geometry_suite/native/src/sequence_ops.rs"]
mod sequence_ops;
mod sources;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/stable_attention.rs"]
mod stable_attention;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/stable_chain.rs"]
mod stable_chain;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/stable_loss.rs"]
mod stable_loss;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/tensor.rs"]
mod tensor;
#[path = "../../../.lake/reproduction/gpt/research/main_model/phase_score_native/src/tiled_transpose.rs"]
mod tiled_transpose;
mod validation;
use engine::{Config, Result};
use sha2::{Digest, Sha256};
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn main() -> std::process::ExitCode {
    let args: Vec<_> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("validate") if args.len() == 4 => validation::run(
            std::path::Path::new(&args[2]),
            std::path::Path::new(&args[3]),
        ),
        Some("analyze") if args.len() == 4 => analysis::run(
            std::path::Path::new(&args[2]),
            std::path::Path::new(&args[3]),
        ),
        _ => runner::run(),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod loss_precision {
    // The control's scalar softmax oracle (gpt 5a37c51 main.rs), for stable_loss's tests.
    pub fn reference_softmax(logits: &[f32], gold: usize) -> (f64, Vec<f64>) {
        let max = logits
            .iter()
            .map(|&v| v as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let mass: Vec<_> = logits.iter().map(|&v| (v as f64 - max).exp()).collect();
        let other: f64 = mass
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != gold)
            .map(|(_, v)| v)
            .sum();
        let total = mass[gold] + other;
        let mut grad: Vec<_> = mass.iter().map(|v| v / total).collect();
        grad[gold] = -other / total;
        let loss = if logits[gold] as f64 == max {
            other.ln_1p()
        } else {
            max - logits[gold] as f64 + total.ln()
        };
        (loss, grad)
    }
}
