//! Independent complex coordinates expanded as [[A,-B],[B,A]] for real GEMM.
//! Generalizes complex_transformer/native/src/layout.rs at GPT commit 786696b:
//! arbitrary vocabulary and 16/32/128-real-coordinate heads, fixed hidden width.
//!
//! Experiment copy of gpt 5a37c51 `phase_score_native/src/layout.rs`: the
//! control architecture `complex-phase-score-lerope` with LeRoPE replaced by
//! a distance bias after the SSMax gain (`Bias`). The trailing group that held
//! LeRoPE's frequencies holds KERPLE's (r1, r2) per head; the fixed biases
//! have none. Every parameter before it is initialized as in the control.
use crate::engine::{Config, Result};
use crate::random::Random;

/// The distance bias b_h(r), r = t - key, added after the SSMax gain.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bias {
    /// ALiBi: -m_h r with fixed slopes m_h = 2^(-8h/H), h = 1..H.
    Alibi,
    /// -ln(1 + r) in every head, fixed: `logBias` of `Recall.LogBias`.
    Log,
    /// KERPLE's logarithmic variant -r1 ln(1 + r2 r), r1, r2 learned.
    KerpleLog,
    /// KERPLE's power variant -r1 r^r2, r1 and 0 < r2 <= 2 learned.
    KerplePower,
    /// No bias and no rotation: causal masking alone orders the keys.
    None,
}

pub fn bias_of(architecture: &str) -> Option<Bias> {
    match architecture {
        "complex-phase-score-alibi" => Some(Bias::Alibi),
        "complex-phase-score-log-bias" => Some(Bias::Log),
        "complex-phase-score-kerple-log" => Some(Bias::KerpleLog),
        "complex-phase-score-kerple-power" => Some(Bias::KerplePower),
        "complex-phase-score-nope" => Some(Bias::None),
        _ => None,
    }
}
pub fn bias(cfg: &Config) -> Option<Bias> {
    bias_of(&cfg.architecture)
}
pub fn kerple(cfg: &Config) -> bool {
    matches!(bias(cfg), Some(Bias::KerpleLog | Bias::KerplePower))
}

pub fn is_complex(architecture: &str) -> bool {
    matches!(
        architecture,
        "complex-linear"
            | "complex-hermitian"
            | "complex-mhc"
            | "complex-mhc-lerope"
            | "complex-mhc-log-lerope"
            | "complex-phase-mhc-lerope"
            | "complex-phase-score-lerope"
    ) || bias_of(architecture).is_some()
}

pub fn is_mhc(cfg: &Config) -> bool {
    matches!(
        cfg.architecture.as_str(),
        "complex-mhc"
            | "complex-mhc-lerope"
            | "complex-mhc-log-lerope"
            | "complex-phase-mhc-lerope"
            | "complex-phase-score-lerope"
    ) || bias(cfg).is_some()
}
pub fn learned_rotary(cfg: &Config) -> bool {
    matches!(
        cfg.architecture.as_str(),
        "complex-mhc-lerope"
            | "complex-mhc-log-lerope"
            | "complex-phase-mhc-lerope"
            | "complex-phase-score-lerope"
    )
}
pub fn log_positions(cfg: &Config) -> bool {
    cfg.architecture == "complex-mhc-log-lerope"
}
/// The trailing group: LeRoPE's frequencies, or KERPLE's (r1, r2) per head.
pub fn frequency_count(cfg: &Config) -> usize {
    if learned_rotary(cfg) {
        cfg.width / cfg.heads / 2
    } else if kerple(cfg) {
        2 * cfg.heads
    } else {
        0
    }
}
pub fn phase_routing(cfg: &Config) -> bool {
    matches!(
        cfg.architecture.as_str(),
        "complex-phase-mhc-lerope" | "complex-phase-score-lerope"
    ) || bias(cfg).is_some()
}
pub fn phase_count(cfg: &Config) -> usize {
    if phase_routing(cfg) {
        6 * cfg.layers
    } else {
        0
    }
}
pub fn frequency_offset(cfg: &Config) -> usize {
    score_offset(cfg) + score_count(cfg)
}
pub fn phase_score(cfg: &Config) -> bool {
    cfg.architecture == "complex-phase-score-lerope" || bias(cfg).is_some()
}
pub fn score_count(cfg: &Config) -> usize {
    if phase_score(cfg) {
        cfg.layers * cfg.heads
    } else {
        0
    }
}
pub fn score_offset(cfg: &Config) -> usize {
    phase_offset(cfg) + phase_count(cfg)
}
pub fn phase_offset(cfg: &Config) -> usize {
    base_count(cfg)
        + if is_mhc(cfg) {
            2 * cfg.layers * router_count(cfg.width)
        } else {
            0
        }
}
pub fn router_count(width: usize) -> usize {
    96 * width + 27
}
pub fn base_count(cfg: &Config) -> usize {
    count(cfg.width, cfg.layers, cfg.vocab)
}
pub fn total_count(cfg: &Config) -> usize {
    frequency_offset(cfg) + frequency_count(cfg)
}
pub fn router_offset(cfg: &Config, sublayer: usize) -> usize {
    base_count(cfg) + sublayer * router_count(cfg.width)
}

pub fn hidden(width: usize, _architecture: &str) -> usize {
    let old = (8 * width + 1) / 3;
    old.div_ceil(2) * 2
}

pub fn count(width: usize, layers: usize, vocab: usize) -> usize {
    vocab * width + layers * (4 * width * width + 3 * width * hidden(width, ""))
}

pub fn initialize(width: usize, layers: usize, vocab: usize, seed: u32) -> Vec<f32> {
    let mut rng = Random::new(seed);
    (0..count(width, layers, vocab))
        .map(|_| rng.gauss(0.0, 0.02) as f32)
        .collect()
}

/// Each owner is [A(real row,real col), B(imag row,real col), -B, A, kind].
/// kind=1 is an embedding pair without duplicated entries; kind=0 is a matrix.
fn matrix(map: &mut Vec<[u32; 5]>, offset: usize, n: usize, k: usize, og: usize, ig: usize) {
    let (np, kp) = (n / (2 * og), k / (2 * ig));
    for r in 0..n / 2 {
        let rr = r / np * 2 * np + r % np;
        let ri = rr + np;
        for c in 0..k / 2 {
            let cr = c / kp * 2 * kp + c % kp;
            let ci = cr + kp;
            map.push([
                (offset + rr * k + cr) as u32,
                (offset + ri * k + cr) as u32,
                (offset + rr * k + ci) as u32,
                (offset + ri * k + ci) as u32,
                0,
            ]);
        }
    }
}

pub fn pairs(cfg: &Config) -> Vec<[u32; 5]> {
    let (w, heads) = (cfg.width, cfg.heads);
    let d = w / heads;
    let h = hidden(w, &cfg.architecture);
    let mut map = Vec::new();
    for token in 0..cfg.vocab {
        for head in 0..heads {
            for c in 0..d / 2 {
                let at = token * w + head * d + c;
                map.push([at as u32, (at + d / 2) as u32, 0, 0, 1]);
            }
        }
    }
    for l in 0..cfg.layers {
        let o = cfg.vocab * w + l * (4 * w * w + 3 * w * h);
        matrix(&mut map, o, 3 * w, w, 3 * heads, heads);
        matrix(&mut map, o + 3 * w * w, w, w, heads, heads);
        matrix(&mut map, o + 4 * w * w, 2 * h, w, 2, heads);
        matrix(&mut map, o + 4 * w * w + 2 * h * w, w, h, heads, 1);
    }
    map
}

pub fn independent_count(cfg: &Config) -> usize {
    if is_complex(&cfg.architecture) {
        2 * pairs(cfg).len() + total_count(cfg) - base_count(cfg)
    } else {
        count(cfg.width, cfg.layers, cfg.vocab)
    }
}

pub fn initialize_model(cfg: &Config, seed: u32) -> Vec<f32> {
    let mut p = initialize(cfg.width, cfg.layers, cfg.vocab, seed);
    if is_complex(&cfg.architecture) {
        // Keep the real arm's A/B entries and embedding. The two complex arms
        // start bit-identically; projection input variance remains unchanged.
        for [ar, bi, br, ai, kind] in pairs(cfg) {
            if kind == 0 {
                p[ai as usize] = p[ar as usize];
                p[br as usize] = -p[bi as usize];
            }
        }
    }
    if is_mhc(cfg) {
        let mut rng = Random::new(seed + 20000);
        for _ in 0..2 * cfg.layers {
            p.extend((0..96 * cfg.width).map(|_| rng.gauss(0.0, 0.02) as f32));
            p.extend((0..24).map(|i| {
                if i >= 8 && (i - 8) / 4 == (i - 8) % 4 {
                    2.
                } else {
                    0.
                }
            }));
            p.extend([0.01; 3]);
        }
    }
    p.resize(total_count(cfg), 0.);
    if kerple(cfg) {
        // KERPLE's released code draws torch.rand for bias_p (scaled by 2)
        // then bias_a, per head, and clamps both at 0.01 before use; log is
        // -p ln(1 + a r), power -a r^p. A separate stream keeps every earlier
        // parameter bit-identical to the control's.
        let mut rng = Random::new(seed + 30000);
        let h = cfg.heads;
        let draws: Vec<f32> = (0..2 * h)
            .map(|i| ((if i < h { 2.0 } else { 1.0 }) * rng.random()) as f32)
            .collect();
        let at = frequency_offset(cfg);
        for head in 0..h {
            let (power, scale) = (draws[head], draws[h + head]);
            let (r1, r2) = if bias(cfg) == Some(Bias::KerpleLog) {
                (power, scale)
            } else {
                (scale, power)
            };
            p[at + 2 * head] = r1.max(0.01);
            p[at + 2 * head + 1] = r2.max(0.01);
        }
    }
    p
}

/// ALiBi's slopes for n heads, as its released code (and KERPLE's copy of it)
/// computes them: 2^(-8h/n) for a power of two, interleaved otherwise.
pub fn alibi_slopes(n: usize) -> Vec<f64> {
    let power = |n: usize| -> Vec<f64> {
        let start = 2f64.powf(-(2f64.powf(-((n as f64).log2() - 3.0))));
        (0..n).map(|i| start * start.powi(i as i32)).collect()
    };
    if n.is_power_of_two() {
        power(n)
    } else {
        let closest = 1 << n.ilog2();
        let mut slopes = power(closest);
        slopes.extend(
            alibi_slopes(2 * closest)
                .into_iter()
                .step_by(2)
                .take(n - closest),
        );
        slopes
    }
}

/// The fixed table b_h(r) at `h * length + r`; None for KERPLE's, which the
/// GPU computes from its parameters before each forward.
pub fn fixed_bias(cfg: &Config, length: usize) -> Option<Vec<f32>> {
    let slopes = alibi_slopes(cfg.heads);
    let value = |h: usize, r: usize| -> f32 {
        match bias(cfg) {
            Some(Bias::Alibi) => (-slopes[h] * r as f64) as f32,
            Some(Bias::Log) => (-(r as f64).ln_1p()) as f32,
            _ => 0.0,
        }
    };
    match bias(cfg) {
        Some(Bias::Alibi | Bias::Log | Bias::None) => Some(
            (0..cfg.heads * length)
                .map(|i| value(i / length, i % length))
                .collect(),
        ),
        _ => None,
    }
}

pub fn validate(cfg: &Config, p: &[f32]) -> Result<()> {
    if p.len() != total_count(cfg) || !p.iter().all(|v| v.is_finite()) {
        return Err("invalid parameter length or nonfinite value".into());
    }
    if is_complex(&cfg.architecture) {
        for [ar, bi, br, ai, kind] in pairs(cfg) {
            if kind == 0 && (p[ar as usize] != p[ai as usize] || p[bi as usize] != -p[br as usize])
            {
                return Err("broken complex-linear parameter constraint".into());
            }
        }
    }
    Ok(())
}

pub fn validate_state(cfg: &Config, p: &[f32], m: &[f32], v: &[f32]) -> Result<()> {
    validate(cfg, p)?;
    validate(cfg, m)?;
    if v.len() != p.len() || !v.iter().all(|x| x.is_finite() && *x >= 0.) {
        return Err("invalid AdamW second moment".into());
    }
    if is_complex(&cfg.architecture) {
        for [ar, bi, br, ai, kind] in pairs(cfg) {
            if cfg.phase_optimizer && v[ar as usize] != v[bi as usize] {
                return Err("phase AdamW requires a shared second moment".into());
            }
            if kind == 0 && (v[ar as usize] != v[ai as usize] || v[bi as usize] != v[br as usize]) {
                return Err("broken complex second-moment constraint".into());
            }
        }
    }
    Ok(())
}
