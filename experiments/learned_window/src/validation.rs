//! `validate CONTROL_ROOT OUT`: what must hold before the cohort starts.
//! CONTROL_ROOT is the recorded control arm, gpt's
//! `runs/main-model-phase-score-01/phase-score-lerope`; OUT is a new directory
//! that receives `validation.json` and the replayed cells.
//!
//! 1. Kernels. With a window no shorter than the sequence, every routed
//!    kernel equals the control's bit for bit at D = 16, 32 and 128, whatever
//!    the far operands hold: the forward's output, probabilities and
//!    imaginary parts, and the backward's adjoint of the rotated q, k and v,
//!    its score adjoint, intermediate sums and score parts. The far adjoint
//!    of q and k, filled with NaN before, is zero after.
//! 2. The engine's attention block beyond the window. The first layer's block
//!    at length 40, from random q, k and v, learned frequencies and score
//!    coordinates, against an f64 oracle that rotates each pair by its
//!    relative position alone: `t - key` within the window, `w` (ReRoPE) or
//!    `w + (t - key - w)/k` (Leaky ReRoPE) beyond it. The output, and the
//!    adjoints of q, k and v, of the learned frequencies and of the score
//!    coordinates against f64 central differences of `⟨G, out⟩`, at D = 16,
//!    32 and 128. The control runs the same check with `t - key` everywhere,
//!    which checks the oracle.
//! 3. Initialization. Every arm's parameters are the recorded control cell's
//!    `initial.f32` bytes: the arms change the position map alone.
//! 4. The learned frequencies' gradient in the whole model against central
//!    differences of the loss, for the control and both arms. Rounding in the
//!    FP16 forward GEMMs moves central differences by percents, so this check
//!    gates only in the `fp32-affine-reference` build; the production build
//!    reports it.
//! 5. The control binary `main-phase-score-native`, built from the pinned gpt
//!    sources, has the control cohort's PTX, suite and plan hashes, and every
//!    source file it archives is the cohort's, byte for byte, except
//!    `UNCOMMITTED_MANIFEST`.
//! 6. Replays. The control arm, run by this binary's runner, reproduces
//!    recorded control cells, and each arm reproduces the recorded control
//!    cells whose sequences all fit in the window: every observation, the end
//!    and the final splits.
//!
//! Items 5 and 6 run in the production build only: the reference build
//! trains nothing.
use crate::{
    Result, data,
    engine::{Engine, Position},
    layout, metrics,
    random::Random,
    runner,
};
use kernite_cuda::{Buffer, Cuda, Grid};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    process::Command,
};

/// The live manifest of gpt's main model as the cohort archived it from a
/// working tree: no gpt commit holds this version. No code reads it; only
/// `sources.rs` captures it, and gpt's own `profile.rs` exempts it from its
/// comparison with this cohort's sources.
const UNCOMMITTED_MANIFEST: &str = "research/main_model/model.json";

/// Control cells the control arm replays to their end: D = 16; D = 32 with
/// one and with six layers; D = 128.
const REPLAYS: [&str; 4] = [
    "suite-boolean-and-shift",
    "arithmetic-add",
    "basis-depth-hard",
    "recurrence-fibonacci-k1-c2-lgt",
];
/// Control cells each arm replays to their end: no sequence of theirs leaves
/// the window. D = 32 with one and with two layers; D = 16.
const NEAR_REPLAYS: [&str; 3] = [
    "arithmetic-add",
    "associative-divisor-n2-l1-w0",
    "gpt-probe-key-value-binding",
];
/// Cases of the whole-model gradient check: two and six layers.
const GRADIENT_CASES: [&str; 2] = ["suite-boolean-and-shift", "basis-depth-hard"];
/// Cases of the attention block check: D = 16, 32 and 128. The fibonacci
/// cell's sequences have two tokens, so its engine is built for the block's.
const BLOCK_CASES: [(&str, bool); 3] = [
    ("suite-boolean-and-shift", false),
    ("basis-depth-hard", false),
    ("recurrence-fibonacci-k1-c2-lgt", true),
];
const BLOCK_LENGTH: usize = 40;

pub fn run(control: &Path, out: &Path) -> Result<()> {
    if out.exists() {
        return Err(format!("refuse overwrite: {}", out.display()).into());
    }
    fs::create_dir_all(out)?;
    let mut report = serde_json::Map::new();
    report.insert("precision".into(), crate::precision::TAG.into());
    report.insert(
        "gradients_gate".into(),
        crate::precision::FP32_AFFINE_REFERENCE.into(),
    );
    report.insert("kernels".into(), kernels()?);
    let mut blocks = Vec::new();
    for (case, extend) in BLOCK_CASES {
        for arm in arms() {
            blocks.push(block(control, case, extend, arm)?);
        }
    }
    report.insert("attention_block".into(), blocks.into());
    report.insert("initialization".into(), initialization(control)?);
    let mut gradients = Vec::new();
    for case in GRADIENT_CASES {
        for arm in arms() {
            gradients.push(gradient(control, case, arm)?);
        }
    }
    report.insert("gradients".into(), gradients.into());
    // The control's precision is the production build's; the FP32 reference
    // build checks the kernels and the gradients alone.
    if !crate::precision::FP32_AFFINE_REFERENCE {
        report.insert(
            "control_binary".into(),
            control_binary(control, &out.join("control-archive"))?,
        );
        let mut replays = Vec::new();
        for case in REPLAYS {
            replays.push(replay(control, &out.join("replay"), case, runner::CONTROL)?);
        }
        for case in NEAR_REPLAYS {
            for arm in runner::ARMS {
                replays.push(replay(control, &out.join("replay"), case, arm)?);
            }
        }
        report.insert("replays".into(), replays.into());
    }
    let report = Value::Object(report);
    let failed = failures(&report);
    let summary = json!({"pass": failed.is_empty(), "failed": failed, "checks": report});
    fs::write(
        out.join("validation.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("{}", json!({"pass": failed.is_empty(), "failed": failed}));
    if failed.is_empty() {
        Ok(())
    } else {
        Err("validation failed".into())
    }
}

/// The control, then the arms.
fn arms() -> impl Iterator<Item = &'static str> {
    std::iter::once(runner::CONTROL).chain(runner::ARMS)
}

/// The JSON paths of every object with `"pass": false`.
fn failures(v: &Value) -> Vec<String> {
    fn walk(v: &Value, path: String, out: &mut Vec<String>) {
        match v {
            Value::Object(m) => {
                if m.get("pass") == Some(&Value::Bool(false)) {
                    out.push(path.clone());
                }
                for (k, x) in m {
                    walk(x, format!("{path}/{k}"), out);
                }
            }
            Value::Array(a) => {
                for (i, x) in a.iter().enumerate() {
                    walk(x, format!("{path}/{i}"), out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(v, String::new(), &mut out);
    out
}

fn gauss(rng: &mut Random, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.gauss(0., 1.) as f32).collect()
}
fn uniform(rng: &mut Random, n: usize, low: f64, high: f64) -> Vec<f32> {
    (0..n)
        .map(|_| (low + (high - low) * rng.random()) as f32)
        .collect()
}
fn bit_mismatches(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
        + a.len().abs_diff(b.len())
}
fn wide(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}
/// The largest deviation of `got` from `expected`, over the largest entry of
/// `expected`.
fn relative_error(got: &[f64], expected: &[f64]) -> f64 {
    let scale = expected.iter().fold(0f64, |m, x| m.max(x.abs()));
    let worst = got
        .iter()
        .zip(expected)
        .fold(0f64, |m, (g, e)| m.max((g - e).abs()));
    if got.len() == expected.len() && scale > 0. {
        worst / scale
    } else {
        f64::INFINITY
    }
}

/// The score table of the coordinates `a`, as gpt's `phase_score::coefficients`
/// lays it out: β = a/√(1 + a²) and dβ/da per head.
fn score_table(a: &[f32]) -> Vec<f32> {
    a.iter()
        .flat_map(|&a| {
            let s = (1. + a as f64 * a as f64).sqrt();
            [(a as f64 / s) as f32, s.powi(-3) as f32]
        })
        .collect()
}

struct Kernels {
    forward: crate::phase_score_forward::kernels::Kernels,
    query: crate::ordinary_attention::kernels::Kernels,
    key: crate::attention::kernels::Kernels,
    score: crate::phase_score::kernels::Kernels,
    routed_forward: crate::rerope_forward::kernels::Kernels,
    routed: crate::rerope_backward::kernels::Kernels,
}

/// The buffers one attention block's forward and backward write.
struct Block {
    out: Buffer<f32>,
    prob: Buffer<f32>,
    imaginary: Buffer<f32>,
    dqkv: Buffer<f32>,
    ds: Buffer<f32>,
    gd: Buffer<f32>,
    dv: Buffer<f32>,
    parts: Buffer<f32>,
    dfar: Buffer<f32>,
}
const BLOCK_BUFFERS: [&str; 8] = [
    "out",
    "prob",
    "imaginary",
    "dqkv",
    "ds",
    "gd",
    "dv",
    "parts",
];
impl Block {
    fn new(cuda: &Cuda, rows: usize, width: usize, heads: usize, length: usize) -> Result<Self> {
        let zeros = |n: usize| cuda.upload(&vec![0f32; n]);
        let scores = rows * heads * length;
        Ok(Self {
            out: zeros(rows * width)?,
            prob: zeros(scores)?,
            imaginary: zeros(scores)?,
            dqkv: zeros(rows * 3 * width)?,
            ds: zeros(scores)?,
            gd: zeros(rows * width)?,
            dv: zeros(rows * width)?,
            parts: zeros(rows * heads)?,
            dfar: cuda.upload(&vec![f32::NAN; rows * 3 * width])?,
        })
    }
    /// The buffers of `BLOCK_BUFFERS`, in its order.
    fn read(&self) -> Result<Vec<Vec<f32>>> {
        [
            &self.out,
            &self.prob,
            &self.imaginary,
            &self.dqkv,
            &self.ds,
            &self.gd,
            &self.dv,
            &self.parts,
        ]
        .iter()
        .map(|b| Ok(b.read()?))
        .collect()
    }
}

fn kernels() -> Result<Value> {
    let cuda = Cuda::new()?;
    let kernels = Kernels {
        forward: crate::phase_score_forward::kernels::Kernels::load(&cuda)?,
        query: crate::ordinary_attention::kernels::Kernels::load(&cuda)?,
        key: crate::attention::kernels::Kernels::load(&cuda)?,
        score: crate::phase_score::kernels::Kernels::load(&cuda)?,
        routed_forward: crate::rerope_forward::kernels::Kernels::load(&cuda)?,
        routed: crate::rerope_backward::kernels::Kernels::load(&cuda)?,
    };
    let mut rows = Vec::new();
    for (d, heads) in [(16, 4), (32, 4), (128, 2)] {
        rows.push(near_check(&cuda, &kernels, d, heads)?);
    }
    Ok(rows.into())
}

/// The control's forward and backward kernels, then the routed ones with a
/// window as long as the sequence and random far operands, in the engine's
/// order and grids.
fn near_check(cuda: &Cuda, k: &Kernels, d: usize, heads: usize) -> Result<Value> {
    let (batch, length) = (2, 19);
    let (width, rows) = (heads * d, batch * length);
    let mut rng = Random::new(40000 + d as u32);
    let qkv = cuda.upload(&gauss(&mut rng, rows * 3 * width))?;
    let far = cuda.upload(&gauss(&mut rng, rows * 3 * width))?;
    let dy = cuda.upload(&gauss(&mut rng, rows * width))?;
    let coefficients = cuda.upload(&score_table(&uniform(&mut rng, heads, -2., 2.)))?;
    let control = Block::new(cuda, rows, width, heads, length)?;
    let routed = Block::new(cuda, rows, width, heads, length)?;
    let (h, l, n, window) = (
        heads as u32,
        length as u32,
        (rows * heads) as u32,
        length as u32,
    );
    let head_grid = || {
        Grid::new(
            ((rows * heads) / if d == 16 { 2 } else { 1 }).div_ceil(4) as u32,
            128,
        )
    };
    let row_grid = || Grid::new((rows * heads).div_ceil(4) as u32, 128);
    let (c, r) = (&control, &routed);
    macro_rules! launch {
        ($d:literal) => {{
            k.forward.forward::<$d>(
                head_grid(),
                c.out.ptr(),
                c.prob.ptr(),
                qkv.ptr(),
                c.imaginary.ptr(),
                coefficients.ptr(),
                h,
                l,
                n,
            )?;
            k.query.query_backward::<$d>(
                head_grid(),
                c.dqkv.ptr(),
                c.ds.ptr(),
                c.gd.ptr(),
                c.dv.ptr(),
                dy.ptr(),
                c.prob.ptr(),
                qkv.ptr(),
                h,
                l,
                n,
            )?;
            k.key.key_backward::<$d>(
                head_grid(),
                c.dqkv.ptr(),
                c.ds.ptr(),
                c.gd.ptr(),
                c.dv.ptr(),
                c.prob.ptr(),
                qkv.ptr(),
                h,
                l,
                n,
            )?;
            k.score.query_correction::<$d>(
                row_grid(),
                c.dqkv.ptr(),
                c.parts.ptr(),
                c.ds.ptr(),
                c.imaginary.ptr(),
                qkv.ptr(),
                coefficients.ptr(),
                h,
                l,
                n,
            )?;
            k.score.key_correction::<$d>(
                row_grid(),
                c.dqkv.ptr(),
                c.ds.ptr(),
                c.imaginary.ptr(),
                qkv.ptr(),
                coefficients.ptr(),
                h,
                l,
                n,
            )?;
            k.routed_forward.forward::<$d>(
                head_grid(),
                r.out.ptr(),
                r.prob.ptr(),
                qkv.ptr(),
                far.ptr(),
                r.imaginary.ptr(),
                coefficients.ptr(),
                h,
                l,
                n,
                window,
            )?;
            k.routed.query_backward::<$d>(
                head_grid(),
                r.dqkv.ptr(),
                r.dfar.ptr(),
                r.ds.ptr(),
                r.gd.ptr(),
                r.dv.ptr(),
                dy.ptr(),
                r.prob.ptr(),
                qkv.ptr(),
                far.ptr(),
                h,
                l,
                n,
                window,
            )?;
            k.routed.key_backward::<$d>(
                head_grid(),
                r.dqkv.ptr(),
                r.dfar.ptr(),
                r.ds.ptr(),
                r.gd.ptr(),
                r.dv.ptr(),
                r.prob.ptr(),
                qkv.ptr(),
                far.ptr(),
                h,
                l,
                n,
                window,
            )?;
            k.routed.query_correction::<$d>(
                row_grid(),
                r.dqkv.ptr(),
                r.dfar.ptr(),
                r.parts.ptr(),
                r.ds.ptr(),
                r.imaginary.ptr(),
                qkv.ptr(),
                far.ptr(),
                coefficients.ptr(),
                h,
                l,
                n,
                window,
            )?;
            k.routed.key_correction::<$d>(
                row_grid(),
                r.dqkv.ptr(),
                r.dfar.ptr(),
                r.ds.ptr(),
                r.imaginary.ptr(),
                qkv.ptr(),
                far.ptr(),
                coefficients.ptr(),
                h,
                l,
                n,
                window,
            )?;
        }};
    }
    // SAFETY: every buffer has the dense shape of `batch` sequences of
    // `length` the kernels index; one stream.
    unsafe {
        match d {
            16 => launch!(16),
            32 => launch!(32),
            128 => launch!(128),
            _ => return Err("no attention instance".into()),
        }
    }
    let (ours, theirs) = (routed.read()?, control.read()?);
    let mut mismatches = serde_json::Map::new();
    for (name, (a, b)) in BLOCK_BUFFERS.iter().zip(ours.iter().zip(&theirs)) {
        mismatches.insert((*name).into(), bit_mismatches(a, b).into());
    }
    let total: u64 = mismatches.values().filter_map(Value::as_u64).sum();
    let dfar = routed.dfar.read()?;
    // dfar is [row][q, k, v][width]: the value's third is never written.
    let (mut far_nonzero, mut value_written) = (0, 0);
    for (i, third) in dfar.chunks(width).enumerate() {
        if i % 3 == 2 {
            value_written += third.iter().filter(|v| !v.is_nan()).count();
        } else {
            far_nonzero += third.iter().filter(|v| v.to_bits() != 0).count();
        }
    }
    Ok(
        json!({"d": d, "heads": heads, "batch": batch, "length": length, "window": window,
        "bit_mismatches": mismatches, "far_adjoint_nonzero": far_nonzero,
        "far_adjoint_value_entries_written": value_written,
        "pass": total == 0 && far_nonzero == 0 && value_written == 0}),
    )
}

/// The relative position the arm's map gives the query `t` and the key `key`.
fn relative(position: Position, t: usize, key: usize) -> f64 {
    let distance = t - key;
    match position {
        Position::Leaky { window, slope } if distance >= window => {
            window as f64 + (distance - window) as f64 / slope as f64
        }
        Position::ReRope { window } if distance >= window => window as f64,
        _ => distance as f64,
    }
}

/// The attention block in f64, scored from relative positions alone: the
/// query rotated by the pair's relative angle, the key not at all.
struct Oracle {
    position: Position,
    batch: usize,
    length: usize,
    heads: usize,
    d: usize,
    /// The adjoint of the output: the probe's loss is `⟨g, out⟩`.
    g: Vec<f64>,
}
impl Oracle {
    /// The output and `⟨g, out⟩` from q and k after their norm and v, `x`, the
    /// learned frequencies `alpha` and the score coordinates `a`.
    fn forward(&self, x: &[f64], alpha: &[f64], a: &[f64]) -> (Vec<f64>, f64) {
        let (length, d) = (self.length, self.d);
        let (width, half) = (self.heads * d, d / 2);
        let frequency: Vec<f64> = (0..half)
            .map(|l| 10000f64.powf(-2. * l as f64 / d as f64) * alpha[l].exp())
            .collect();
        let mut out = vec![0.; self.batch * length * width];
        let (mut total, mut scores, mut rotated) = (0., vec![0.; length], vec![0.; d]);
        for b in 0..self.batch {
            for (head, a) in a.iter().enumerate() {
                let beta = a / (1. + a * a).sqrt();
                let at = |t: usize, part: usize| ((b * length + t) * 3 + part) * width + head * d;
                for t in 0..length {
                    let gain = ((t + 1) as f64).ln() / d as f64;
                    let q = &x[at(t, 0)..at(t, 0) + d];
                    for key in 0..=t {
                        let k = &x[at(key, 1)..at(key, 1) + d];
                        let r = relative(self.position, t, key);
                        for l in 0..half {
                            let (sin, cos) = (r * frequency[l]).sin_cos();
                            rotated[l] = q[l] * cos - q[l + half] * sin;
                            rotated[l + half] = q[l + half] * cos + q[l] * sin;
                        }
                        let (mut re, mut im) = (0., 0.);
                        for l in 0..half {
                            re += rotated[l] * k[l] + rotated[l + half] * k[l + half];
                            im += rotated[l] * k[l + half] - rotated[l + half] * k[l];
                        }
                        scores[key] = gain * (re + beta * im * im / d as f64);
                    }
                    let max = scores[..=t]
                        .iter()
                        .fold(f64::NEG_INFINITY, |m, &s| m.max(s));
                    let weights: Vec<f64> = scores[..=t].iter().map(|s| (s - max).exp()).collect();
                    let sum: f64 = weights.iter().sum();
                    for col in 0..d {
                        let mix: f64 = (0..=t).map(|key| weights[key] * x[at(key, 2) + col]).sum();
                        let i = (b * length + t) * width + head * d + col;
                        out[i] = mix / sum;
                        total += self.g[i] * out[i];
                    }
                }
            }
        }
        (out, total)
    }
}

/// The first layer's attention block, run by the engine, against `Oracle`.
fn block(control: &Path, case: &str, extend: bool, arm: &str) -> Result<Value> {
    let (row, seed, _) = recorded(control, case)?;
    let task = data::build(&row)?;
    let mut cfg = runner::config(&row, &task, arm)?;
    let (batch, length) = (2, BLOCK_LENGTH);
    if extend {
        cfg.max_length = cfg.max_length.max(length);
    }
    if cfg.max_batch < batch || cfg.max_length < length {
        return Err("the block check's engine is too small".into());
    }
    let (heads, width, position) = (cfg.heads, cfg.width, cfg.position);
    let d = width / heads;
    let (frequencies, scores) = (layout::frequency_offset(&cfg), layout::score_offset(&cfg));
    let bands = layout::frequency_count(&cfg);
    if bands != d / 2 {
        return Err("LeRoPE shares one frequency per band".into());
    }
    let mut rng = Random::new(60000 + d as u32);
    let mut parameters = layout::initialize_model(&cfg, seed);
    parameters[frequencies..frequencies + bands]
        .copy_from_slice(&uniform(&mut rng, bands, -0.5, 0.5));
    parameters[scores..scores + heads].copy_from_slice(&uniform(&mut rng, heads, -2., 2.));
    let qkv = gauss(&mut rng, batch * length * 3 * width);
    let g = gauss(&mut rng, batch * length * width);
    let [x, out, dx, dalpha, dscore] =
        Engine::new(cfg, &parameters)?.attention_probe(&qkv, &g, batch, length)?;
    let oracle = Oracle {
        position,
        batch,
        length,
        heads,
        d,
        g: wide(&g),
    };
    let x64 = wide(&x);
    let alpha = wide(&parameters[frequencies..frequencies + bands]);
    let a = wide(&parameters[scores..scores + heads]);
    let (expected, _) = oracle.forward(&x64, &alpha, &a);
    let step = 1e-6;
    let central = |f: &dyn Fn(f64) -> f64| (f(step) - f(-step)) / (2. * step);
    let loss = |x: &[f64], alpha: &[f64], a: &[f64]| oracle.forward(x, alpha, a).1;
    let shifted = |v: &[f64], i: usize, e: f64| {
        let mut v = v.to_vec();
        v[i] += e;
        v
    };
    let alpha_difference: Vec<f64> = (0..bands)
        .map(|l| central(&|e| loss(&x64, &shifted(&alpha, l, e), &a)))
        .collect();
    let score_difference: Vec<f64> = (0..heads)
        .map(|h| central(&|e| loss(&x64, &alpha, &shifted(&a, h, e))))
        .collect();
    // Both sides of the window's edge, the first and last positions, q, k and
    // v; then random coordinates.
    let w = runner::WINDOW;
    let mut coordinates = Vec::new();
    for b in [0, batch - 1] {
        for t in [0, 1, w - 1, w, w + 1, length - 2, length - 1] {
            for part in 0..3 {
                let (head, col) = (rng.below(heads as u32), rng.below(d as u32));
                coordinates
                    .push(((b * length + t) * 3 + part) * width + head as usize * d + col as usize);
            }
        }
    }
    for _ in 0..22 {
        coordinates.push(rng.below(x.len() as u32) as usize);
    }
    let x_difference: Vec<f64> = coordinates
        .iter()
        .map(|&i| central(&|e| loss(&shifted(&x64, i, e), &alpha, &a)))
        .collect();
    let x_backward: Vec<f64> = coordinates.iter().map(|&i| dx[i] as f64).collect();
    let direction = gauss(&mut rng, x.len());
    let along = |e: f64| -> Vec<f64> {
        x64.iter()
            .zip(&direction)
            .map(|(x, u)| x + e * *u as f64)
            .collect()
    };
    let directional_difference = central(&|e| loss(&along(e), &alpha, &a));
    let directional: f64 = dx
        .iter()
        .zip(&direction)
        .map(|(g, u)| *g as f64 * *u as f64)
        .sum();
    let errors = [
        relative_error(&wide(&out), &expected),
        relative_error(&wide(&dalpha), &alpha_difference),
        relative_error(&wide(&dscore), &score_difference),
        relative_error(&x_backward, &x_difference),
        ((directional - directional_difference) / directional_difference).abs(),
    ];
    let far_pairs: usize = match position.window() {
        Some((window, _)) => (window..length).map(|t| t + 1 - window).sum(),
        None => 0,
    };
    Ok(
        json!({"case": case, "arm": arm, "d": d, "heads": heads, "batch": batch,
        "length": length, "far_pairs_per_head": far_pairs,
        "output_relative_error": errors[0],
        "frequency_gradient_relative_error": errors[1],
        "score_gradient_relative_error": errors[2],
        "qkv_gradient_relative_error": errors[3],
        "directional_relative_error": errors[4],
        "frequency_gradient": {"backward": dalpha, "central_difference": alpha_difference},
        "score_gradient": {"backward": dscore, "central_difference": score_difference},
        "qkv_coordinates": coordinates, "directional": [directional, directional_difference],
        "pass": errors[0] < 1e-4 && errors[1..].iter().all(|e| *e < 2e-3)
            && (arm == runner::CONTROL || far_pairs > 0)}),
    )
}

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
/// A recorded control cell's case row, seed and initial parameter bytes.
fn recorded(control: &Path, case: &str) -> Result<(Value, u32, Vec<u8>)> {
    let dir = control.join(case).join("seed0");
    let protocol = read_json(&dir.join("protocol.json"))?;
    let seed = protocol["seed"].as_u64().ok_or("missing seed")? as u32;
    Ok((
        protocol["case"].clone(),
        seed,
        fs::read(dir.join("initial.f32"))?,
    ))
}
fn bytes(p: &[f32]) -> Vec<u8> {
    p.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn initialization(control: &Path) -> Result<Value> {
    let cases: BTreeSet<&str> = REPLAYS.iter().chain(&NEAR_REPLAYS).copied().collect();
    let mut rows = Vec::new();
    for case in cases {
        let (row, seed, initial) = recorded(control, case)?;
        let task = data::build(&row)?;
        let mut identical = serde_json::Map::new();
        for arm in arms() {
            let cfg = runner::config(&row, &task, arm)?;
            identical.insert(
                arm.into(),
                (bytes(&layout::initialize_model(&cfg, seed)) == initial).into(),
            );
        }
        let pass = identical.values().all(|v| *v == Value::Bool(true));
        rows.push(json!({"case": case, "seed": seed, "identical": identical, "pass": pass}));
    }
    Ok(rows.into())
}

/// The learned frequencies' backward gradient against central differences of
/// the training loss on one batch, at the initial parameters of seed 0.
fn gradient(control: &Path, case: &str, arm: &str) -> Result<Value> {
    let (row, seed, _) = recorded(control, case)?;
    let task = data::build(&row)?;
    let cfg = runner::config(&row, &task, arm)?;
    let train = &task.splits["train"];
    let batch = cfg.max_batch.min(train.len());
    let parameters = layout::initialize_model(&cfg, seed);
    let (at, n, vocab) = (
        layout::frequency_offset(&cfg),
        layout::frequency_count(&cfg),
        cfg.vocab,
    );
    let mut engine = Engine::new(cfg, &parameters)?;
    let mut packed = metrics::Packed::default();
    let indices: Vec<u32> = (0..batch as u32).collect();
    packed.fill(train, &indices);
    let supervised = packed
        .labels
        .iter()
        .filter(|&&y| (y as usize) < vocab)
        .count()
        .max(1) as f64;
    let rows = batch * packed.length;
    let loss = |engine: &mut Engine, p: &[f32]| -> Result<f64> {
        engine.parameters.write(p)?;
        engine.forward(&packed.tokens, &packed.labels, batch, packed.length)?;
        Ok(engine.losses(rows)?.iter().map(|&v| v as f64).sum::<f64>() / supervised)
    };
    let center = loss(&mut engine, &parameters)?;
    engine.backward(batch, packed.length)?;
    let gradient: Vec<f64> = engine.gradients.read()?[at..at + n]
        .iter()
        .map(|&g| g as f64)
        .collect();
    let step = 1e-2f32;
    let mut coordinates = Vec::new();
    let mut estimates = Vec::new();
    for i in 0..n {
        let mut p = parameters.clone();
        p[at + i] = parameters[at + i] + step;
        let (up, high) = (loss(&mut engine, &p)?, p[at + i]);
        p[at + i] = parameters[at + i] - step;
        let (down, low) = (loss(&mut engine, &p)?, p[at + i]);
        let estimate = (up - down) / (high - low) as f64;
        estimates.push(estimate);
        coordinates
            .push(json!({"index": i, "backward": gradient[i], "central_difference": estimate}));
    }
    let norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
    let delta = 1e-2;
    let (mut high, mut low) = (parameters.clone(), parameters.clone());
    for i in 0..n {
        high[at + i] = (parameters[at + i] as f64 + delta * gradient[i] / norm) as f32;
        low[at + i] = (parameters[at + i] as f64 - delta * gradient[i] / norm) as f32;
    }
    let predicted: f64 = (0..n)
        .map(|i| gradient[i] * (high[at + i] - low[at + i]) as f64)
        .sum();
    let measured = loss(&mut engine, &high)? - loss(&mut engine, &low)?;
    let ratio = measured / predicted;
    let largest = gradient.iter().fold(0f64, |m, g| m.max(g.abs()));
    let worst = (0..n)
        .filter(|&i| gradient[i].abs() >= 0.05 * largest)
        .map(|i| (estimates[i] - gradient[i]).abs() / gradient[i].abs())
        .fold(0f64, f64::max);
    let met = norm > 0. && (ratio - 1.).abs() < 0.02 && worst < 0.05;
    let mut row = json!({"case": case, "arm": arm, "seed": seed, "batch": batch,
        "length": packed.length, "loss": center, "gradient_norm": norm, "step": step,
        "directional_ratio": ratio, "worst_relative_error_of_large_coordinates": worst,
        "coordinates": coordinates, "criterion_met": met});
    if crate::precision::FP32_AFFINE_REFERENCE {
        row["pass"] = met.into();
    }
    Ok(row)
}

fn control_binary(control: &Path, root: &Path) -> Result<Value> {
    let cohort = control.parent().ok_or("control root")?;
    let expected = read_json(&cohort.join("protocol.json"))?;
    let executable = std::env::current_exe()?.with_file_name("main-phase-score-native");
    let status = Command::new(&executable)
        .arg("archive")
        .arg(root)
        .status()?;
    if !status.success() {
        return Err("control archive failed".into());
    }
    let built = read_json(&root.join("protocol.json"))?;
    let keys = ["ptx_sha256", "suite_sha256", "plan_sha256"];
    let identical = keys
        .iter()
        .all(|k| built[k] == expected[k] && built[k].is_string());
    let (ours, recorded) = (tree(&root.join("source"))?, tree(&cohort.join("source"))?);
    let names: BTreeSet<&String> = ours.keys().chain(recorded.keys()).collect();
    let differing: Vec<&String> = names
        .into_iter()
        .filter(|&name| ours.get(name) != recorded.get(name))
        .collect();
    let sources = differing.iter().all(|name| *name == UNCOMMITTED_MANIFEST);
    Ok(json!({"executable": executable, "hashes": keys,
        "expected": keys.map(|k| expected[k].clone()), "built": keys.map(|k| built[k].clone()),
        "source_sha256": {"built": built["source_sha256"], "expected": expected["source_sha256"]},
        "source_files": ours.len(), "recorded_source_files": recorded.len(),
        "differing_source_files": differing, "exempt": UNCOMMITTED_MANIFEST,
        "pass": identical && sources}))
}

/// Every file under `dir`, by its path relative to `dir`.
fn tree(dir: &Path) -> Result<BTreeMap<String, Vec<u8>>> {
    let mut files = BTreeMap::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(at) = pending.pop() {
        for entry in fs::read_dir(&at)? {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
            } else {
                let name = path.strip_prefix(dir)?.to_string_lossy().into_owned();
                files.insert(name, fs::read(&path)?);
            }
        }
    }
    Ok(files)
}

fn without_seconds(mut v: Value) -> Value {
    if let Value::Object(m) = &mut v {
        m.remove("seconds");
    }
    v
}
pub(crate) fn history(dir: &Path) -> Result<Vec<Value>> {
    fs::read(dir.join("history.jsonl"))?
        .split(|&c| c == b'\n')
        .filter(|s| !s.is_empty())
        .map(|s| Ok(without_seconds(serde_json::from_slice(s)?)))
        .collect()
}
/// The result keys a replay and the analysis compare.
pub(crate) const RESULT_KEYS: [&str; 7] = [
    "status",
    "completed",
    "examples",
    "epochs",
    "first99_epochs",
    "previous_observed_epochs",
    "final_splits",
];
/// Whether the cells `ours` and `theirs` trained alike: the same observations
/// but for their clocks, the same result keys, end and initial parameters.
pub(crate) fn same_cell(ours: &Path, theirs: &Path) -> Result<Value> {
    let (a, b) = (history(ours)?, history(theirs)?);
    let first_difference = a.iter().zip(&b).position(|(x, y)| x != y);
    let (ra, rb) = (
        read_json(&ours.join("result.json"))?,
        read_json(&theirs.join("result.json"))?,
    );
    let differing: Vec<&str> = RESULT_KEYS.into_iter().filter(|k| ra[k] != rb[k]).collect();
    let endpoint =
        without_seconds(ra["endpoint"].clone()) == without_seconds(rb["endpoint"].clone());
    let initial = fs::read(ours.join("initial.f32"))? == fs::read(theirs.join("initial.f32"))?;
    Ok(
        json!({"observations": a.len(), "recorded_observations": b.len(),
        "first_differing_observation": first_difference, "differing_result_keys": differing,
        "endpoint_identical": endpoint, "initial_identical": initial,
        "status": ra["status"], "first99_epochs": ra["first99_epochs"],
        "pass": a.len() == b.len() && first_difference.is_none() && differing.is_empty()
            && endpoint && initial}),
    )
}

/// The arm `arm` trains the recorded control cell `case`, seed 0, to its end.
fn replay(control: &Path, root: &Path, case: &str, arm: &str) -> Result<Value> {
    let manifest = runner::suite();
    let row = manifest["tasks"]
        .as_array()
        .ok_or("suite tasks")?
        .iter()
        .find(|r| r["name"] == case)
        .ok_or("unknown case")?;
    let task = data::build(row)?;
    let max_length = runner::config(row, &task, arm)?.max_length;
    if arm != runner::CONTROL && max_length > runner::WINDOW {
        return Err(format!("{case} reaches beyond the window: no recorded cell to replay").into());
    }
    let theirs = control.join(case).join("seed0");
    let seed = read_json(&theirs.join("protocol.json"))?["seed"]
        .as_u64()
        .ok_or("missing seed")? as u32;
    runner::archive(root)?;
    runner::run_cell(root, row, seed, arm, false)?;
    let ours = root.join(arm).join(case).join(format!("seed{seed}"));
    let mut row = same_cell(&ours, &theirs)?;
    row["case"] = case.into();
    row["arm"] = arm.into();
    row["seed"] = seed.into();
    row["max_length"] = max_length.into();
    Ok(row)
}
