//! `validate CONTROL_ROOT OUT`: what must hold before the cohort starts.
//! CONTROL_ROOT is the recorded control arm, gpt's
//! `runs/main-model-phase-score-01/phase-score-lerope`; OUT is a new directory
//! that receives `validation.json` and the replayed cells.
//!
//! 1. Kernels. `biased_forward` with a zero table equals the control forward
//!    bit for bit at D = 16, 32 and 128. With a table b its probabilities are
//!    the control's times e^b, renormalized in f64, so b is added after the
//!    SSMax gain, and its output mixes the values with them. KERPLE's table,
//!    distance sums, parameter gradient and clamps match f64 references.
//! 2. Initialization. Every arm's parameters before the trailing group are
//!    the recorded control cell's `initial.f32` bytes.
//! 3. KERPLE's gradient in the whole model matches central differences of
//!    the loss. Rounding in the FP16 forward GEMMs moves central differences
//!    by percents, so this check gates only in the `fp32-affine-reference`
//!    build; the production build reports it.
//! 4. The control binary `main-phase-score-native`, built from the pinned gpt
//!    sources, has the control cohort's PTX, suite and plan hashes, and every
//!    source file it archives is the cohort's, byte for byte, except
//!    `UNCOMMITTED_MANIFEST`.
//! 5. Replays. The control arm, run by this binary's runner, reproduces
//!    recorded control cells: every observation, the end and the final splits.
//!
//! Items 4 and 5 run in the production build only: the reference build
//! trains nothing.
use crate::{Result, data, engine::Engine, layout, metrics, random::Random, runner};
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

/// Control cells replayed to their end: D = 16; D = 32 with one and with six
/// layers; D = 128.
const REPLAYS: [&str; 4] = [
    "suite-boolean-and-shift",
    "arithmetic-add",
    "basis-depth-hard",
    "recurrence-fibonacci-k1-c2-lgt",
];
/// Cases of the whole-model gradient check: two and six layers.
const GRADIENT_CASES: [&str; 2] = ["suite-boolean-and-shift", "basis-depth-hard"];

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
    report.insert("initialization".into(), initialization(control)?);
    let mut gradients = Vec::new();
    for case in GRADIENT_CASES {
        for arm in ["kerple-log", "kerple-power"] {
            gradients.push(gradient(control, case, arm)?);
        }
    }
    report.insert("gradients".into(), gradients.into());
    // The control's precision is the production build's; the FP32 reference
    // build checks the gradient alone.
    if !crate::precision::FP32_AFFINE_REFERENCE {
        report.insert(
            "control_binary".into(),
            control_binary(control, &out.join("control-archive"))?,
        );
        let mut replays = Vec::new();
        for case in REPLAYS {
            replays.push(replay(control, &out.join("replay"), case)?);
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

struct Shape {
    heads: usize,
    d: usize,
    batch: usize,
    length: usize,
}
impl Shape {
    fn rows(&self) -> usize {
        self.batch * self.length
    }
    fn width(&self) -> usize {
        self.heads * self.d
    }
    fn scores(&self) -> usize {
        self.batch * self.heads * self.length * self.length
    }
}
struct Forward {
    out: Vec<f32>,
    prob: Vec<f32>,
    imaginary: Vec<f32>,
}
struct Forwards {
    control: crate::phase_score_forward::kernels::Kernels,
    biased: crate::biased_forward::kernels::Kernels,
}

/// The control forward (`table = None`) or the biased one, as the engine
/// launches them.
fn forward(
    cuda: &Cuda,
    kernels: &Forwards,
    s: &Shape,
    qkv: &Buffer<f32>,
    coefficients: &Buffer<f32>,
    table: Option<&Buffer<f32>>,
) -> Result<Forward> {
    let out = cuda.upload(&vec![0f32; s.rows() * s.width()])?;
    let prob = cuda.upload(&vec![0f32; s.scores()])?;
    let imaginary = cuda.upload(&vec![0f32; s.scores()])?;
    let count = s.rows() * s.heads;
    let grid = || {
        Grid::new(
            (count / if s.d == 16 { 2 } else { 1 }).div_ceil(4) as u32,
            128,
        )
    };
    let (h, l, n) = (s.heads as u32, s.length as u32, count as u32);
    macro_rules! launch {
        ($d:literal) => {
            match table {
                None => kernels.control.forward::<$d>(
                    grid(),
                    out.ptr(),
                    prob.ptr(),
                    qkv.ptr(),
                    imaginary.ptr(),
                    coefficients.ptr(),
                    h,
                    l,
                    n,
                )?,
                Some(t) => kernels.biased.forward::<$d>(
                    grid(),
                    out.ptr(),
                    prob.ptr(),
                    qkv.ptr(),
                    imaginary.ptr(),
                    coefficients.ptr(),
                    t.ptr(),
                    h,
                    l,
                    n,
                )?,
            }
        };
    }
    // SAFETY: every buffer has the shape the kernels index; one stream.
    unsafe {
        match s.d {
            16 => launch!(16),
            32 => launch!(32),
            128 => launch!(128),
            _ => return Err("no forward instance".into()),
        }
    }
    Ok(Forward {
        out: out.read()?,
        prob: prob.read()?,
        imaginary: imaginary.read()?,
    })
}

fn forward_check(cuda: &Cuda, kernels: &Forwards, d: usize, heads: usize) -> Result<Value> {
    let s = Shape {
        heads,
        d,
        batch: 2,
        length: 19,
    };
    let (w, length) = (s.width(), s.length);
    let mut rng = Random::new(40000 + d as u32);
    let qkv_host = gauss(&mut rng, s.rows() * 3 * w);
    let coefficients_host = uniform(&mut rng, 2 * heads, -1., 1.);
    let table_host = uniform(&mut rng, heads * length, -4., 1.);
    let qkv = cuda.upload(&qkv_host)?;
    let coefficients = cuda.upload(&coefficients_host)?;
    let zero = cuda.upload(&vec![0f32; heads * length])?;
    let table = cuda.upload(&table_host)?;
    let reference = forward(cuda, kernels, &s, &qkv, &coefficients, None)?;
    let zeroed = forward(cuda, kernels, &s, &qkv, &coefficients, Some(&zero))?;
    let biased = forward(cuda, kernels, &s, &qkv, &coefficients, Some(&table))?;
    let zero_mismatches = bit_mismatches(&reference.out, &zeroed.out)
        + bit_mismatches(&reference.prob, &zeroed.prob)
        + bit_mismatches(&reference.imaginary, &zeroed.imaginary);
    let imaginary_mismatches = bit_mismatches(&reference.imaginary, &biased.imaginary);
    let (mut prob_error, mut out_error) = (0f64, 0f64);
    for b in 0..s.batch {
        for h in 0..heads {
            for t in 0..length {
                let row = ((b * heads + h) * length + t) * length;
                let weights: Vec<f64> = (0..=t)
                    .map(|key| {
                        reference.prob[row + key] as f64
                            * (table_host[h * length + t - key] as f64).exp()
                    })
                    .collect();
                let total: f64 = weights.iter().sum();
                for (key, weight) in weights.iter().enumerate() {
                    let error = (biased.prob[row + key] as f64 - weight / total).abs();
                    prob_error = prob_error.max(error);
                }
                for col in 0..d {
                    let value = |key: usize| {
                        qkv_host[(b * length + key) * 3 * w + 2 * w + h * d + col] as f64
                    };
                    let first = value(0);
                    let (mut mix, mut scale) = (first, first.abs());
                    for key in 0..=t {
                        let term = biased.prob[row + key] as f64 * (value(key) - first);
                        mix += term;
                        scale += term.abs();
                    }
                    let got = biased.out[(b * length + t) * w + h * d + col] as f64;
                    out_error = out_error.max((got - mix).abs() / scale.max(1e-30));
                }
            }
        }
    }
    Ok(
        json!({"d": d, "heads": heads, "batch": s.batch, "length": length,
        "zero_table_bit_mismatches": zero_mismatches,
        "biased_imaginary_bit_mismatches": imaginary_mismatches,
        "biased_probability_max_abs_error": prob_error,
        "biased_output_max_relative_error": out_error,
        "pass": zero_mismatches == 0 && imaginary_mismatches == 0
            && prob_error < 1e-5 && out_error < 1e-5}),
    )
}

/// b(r), d b / d r1 and d b / d r2 in f64.
fn kerple(power: bool, r1: f64, r2: f64, r: f64) -> (f64, f64, f64) {
    if power {
        let y = r.powf(r2);
        let shape = if r > 0. { -r1 * y * r.ln() } else { 0. };
        (-r1 * y, -y, shape)
    } else {
        let y = 1. + r2 * r;
        (-r1 * y.ln(), -y.ln(), -r1 * r / y)
    }
}

fn table_check(cuda: &Cuda, bias: &crate::bias::kernels::Kernels, power: bool) -> Result<Value> {
    let (heads, length) = (4usize, 37usize);
    let mut rng = Random::new(50000 + u32::from(power));
    let parameters_host = uniform(&mut rng, 2 * heads, 0.01, 2.0);
    let parameters = cuda.upload(&parameters_host)?;
    let table = cuda.upload(&vec![f32::NAN; heads * length])?;
    let count = heads * length;
    let grid = Grid::new(count.div_ceil(256) as u32, 256);
    // SAFETY: `table` holds `count` values and `parameters` two per head.
    unsafe {
        if power {
            bias.table::<true>(
                grid,
                table.ptr(),
                parameters.ptr(),
                length as u32,
                count as u32,
            )?;
        } else {
            bias.table::<false>(
                grid,
                table.ptr(),
                parameters.ptr(),
                length as u32,
                count as u32,
            )?;
        }
    }
    let got = table.read()?;
    let mut error = 0f64;
    for h in 0..heads {
        let (r1, r2) = (
            parameters_host[2 * h] as f64,
            parameters_host[2 * h + 1] as f64,
        );
        for r in 0..length {
            let (b, _, _) = kerple(power, r1, r2, r as f64);
            // lg2.approx has an absolute error near 1e-7: the floor 0.1 r1.
            let e = (got[h * length + r] as f64 - b).abs() / (b.abs() + 0.1 * r1);
            error = error.max(if e.is_nan() { f64::INFINITY } else { e });
        }
    }
    Ok(
        json!({"kind": if power {"power"} else {"log"}, "heads": heads, "length": length,
        "max_relative_error": error, "pass": error < 1e-5}),
    )
}

fn gradient_kernels_check(
    cuda: &Cuda,
    bias: &crate::bias::kernels::Kernels,
    d: usize,
) -> Result<Value> {
    let (batch, heads, length) = (3usize, 4usize, 23usize);
    let mut rng = Random::new(60000 + d as u32);
    let ds_host = gauss(&mut rng, batch * heads * length * length);
    let ds = cuda.upload(&ds_host)?;
    let parts = cuda.upload(&vec![f32::NAN; heads * length])?;
    let grid = || Grid::new((heads * length) as u32, 32);
    let (h32, l32, b32) = (heads as u32, length as u32, batch as u32);
    // SAFETY: `ds` holds batch * heads * length^2 values, `parts` heads * length.
    unsafe {
        match d {
            16 => bias.distance_sums::<16>(grid(), parts.ptr(), ds.ptr(), h32, l32, b32)?,
            32 => bias.distance_sums::<32>(grid(), parts.ptr(), ds.ptr(), h32, l32, b32)?,
            128 => bias.distance_sums::<128>(grid(), parts.ptr(), ds.ptr(), h32, l32, b32)?,
            _ => return Err("no distance_sums instance".into()),
        }
    }
    let sums = parts.read()?;
    let mut sum_error = 0f64;
    let mut origin_zero = true;
    for h in 0..heads {
        origin_zero &= sums[h * length].to_bits() == 0;
        for r in 1..length {
            let (mut sum, mut scale) = (0f64, 0f64);
            for b in 0..batch {
                for t in r..length {
                    let gain = ((t + 1) as f64).ln() / d as f64;
                    let v = ds_host[((b * heads + h) * length + t) * length + t - r] as f64 / gain;
                    sum += v;
                    scale += v.abs();
                }
            }
            let e = (sums[h * length + r] as f64 - sum).abs() / scale;
            sum_error = sum_error.max(if e.is_nan() { f64::INFINITY } else { e });
        }
    }
    let mut kinds = Vec::new();
    for power in [false, true] {
        let parameters_host = uniform(&mut rng, 2 * heads, 0.01, 2.0);
        let parameters = cuda.upload(&parameters_host)?;
        let gradient = cuda.upload(&vec![f32::NAN; 2 * heads])?;
        let launch = |first: u32| -> Result<()> {
            // SAFETY: two parameters and two gradients per head; `parts` as above.
            unsafe {
                if power {
                    bias.parameter_gradient::<true>(
                        Grid::new(1, 32),
                        gradient.ptr(),
                        parts.ptr(),
                        parameters.ptr(),
                        h32,
                        l32,
                        first,
                    )?;
                } else {
                    bias.parameter_gradient::<false>(
                        Grid::new(1, 32),
                        gradient.ptr(),
                        parts.ptr(),
                        parameters.ptr(),
                        h32,
                        l32,
                        first,
                    )?;
                }
            }
            Ok(())
        };
        launch(1)?;
        let once = gradient.read()?;
        launch(0)?;
        let twice = gradient.read()?;
        let mut error = 0f64;
        for h in 0..heads {
            let (r1, r2) = (
                parameters_host[2 * h] as f64,
                parameters_host[2 * h + 1] as f64,
            );
            let (mut g, mut scale) = ([0f64; 2], [0f64; 2]);
            for r in 1..length {
                let (_, d1, d2) = kerple(power, r1, r2, r as f64);
                let part = sums[h * length + r] as f64;
                g[0] += part * d1;
                g[1] += part * d2;
                scale[0] += (part * d1).abs();
                scale[1] += (part * d2).abs();
            }
            for j in 0..2 {
                let e = (once[2 * h + j] as f64 - g[j]).abs() / scale[j];
                error = error.max(if e.is_nan() { f64::INFINITY } else { e });
            }
        }
        let doubled = twice
            .iter()
            .zip(&once)
            .all(|(a, b)| a.to_bits() == (2. * b).to_bits());
        kinds.push(json!({"kind": if power {"power"} else {"log"},
            "max_relative_error": error, "accumulation_doubles_exactly": doubled,
            "pass": error < 1e-4 && doubled}));
    }
    Ok(
        json!({"d": d, "batch": batch, "heads": heads, "length": length,
        "distance_sums_max_relative_error": sum_error, "distance_zero_is_zero": origin_zero,
        "parameter_gradient": kinds, "pass": sum_error < 1e-5 && origin_zero}),
    )
}

fn project_check(cuda: &Cuda, bias: &crate::bias::kernels::Kernels, power: bool) -> Result<Value> {
    let values: Vec<f32> = vec![
        -1.0, 0.0, 0.005, 0.01, 0.5, 1.9, 2.0, 2.5, 3.0, 7.0, -0.0, 0.02,
    ];
    let buffer = cuda.upload(&values)?;
    // SAFETY: one thread per value, `values.len()` <= 32.
    unsafe {
        if power {
            bias.project::<true>(Grid::new(1, 32), buffer.ptr(), values.len() as u32)?;
        } else {
            bias.project::<false>(Grid::new(1, 32), buffer.ptr(), values.len() as u32)?;
        }
    }
    let got = buffer.read()?;
    let expected: Vec<f32> = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let v = v.max(0.01);
            if power && i % 2 == 1 { v.min(2.0) } else { v }
        })
        .collect();
    let mismatches = bit_mismatches(&got, &expected);
    Ok(
        json!({"kind": if power {"power"} else {"log"}, "inputs": values, "outputs": got,
        "pass": mismatches == 0}),
    )
}

fn kernels() -> Result<Value> {
    let cuda = Cuda::new()?;
    let forwards = Forwards {
        control: crate::phase_score_forward::kernels::Kernels::load(&cuda)?,
        biased: crate::biased_forward::kernels::Kernels::load(&cuda)?,
    };
    let bias = crate::bias::kernels::Kernels::load(&cuda)?;
    let mut forward_rows = Vec::new();
    for (d, heads) in [(16, 4), (32, 4), (128, 2)] {
        forward_rows.push(forward_check(&cuda, &forwards, d, heads)?);
    }
    let mut kerple_rows = Vec::new();
    for power in [false, true] {
        kerple_rows.push(table_check(&cuda, &bias, power)?);
        kerple_rows.push(project_check(&cuda, &bias, power)?);
    }
    for d in [16, 32, 128] {
        kerple_rows.push(gradient_kernels_check(&cuda, &bias, d)?);
    }
    Ok(json!({"forward": forward_rows, "kerple": kerple_rows}))
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
    let mut rows = Vec::new();
    for case in REPLAYS {
        let (row, seed, initial) = recorded(control, case)?;
        let task = data::build(&row)?;
        let cfg = runner::config(&row, &task, runner::CONTROL)?;
        let control_identical = bytes(&layout::initialize_model(&cfg, seed)) == initial;
        let mut arms = serde_json::Map::new();
        for arm in runner::ARMS {
            let cfg = runner::config(&row, &task, arm)?;
            let p = layout::initialize_model(&cfg, seed);
            let at = layout::frequency_offset(&cfg);
            let shared = initial.len() >= 4 * at && bytes(&p[..at]) == initial[..4 * at];
            let trailing = &p[at..];
            let within = |c: &[f32], r1: f32, r2: f32| {
                (0.01..=r1).contains(&c[0]) && (0.01..=r2).contains(&c[1])
            };
            let declared = trailing.len() == layout::frequency_count(&cfg)
                && match layout::bias(&cfg) {
                    Some(layout::Bias::KerpleLog) => trailing.chunks(2).all(|c| within(c, 2., 1.)),
                    Some(layout::Bias::KerplePower) => {
                        trailing.chunks(2).all(|c| within(c, 1., 2.))
                    }
                    _ => trailing.is_empty(),
                };
            arms.insert(
                arm.into(),
                json!({"shared_prefix_parameters": at, "shared_prefix_identical": shared,
                    "position_initial_values": trailing, "pass": shared && declared}),
            );
        }
        rows.push(
            json!({"case": case, "seed": seed, "control_identical": control_identical,
            "arms": arms, "pass": control_identical}),
        );
    }
    Ok(rows.into())
}

/// KERPLE's backward gradient against central differences of the training
/// loss on one batch, at the initial parameters of seed 0.
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
fn history(dir: &Path) -> Result<Vec<Value>> {
    fs::read(dir.join("history.jsonl"))?
        .split(|&c| c == b'\n')
        .filter(|s| !s.is_empty())
        .map(|s| Ok(without_seconds(serde_json::from_slice(s)?)))
        .collect()
}

fn replay(control: &Path, root: &Path, case: &str) -> Result<Value> {
    let manifest = runner::suite();
    let row = manifest["tasks"]
        .as_array()
        .ok_or("suite tasks")?
        .iter()
        .find(|r| r["name"] == case)
        .ok_or("unknown case")?;
    let theirs = control.join(case).join("seed0");
    let seed = read_json(&theirs.join("protocol.json"))?["seed"]
        .as_u64()
        .ok_or("missing seed")? as u32;
    runner::archive(root)?;
    runner::run_cell(root, row, seed, runner::CONTROL, false)?;
    let ours = root
        .join(runner::CONTROL)
        .join(case)
        .join(format!("seed{seed}"));
    let (a, b) = (history(&ours)?, history(&theirs)?);
    let first_difference = a.iter().zip(&b).position(|(x, y)| x != y);
    let (ra, rb) = (
        read_json(&ours.join("result.json"))?,
        read_json(&theirs.join("result.json"))?,
    );
    let keys = [
        "status",
        "completed",
        "examples",
        "epochs",
        "first99_epochs",
        "previous_observed_epochs",
        "final_splits",
    ];
    let differing: Vec<&str> = keys.into_iter().filter(|k| ra[k] != rb[k]).collect();
    let endpoint =
        without_seconds(ra["endpoint"].clone()) == without_seconds(rb["endpoint"].clone());
    let initial = fs::read(ours.join("initial.f32"))? == fs::read(theirs.join("initial.f32"))?;
    Ok(
        json!({"case": case, "seed": seed, "observations": a.len(), "recorded_observations": b.len(),
        "first_differing_observation": first_difference, "differing_result_keys": differing,
        "endpoint_identical": endpoint, "initial_identical": initial,
        "status": ra["status"], "first99_epochs": ra["first99_epochs"],
        "pass": a.len() == b.len() && first_difference.is_none() && differing.is_empty()
            && endpoint && initial}),
    )
}
