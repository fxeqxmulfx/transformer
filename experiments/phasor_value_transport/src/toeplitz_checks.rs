//! Dense FP64 matrices, independent pullbacks and the active-interval shift witness.
use super::*;
use kernite_cuda::{Cuda, Grid};
use memory_checks::linear;

// Materialize each complex affine matrix entry before multiplying the value.
fn reference(parts: &[Vec<f64>; 4], d: usize, length: usize) -> Vec<f64> {
    let [v, stencil, gain, bias] = parts;
    let mut out = v.clone();
    for (r, value) in v
        .chunks_exact(2 * d)
        .enumerate()
        .filter(|(r, _)| r % length != 0)
    {
        let s = (r - 1) * (2 * d - 1);
        let h = (r - 1) * 2 * d;
        for j in 0..d {
            let (mut re, mut im) = (bias[h + j], bias[h + d + j]);
            for k in 0..d {
                let w = f64::from(k == j) + stencil[s + d - 1 + k - j];
                let a = (1. + gain[h + j]) * w;
                let b = gain[h + d + j] * w;
                re += a * value[k] - b * value[d + k];
                im += b * value[k] + a * value[d + k];
            }
            out[r * 2 * d + j] = re;
            out[r * 2 * d + d + j] = im;
        }
    }
    out
}

fn raw(d: usize, length: usize) -> Result<()> {
    let rows = 2 * length;
    let mut rng = random::Random::new(827);
    let mut sample = |n| {
        (0..n)
            .map(|_| rng.gauss(0., 0.29) as f32)
            .collect::<Vec<_>>()
    };
    let parts = [2 * d, 2 * d - 1, 2 * d, 2 * d].map(|n| sample(rows * n));
    let g = sample(rows * 2 * d);
    let cuda = Cuda::new()?;
    let kernels = toeplitz::kernels::Kernels::load(&cuda)?;
    let inputs = parts
        .iter()
        .map(|a| cuda.upload(a))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let grads = parts
        .iter()
        .map(|a| cuda.alloc::<f32>(a.len()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let upstream = cuda.upload(&g)?;
    let mixed = cuda.alloc::<f32>(g.len())?;
    let dmixed = cuda.alloc::<f32>(g.len())?;
    let out = cuda.alloc::<f32>(g.len())?;
    let launch = Grid::new(rows as u32, d as u32);
    let [v, k, gain, bias] = inputs.as_slice() else {
        unreachable!()
    };
    let [dv, dk, dg, db] = grads.as_slice() else {
        unreachable!()
    };
    unsafe {
        kernels.forward(
            launch,
            out.ptr(),
            mixed.ptr(),
            v.ptr(),
            k.ptr(),
            gain.ptr(),
            bias.ptr(),
            d as u32,
            length as u32,
        )?;
        kernels.diagonal_back(
            launch,
            dmixed.ptr(),
            dg.ptr(),
            db.ptr(),
            upstream.ptr(),
            mixed.ptr(),
            gain.ptr(),
            d as u32,
            length as u32,
        )?;
        kernels.value_back(
            launch,
            dv.ptr(),
            dmixed.ptr(),
            k.ptr(),
            d as u32,
            length as u32,
        )?;
        kernels.stencil_back(
            Grid::new(rows as u32, (2 * d - 1) as u32),
            dk.ptr(),
            dmixed.ptr(),
            v.ptr(),
            d as u32,
            length as u32,
        )?;
    }
    let fp64 = parts.map(|x| x.into_iter().map(f64::from).collect::<Vec<_>>());
    let forward = phase_checks::error(&out.read()?, &reference(&fp64, d, length), 3e-5)?;
    let mut worst = 0_f64;
    for (group, grad) in grads.iter().enumerate() {
        let actual = grad.read()?;
        for _ in 0..4 {
            let direction = sample(actual.len());
            let analytic: f64 = actual
                .iter()
                .zip(&direction)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let score = |step: f64| {
                let mut changed = fp64.clone();
                for (a, &b) in changed[group].iter_mut().zip(&direction) {
                    *a += step * f64::from(b);
                }
                reference(&changed, d, length)
                    .iter()
                    .zip(&g)
                    .map(|(a, &b)| a * f64::from(b))
                    .sum::<f64>()
            };
            let numeric = (score(1e-4) - score(-1e-4)) / 2e-4;
            let error = (analytic - numeric).abs();
            if !error.is_finite() || error > 5e-5 * (1. + numeric.abs()) {
                return Err(format!(
                    "Toeplitz derivative group {group}: {analytic} versus {numeric}"
                )
                .into());
            }
            worst = worst.max(error);
        }
    }
    println!(
        "Toeplitz width={d} length={length} max_forward={forward:.3e} max_direction_error={worst:.3e}"
    );
    Ok(())
}

fn rotation_witness() -> Result<()> {
    let (d, n, shift) = (16usize, 11usize, 4usize);
    let mut weights = vec![0.; 2 * (2 * d - 1)];
    weights[d - 1] = -1.;
    weights[d - 1 + shift] = 1.;
    weights[d - 1 + shift - n] = 1.;
    let mut value = vec![0.; 4 * d];
    for k in 0..n {
        value[2 * d + k] = (1 + k) as f32;
        value[3 * d + k] = -(k as f32) - 0.5;
    }
    let cuda = Cuda::new()?;
    let kernels = toeplitz::kernels::Kernels::load(&cuda)?;
    let stencil = cuda.upload(&weights)?;
    let zero = cuda.upload(&vec![0.; 4 * d])?;
    let input = cuda.upload(&value)?;
    let mixed = cuda.alloc::<f32>(4 * d)?;
    let out = cuda.alloc::<f32>(4 * d)?;
    unsafe {
        kernels.forward(
            Grid::new(2, d as u32),
            out.ptr(),
            mixed.ptr(),
            input.ptr(),
            stencil.ptr(),
            zero.ptr(),
            zero.ptr(),
            d as u32,
            2,
        )?;
    }
    let result = out.read()?;
    for j in 0..n {
        for offset in [2 * d, 3 * d] {
            if result[offset + j] != value[offset + (j + shift) % n] {
                return Err("padded interval shift did not match its Lean witness".into());
            }
        }
    }
    Ok(())
}

fn network(d: usize) -> Result<()> {
    let mut cfg = config(2)?;
    cfg.head = architecture::HEAD;
    cfg.width = d;
    cfg.heads = d / 32;
    cfg.gain = cfg.heads as f32 / d as f32;
    cfg.vocab = 19;
    cfg.max_length = 5;
    let first = architecture::start(&cfg);
    let input = architecture::memory_start(&cfg);
    let output = input + 2 * d * d;
    let stencil = output + 2 * d * d;
    let gain = stencil + (2 * d - 1) * d;
    let bias = gain + 2 * d * d;
    let end = layout::frequency_offset(&cfg);
    if bias + 2 * d * d != end {
        return Err("Toeplitz layout mismatch".into());
    }
    let weights = layout::initialize_model(&cfg, 54);
    let mut e = Engine::new(cfg.clone(), &weights)?;
    let tokens = [1, 3, 5, 7, 9, 9, 8, 6, 4, 2];
    let labels = [u32::MAX, 6, 2, 4, 8, 7, u32::MAX, 5, 3, 1];
    e.forward(&tokens, &labels, 2, 5)?;
    let snapshot = e.diagnostic_snapshot(tokens.len())?;
    let project = |name: &str, start: usize, end: usize| {
        snapshot[name]
            .chunks_exact(d)
            .flat_map(|x| {
                linear(
                    &weights[start..end],
                    &x.iter().copied().map(f64::from).collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    };
    let parts = [
        project("embedding_normalized", input, output),
        project("readout_input", stencil, gain),
        project("readout_input", gain, bias),
        project("readout_input", bias, end),
    ];
    let expected: Vec<_> = reference(&parts, d, 5)
        .chunks_exact(2 * d)
        .flat_map(|x| {
            let h = linear(&weights[output..stencil], x);
            linear(&weights[..cfg.vocab * d], &h)
        })
        .collect();
    let forward = phase_checks::error(&e.logits.read()?, &expected, 3e-5)?;
    let causal = memory_checks::prefix_error(&mut e, &tokens[..5])?;
    if causal > 2e-5 {
        return Err("future embedding changed a Toeplitz prefix".into());
    }
    println!("Toeplitz network width={d} max_logit_error={forward:.3e} prefix={causal:.3e}");
    let router = layout::router_offset(&cfg, 0);
    memory_checks::check_directions(
        &mut e,
        &weights,
        &tokens,
        &labels,
        5,
        [
            ("tied embedding", 0..cfg.vocab * d),
            ("backbone router", router..router + 96 * d),
            ("stream read", first..input),
            ("current values", input..output),
            ("output", output..stencil),
            ("context stencil", stencil..gain),
            ("context gain", gain..bias),
            ("context bias", bias..end),
        ]
        .into_iter()
        .map(|(name, range)| (name, range.collect()))
        .collect(),
    )
}

pub fn run() -> Result<()> {
    if !precision::FP32_AFFINE_REFERENCE {
        return Err("Toeplitz checks require FP32 reference affines".into());
    }
    for (d, l) in [(7, 1), (13, 9), (32, 257)] {
        raw(d, l)?;
    }
    rotation_witness()?;
    for width in [64, 128] {
        network(width)?;
    }
    Ok(())
}
