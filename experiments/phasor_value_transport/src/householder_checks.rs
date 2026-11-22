//! FP64 reflection matrices and directional derivatives on off-table inputs.
use super::*;
use householder::{CHANNELS, FEATURES, INPUTS};
use kernite_cuda::{Cuda, Grid};

fn reflect(slopes: &[f64], state: &[f64]) -> [f64; 4] {
    let normal = [1., slopes[0], slopes[1], slopes[2]];
    let norm: f64 = normal.iter().map(|v| v * v).sum();
    // Form all 16 matrix entries independently of the scan's vector formula.
    std::array::from_fn(|i| {
        (0..4)
            .map(|j| (f64::from(i == j) - 2. * normal[i] * normal[j] / norm) * state[j])
            .sum()
    })
}

fn reference(input: &[f64], start: &[f64], length: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(input.len() / INPUTS * FEATURES);
    for seq in input.chunks_exact(length * INPUTS) {
        let mut state = start.to_vec();
        for x in seq.as_chunks::<INPUTS>().0 {
            for (k, s) in state.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let new = reflect(&x[k * 3..k * 3 + 3], s);
                s.copy_from_slice(&new);
            }
            out.extend(state.iter().zip(start).map(|(a, b)| a - b));
        }
    }
    out
}

fn scan(length: usize) -> Result<()> {
    let batch = 3;
    let rows = batch * length;
    let mut rng = random::Random::new(863);
    let mut samples = |n| {
        (0..n)
            .map(|_| rng.gauss(0., 0.39) as f32)
            .collect::<Vec<_>>()
    };
    let mut input = samples(rows * INPUTS);
    let initial = samples(FEATURES);
    let upstream = samples(rows * FEATURES);
    // Exact repetitions also exercise zero slopes and cancellation.
    if length > 1 {
        for seq in input.chunks_exact_mut(length * INPUTS) {
            seq[..3].fill(0.);
            seq.copy_within(..INPUTS, INPUTS);
        }
    }
    let cuda = Cuda::new()?;
    let kernels = householder::kernels::Kernels::load(&cuda)?;
    let reducer = reflection_phase::kernels::Kernels::load(&cuda)?;
    let x = cuda.upload(&input)?;
    let base = cuda.upload(&initial)?;
    let g = cuda.upload(&upstream)?;
    let out = cuda.alloc::<f32>(rows * FEATURES)?;
    let previous = cuda.alloc::<f32>(rows * FEATURES)?;
    let grad = cuda.alloc::<f32>(rows * INPUTS)?;
    let partial = cuda.alloc::<f32>(batch * FEATURES)?;
    let dbase = cuda.alloc::<f32>(FEATURES)?;
    let launch = Grid::new(batch as u32, FEATURES as u32);
    unsafe {
        kernels.forward(
            launch,
            out.ptr(),
            previous.ptr(),
            x.ptr(),
            base.ptr(),
            length as u32,
        )?;
        kernels.backward(
            launch,
            grad.ptr(),
            partial.ptr(),
            previous.ptr(),
            x.ptr(),
            g.ptr(),
            length as u32,
        )?;
        reducer.sum_partials::<true>(
            Grid::new(1, FEATURES as u32),
            dbase.ptr(),
            partial.ptr(),
            batch as u32,
        )?;
    }
    let to_double = |v: &[f32]| v.iter().copied().map(f64::from).collect::<Vec<_>>();
    let x = to_double(&input);
    let base = to_double(&initial);
    let dx = to_double(&grad.read()?);
    let ds = to_double(&dbase.read()?);
    let actual = out.read()?;
    let forward = phase_checks::error(&actual, &reference(&x, &base, length), 2e-5)?;
    let mut norm_error = 0_f64;
    for (index, row) in actual.as_chunks::<FEATURES>().0.iter().enumerate() {
        for k in 0..CHANNELS {
            let norm: f64 = (0..4)
                .map(|j| {
                    let i = 4 * k + j;
                    (f64::from(row[i]) + base[i]).powi(2) - base[i].powi(2)
                })
                .sum();
            norm_error = norm_error.max(norm.abs());
        }
        if index % length == 1 && row.iter().any(|v| v.abs() > 2e-6) {
            return Err("identical consecutive normals failed to cancel".into());
        }
    }
    if norm_error > 3e-5 {
        return Err("Householder state norm drift".into());
    }
    let mut worst = 0_f64;
    for _ in 0..6 {
        let tangent_x = to_double(&samples(x.len()));
        let tangent_s = to_double(&samples(base.len()));
        let analytic: f64 = dx
            .iter()
            .chain(&ds)
            .zip(tangent_x.iter().chain(&tangent_s))
            .map(|(a, b)| a * b)
            .sum();
        let score = |eps: f64| {
            let perturb = |v: &[f64], d: &[f64]| {
                v.iter()
                    .zip(d)
                    .map(|(a, b)| a + eps * b)
                    .collect::<Vec<_>>()
            };
            reference(
                &perturb(&x, &tangent_x),
                &perturb(&base, &tangent_s),
                length,
            )
            .iter()
            .zip(&upstream)
            .map(|(a, &b)| a * f64::from(b))
            .sum::<f64>()
        };
        let numeric = (score(1e-5) - score(-1e-5)) / 2e-5;
        let error = (numeric - analytic).abs();
        if !error.is_finite() || error > 6e-4 * (1. + numeric.abs()) {
            return Err(format!(
                "Householder direction analytic {analytic} versus numeric {numeric}"
            )
            .into());
        }
        worst = worst.max(error);
    }
    println!(
        "Householder scan length={length} features={} max_absolute_forward={forward:.3e} norm_error={norm_error:.3e} max_direction_error={worst:.3e}",
        actual.len()
    );
    Ok(())
}

fn network(width: usize) -> Result<()> {
    let mut cfg = config(2)?;
    cfg.head = architecture::HEAD;
    cfg.width = width;
    cfg.heads = width / 32;
    cfg.gain = cfg.heads as f32 / width as f32;
    cfg.vocab = 19;
    cfg.max_length = 5;
    let first = architecture::start(&cfg);
    let input = architecture::memory_start(&cfg);
    let output = input + INPUTS * width;
    let initial = output + FEATURES * width;
    let end = layout::frequency_offset(&cfg);
    let mut weights = layout::initialize_model(&cfg, 54);
    let mut e = Engine::new(cfg.clone(), &weights)?;
    let mut control = phase_controls::stream_control(&cfg, &weights)?;
    phase_controls::compare_dense(&cfg, &mut e, &mut control, "zero Householder output")?;
    let tokens = [1, 3, 5, 7, 9, 9, 8, 6, 4, 2];
    let labels = [u32::MAX, 6, 2, 4, 8, 7, u32::MAX, 5, 3, 1];
    e.forward(&tokens, &labels, 2, 5)?;
    e.backward(2, 5)?;
    let grad = e.gradients.read()?;
    if grad[input..output]
        .iter()
        .chain(&grad[initial..end])
        .any(|&g| g != 0.)
        || !grad[output..initial].iter().any(|v| v.abs() > 1e-6)
    {
        return Err("incorrect zero-read reflection gradients".into());
    }
    let mut rng = random::Random::new(142);
    for v in &mut weights[output..initial] {
        *v = rng.gauss(0., 0.08) as f32;
    }
    e.parameters.write(&weights)?;
    e.forward(&tokens, &labels, 2, 5)?;
    let diagnostic = e.diagnostic_snapshot(tokens.len())?;
    let x = &diagnostic["embedding_normalized"];
    let steps: Vec<_> = x
        .chunks_exact(width)
        .flat_map(|row| {
            memory_checks::linear(
                &weights[input..output],
                &row.iter().copied().map(f64::from).collect::<Vec<_>>(),
            )
        })
        .collect();
    let base: Vec<_> = weights[initial..end]
        .iter()
        .copied()
        .map(f64::from)
        .collect();
    let read = reference(&steps, &base, 5);
    let mut expected = Vec::new();
    for (s, h) in read
        .as_chunks::<FEATURES>()
        .0
        .iter()
        .zip(diagnostic["readout_input"].chunks_exact(width))
    {
        let mut f = memory_checks::linear(&weights[output..initial], s);
        for (a, &b) in f.iter_mut().zip(h) {
            *a += f64::from(b);
        }
        expected.extend(memory_checks::linear(&weights[..cfg.vocab * width], &f));
    }
    let forward = phase_checks::error(&e.logits.read()?, &expected, 1e-5)?;
    let causal = memory_checks::prefix_error(&mut e, &tokens[..5])?;
    if causal > 2e-5 {
        return Err("future embedding changed a Householder prefix".into());
    }
    println!("Householder network width={width} logit_error={forward:.3e} prefix={causal:.3e}");
    let router = layout::router_offset(&cfg, 0);
    memory_checks::check_directions(
        &mut e,
        &weights,
        &tokens,
        &labels,
        5,
        vec![
            ("tied embedding", (0..cfg.vocab * width).collect()),
            ("backbone router", (router..router + 96 * width).collect()),
            ("stream read", (first..input).collect()),
            ("reflection slopes", (input..output).collect()),
            ("reflected read", (output..initial).collect()),
            ("initial state", (initial..end).collect()),
        ],
    )
}

pub fn run() -> Result<()> {
    if !precision::FP32_AFFINE_REFERENCE {
        return Err("Householder checks require FP32 reference affines".into());
    }
    for length in [1, 7, 257] {
        scan(length)?;
    }
    for width in [64, 128] {
        network(width)?;
    }
    Ok(())
}
