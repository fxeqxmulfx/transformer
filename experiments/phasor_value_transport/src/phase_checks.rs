//! Independent FP64 prefix sums, reverse sums and masked classification
//! gradients for the learned phase memory from Transformer.Phasor.PhasePrefixRead.
use super::*;
use head::MEMORY_WIDTH as M;
use kernite_cuda::{Cuda, Grid};
use memory_checks::linear;

fn error(native: &[f32], expected: &[f64], tolerance: f64) -> Result<f64> {
    if native.is_empty() || native.len() != expected.len() {
        return Err("phase reference shapes differ".into());
    }
    let mut worst = 0_f64;
    for (&x, &y) in native.iter().zip(expected) {
        let diff = (f64::from(x) - y).abs();
        if !diff.is_finite() || diff > tolerance {
            return Err(format!("phase error {diff:.3e} exceeds {tolerance:.3e}").into());
        }
        worst = worst.max(diff);
    }
    Ok(worst)
}

fn scan(length: usize) -> Result<()> {
    let batch = 3;
    let rows = batch * length;
    let cuda = Cuda::new()?;
    let kernel = phase_prefix::kernels::Kernels::load(&cuda)?;
    let mut rng = random::Random::new(903);
    let mut samples = |n| {
        (0..n)
            .map(|_| rng.gauss(0., 0.31) as f32)
            .collect::<Vec<_>>()
    };
    let increments = samples(rows * M);
    let upstream = samples(rows * 2 * M);
    let saved = cuda.upload(&increments)?;
    let g = cuda.upload(&upstream)?;
    let read = cuda.alloc::<f32>(upstream.len())?;
    let grad = cuda.alloc::<f32>(increments.len())?;
    let grid = Grid::new(batch as u32, M as u32);
    // One block per full sequence. Each lane has its own projected channel.
    unsafe {
        kernel.forward(grid, read.ptr(), saved.ptr(), length as u32)?;
        kernel.backward(grid, grad.ptr(), saved.ptr(), g.ptr(), length as u32)?;
    }
    let mut features = vec![0.; upstream.len()];
    let mut derivative = vec![0.; increments.len()];
    // Each read contributes directly to all earlier inputs. This oracle
    // never carries an adjoint across rows and resets at batch boundaries.
    for row in 0..rows {
        let start = row - row % length;
        for k in 0..M {
            let angle: f64 = (start..=row)
                .map(|i| f64::from(increments[i * M + k]))
                .sum();
            let (s, c) = angle.sin_cos();
            let offset = row * 2 * M + k;
            features[offset] = s;
            features[offset + M] = c - 1.;
            let g = f64::from(upstream[offset]) * c - f64::from(upstream[offset + M]) * s;
            for i in start..=row {
                derivative[i * M + k] += g;
            }
        }
    }
    let forward = error(&read.read()?, &features, 2e-5)?;
    let backward = error(&grad.read()?, &derivative, 1e-4)?;
    println!(
        "phase scan length={length} reads={} gradients={} max_absolute_forward={forward:.3e} backward={backward:.3e}",
        features.len(),
        derivative.len()
    );
    Ok(())
}

fn network(width: usize) -> Result<()> {
    let tokens = [1, 4, 2, 4, 3, 7, 8, 1, 4, 2, 5, 4];
    let labels = [
        u32::MAX,
        5,
        u32::MAX,
        7,
        4,
        3,
        u32::MAX,
        u32::MAX,
        1,
        4,
        u32::MAX,
        6,
    ];
    let length = tokens.len() / 2;
    let mut cfg = config(2)?;
    cfg.width = width;
    cfg.heads = width / 32;
    cfg.gain = cfg.heads as f32 / width as f32;
    cfg.vocab = 17;
    cfg.max_length = length;
    cfg.head = head::Head::PhasePrefix;
    let mut weights = layout::initialize_model(&cfg, 3);
    let first = architecture::start(&cfg);
    let increments = architecture::memory_start(&cfg);
    let output = increments + M * width;
    let end = first + cfg.head.parameters(&cfg);
    let mut e = Engine::new(cfg.clone(), &weights)?;
    e.forward(&tokens, &labels, 2, length)?;
    e.backward(2, length)?;
    let grad = e.gradients.read()?;
    let output_indices =
        |half: usize| (output..end).filter(move |&i| (i - output) % (2 * M) / M == half);
    if grad[increments..output].iter().any(|&g| g != 0.)
        || (0..2).any(|half| !output_indices(half).any(|i| grad[i].abs() > 1e-7))
    {
        return Err("zero phase output maps produced incorrect parameter gradients".into());
    }
    let mut rng = random::Random::new(71);
    for w in &mut weights[first..increments] {
        *w = rng.gauss(0., 0.12) as f32;
    }
    for w in &mut weights[output..end] {
        *w = rng.gauss(0., 0.08) as f32;
    }
    e.parameters.write(&weights)?;
    e.forward(&tokens, &labels, 2, length)?;
    let actual = e.logits.read()?;
    let hidden = e.diagnostic_snapshot(tokens.len())?;
    let mut expected = Vec::new();
    for seq in hidden["readout_input"].chunks_exact(width * length) {
        let mut angles = [0.; M];
        for h in seq.chunks_exact(width) {
            let h: Vec<_> = h.iter().copied().map(f64::from).collect();
            let step = linear(&weights[increments..output], &h);
            let mut pair = [0.; 2 * M];
            for k in 0..M {
                angles[k] += step[k];
                let (s, c) = angles[k].sin_cos();
                pair[k] = s;
                pair[M + k] = c - 1.;
            }
            let residual = linear(&weights[output..end], &pair);
            let merged: Vec<_> = h.iter().zip(residual).map(|(x, y)| x + y).collect();
            expected.extend(linear(&weights[..cfg.vocab * width], &merged));
        }
    }
    let logits = error(&actual, &expected, 1e-5)?;
    let prefix = memory_checks::prefix_error(&mut e, &tokens[..length])?;
    if prefix > 2e-5 {
        return Err("prefix changed a phase-model logit".into());
    }
    let router = layout::router_offset(&cfg, 0);
    let groups = vec![
        ("tied embedding", (0..cfg.vocab * width).collect()),
        ("backbone router", (router..router + 96 * width).collect()),
        ("stream read", (first..increments).collect()),
        ("phase increment", (increments..output).collect()),
        ("sine output", output_indices(0).collect()),
        ("cosine output", output_indices(1).collect()),
    ];
    println!("phase network width={width} max_absolute_logits={logits:.3e} prefix={prefix:.3e}");
    memory_checks::check_directions(&mut e, &weights, &tokens, &labels, length, groups)
}

pub fn run() -> Result<()> {
    if !precision::FP32_AFFINE_REFERENCE {
        return Err("phase memory checks require --features fp32-affine-reference".into());
    }
    for length in [1, 7, 257] {
        scan(length)?;
    }
    for width in [64, 128] {
        network(width)?;
    }
    Ok(())
}
