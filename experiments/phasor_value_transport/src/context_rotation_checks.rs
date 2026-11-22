//! Independent FP64 checks for the exclusive rotation and its two input paths.
use super::*;
use kernite_cuda::{Cuda, Grid};

fn coordinates(width: usize, k: usize, adjacent: bool) -> (usize, usize) {
    if adjacent {
        (2 * k, 2 * k + 1)
    } else {
        (k, k + width / 2)
    }
}

fn scan<const ADJACENT: bool>(width: usize, length: usize) -> Result<()> {
    let batch = 3;
    let pairs = width / 2;
    let rows = batch * length;
    let cuda = Cuda::new()?;
    let kernel = context_rotation::kernels::Kernels::load(&cuda)?;
    let mut rng = random::Random::new(235);
    let mut sample = |n| {
        (0..n)
            .map(|_| rng.gauss(0., 0.31) as f32)
            .collect::<Vec<_>>()
    };
    let x = sample(rows * width);
    let g = sample(x.len());
    let increments = sample(rows * pairs);
    let context = cuda.upload(&x)?;
    let saved = cuda.upload(&increments)?;
    let dx = cuda.upload(&g)?;
    let output = cuda.alloc::<f32>(x.len())?;
    let dsteps = cuda.alloc::<f32>(increments.len())?;
    let grid = Grid::new(batch as u32, pairs as u32);
    unsafe {
        kernel.forward::<ADJACENT>(
            grid,
            output.ptr(),
            saved.ptr(),
            context.ptr(),
            width as u32,
            length as u32,
        )?;
        kernel.backward::<ADJACENT>(
            grid,
            dx.ptr(),
            dsteps.ptr(),
            output.ptr(),
            saved.ptr(),
            width as u32,
            length as u32,
        )?;
    }
    let mut forward = vec![0.; x.len()];
    let mut direct = vec![0.; x.len()];
    let mut step_grad = vec![0.; increments.len()];
    for row in 0..rows {
        let first = row - row % length;
        for k in 0..pairs {
            let angle: f64 = (first..row)
                .map(|i| f64::from(increments[i * pairs + k]))
                .sum();
            let (s, c) = angle.sin_cos();
            let (i, j) = coordinates(width, k, ADJACENT);
            let lo = row * width + i;
            let hi = row * width + j;
            let (u, v) = (f64::from(x[lo]), f64::from(x[hi]));
            let (gu, gv) = (f64::from(g[lo]), f64::from(g[hi]));
            forward[lo] = c * u - s * v;
            forward[hi] = s * u + c * v;
            direct[lo] = c * gu + s * gv;
            direct[hi] = c * gv - s * gu;
            // Differentiate the sine and cosine expression directly. Current
            // and later increments have no influence on this output.
            let torque = gu * (-s * u - c * v) + gv * (c * u - s * v);
            for i in first..row {
                step_grad[i * pairs + k] += torque;
            }
        }
    }
    let read = phase_checks::error(&output.read()?, &forward, 2e-5)?;
    let context_grad = phase_checks::error(&dx.read()?, &direct, 2e-5)?;
    let angle_grad = phase_checks::error(&dsteps.read()?, &step_grad, 1e-4)?;
    println!(
        "context rotation adjacent={ADJACENT} width={width} length={length} reads={} max_absolute_forward={read:.3e} context_gradient={context_grad:.3e} increment_gradient={angle_grad:.3e}",
        x.len()
    );
    Ok(())
}

fn network(width: usize) -> Result<()> {
    let mut cfg = config(2)?;
    cfg.width = width;
    cfg.heads = width / 32;
    cfg.gain = cfg.heads as f32 / width as f32;
    cfg.vocab = 29;
    cfg.max_length = 5;
    cfg.head = architecture::HEAD;
    let mut weights = layout::initialize_model(&cfg, 17);
    let first = architecture::start(&cfg);
    let projection = architecture::memory_start(&cfg);
    let stage_size = (width / 2) * width;
    let after_first = projection + stage_size;
    let end = layout::frequency_offset(&cfg);
    let toks = [1, 9, 4, 9, 5, 11, 3, 11, 2, 12];
    let labels = [u32::MAX, 5, 13, u32::MAX, 7, 2, u32::MAX, 1, 4, 6];
    let mut e = Engine::new(cfg.clone(), &weights)?;
    e.forward(&toks, &labels, 2, 5)?;
    e.backward(2, 5)?;
    let grad = e.gradients.read()?;
    if weights[projection..end].iter().any(|&w| w != 0.)
        || grad[projection..end]
            .chunks_exact(stage_size)
            .any(|stage| !stage.iter().any(|x| x.abs() > 1e-5))
    {
        return Err("zero rotation projection must receive a nonzero masked-loss gradient".into());
    }
    let mut rng = random::Random::new(365);
    for w in &mut weights[first..end] {
        *w = rng.gauss(0., 0.07) as f32;
    }
    if cfg.head == head::Head::CrossedContext {
        let mut single_cfg = cfg.clone();
        single_cfg.head = head::Head::ContextRotation;
        let single_weights = [&weights[..after_first], &weights[end..]].concat();
        let mut single = Engine::new(single_cfg, &single_weights)?;
        let mut zero = weights.clone();
        zero[after_first..end].fill(0.);
        e.parameters.write(&zero)?;
        phase_controls::compare_dense(&cfg, &mut e, &mut single, "zero adjacent rotation")?;
    }
    e.parameters.write(&weights)?;
    e.forward(&toks, &labels, 2, 5)?;
    let actual = e.logits.read()?;
    let states = e.diagnostic_snapshot(toks.len())?;
    let pairs = width / 2;
    let mut expected = Vec::new();
    for (xs, hs) in states["embedding_normalized"]
        .chunks_exact(5 * width)
        .zip(states["readout_input"].chunks_exact(5 * width))
    {
        let mut phases = vec![vec![0.0_f64; pairs]; cfg.head.context_stages()];
        for (x, h) in xs.chunks_exact(width).zip(hs.chunks_exact(width)) {
            let x: Vec<_> = x.iter().copied().map(f64::from).collect();
            let mut y: Vec<_> = h.iter().copied().map(f64::from).collect();
            for (level, (w, a)) in weights[projection..end]
                .chunks_exact(stage_size)
                .zip(&mut phases)
                .enumerate()
            {
                let step = memory_checks::linear(w, &x);
                for k in 0..pairs {
                    let (i, j) = coordinates(width, k, level == 1);
                    let (s, c) = a[k].sin_cos();
                    let (u, v) = (y[i], y[j]);
                    y[i] = c * u - s * v;
                    y[j] = s * u + c * v;
                    a[k] += step[k];
                }
            }
            expected.extend(memory_checks::linear(&weights[..cfg.vocab * width], &y));
        }
    }
    let logits = phase_checks::error(&actual, &expected, 1e-5)?;
    let prefix = memory_checks::prefix_error(&mut e, &toks[..5])?;
    if prefix > 2e-5 {
        return Err("context rotation changed an earlier logit".into());
    }
    let vectors: Vec<_> = (0..10 * width)
        .map(|_| rng.gauss(0., 0.29) as f32)
        .collect();
    let mut base = phase_controls::stream_control(&cfg, &weights)?;
    let plain = base.infer_embeddings(&vectors, 2, 5)?;
    let rotated = e.infer_embeddings(&vectors, 2, 5)?;
    let norm = |row: &[f32]| {
        row.iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let mut norm_error = 0.0_f64;
    let mut displacement = 0.0_f64;
    for (i, (x, y)) in plain
        .hidden
        .chunks_exact(width)
        .zip(rotated.hidden.chunks_exact(width))
        .enumerate()
    {
        if i % 5 == 0 && x != y {
            return Err("exclusive rotation changed a first-position off-table feature".into());
        }
        norm_error = norm_error.max((norm(x) - norm(y)).abs());
        displacement = x.iter().zip(y).fold(displacement, |m, (&a, &b)| {
            m.max((f64::from(a) - f64::from(b)).abs())
        });
    }
    if norm_error > 2e-5 || displacement < 0.01 {
        return Err("off-table norm preservation requires a nontrivial isometry".into());
    }
    let router = layout::router_offset(&cfg, 0);
    let mut directions = vec![
        ("tied embedding", (0..cfg.vocab * width).collect()),
        ("backbone router", (router..router + 96 * width).collect()),
        ("stream read", (first..projection).collect()),
        (
            "context rotation projection",
            (projection..after_first).collect(),
        ),
    ];
    if after_first < end {
        directions.push(("adjacent rotation projection", (after_first..end).collect()));
    }
    println!(
        "context rotation network width={width} max_absolute_logits={logits:.3e} prefix={prefix:.3e} norm_difference={norm_error:.3e} displacement={displacement:.3e}"
    );
    memory_checks::check_directions(&mut e, &weights, &toks, &labels, 5, directions)
}

// The Lean width-four witness is also a direct test of the physical pair layout.
fn transfers_between_planes() -> Result<()> {
    let cuda = Cuda::new()?;
    let kernel = context_rotation::kernels::Kernels::load(&cuda)?;
    let x = cuda.upload(&[0_f32, 0., 0., 0., 1., 0., 0., 0.])?;
    let increments = [std::f32::consts::FRAC_PI_2, 0., 0., 0.];
    let half_angles = cuda.upload(&increments)?;
    let adjacent_angles = cuda.upload(&increments)?;
    let half = cuda.alloc::<f32>(8)?;
    let adjacent = cuda.alloc::<f32>(8)?;
    unsafe {
        kernel.forward::<false>(
            Grid::new(1, 2),
            half.ptr(),
            half_angles.ptr(),
            x.ptr(),
            4,
            2,
        )?;
        kernel.forward::<true>(
            Grid::new(1, 2),
            adjacent.ptr(),
            adjacent_angles.ptr(),
            x.ptr(),
            4,
            2,
        )?;
    }
    let left = phase_checks::error(&half.read()?, &[0., 0., 0., 0., 0., 0., 1., 0.], 1e-6)?;
    let right = phase_checks::error(&adjacent.read()?, &[0., 0., 0., 0., 0., 1., 0., 0.], 1e-6)?;
    println!(
        "crossed rotation unit-vector transfer half_error={left:.3e} adjacent_error={right:.3e}"
    );
    Ok(())
}

pub fn run() -> Result<()> {
    if !precision::FP32_AFFINE_REFERENCE {
        return Err("context rotation checks require fp32-affine-reference".into());
    }
    for (w, l) in [(64, 1), (128, 7), (256, 257)] {
        scan::<false>(w, l)?;
        if architecture::CROSSED_CONTEXT {
            scan::<true>(w, l)?;
        }
    }
    for w in [64, 128] {
        network(w)?;
    }
    if architecture::CROSSED_CONTEXT {
        transfers_between_planes()?;
    }
    Ok(())
}
