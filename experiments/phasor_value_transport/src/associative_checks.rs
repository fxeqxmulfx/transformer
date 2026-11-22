//! Independent FP64 recurrence, finite differences and embedding-only model checks.
use super::*;
use associative::{KEYS as K, PROJECTED as P, VALUES as V};
use kernite_cuda::{Cuda, Grid};

fn bound(x: &[f64]) -> [f64; K] {
    let denominator = (1. + x.iter().map(|v| v * v).sum::<f64>()).sqrt();
    std::array::from_fn(|i| x[i] / denominator)
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

fn reference(projected: &[f64], length: usize) -> (Vec<f64>, Vec<f64>) {
    let mut reads = Vec::new();
    let mut states = Vec::new();
    for sequence in projected.chunks_exact(length * P) {
        let mut matrix = [[0.; K]; V];
        let mut previous: Option<[f64; K]> = None;
        for row in sequence.as_chunks::<P>().0 {
            let query = bound(&row[K..2 * K]);
            for (i, state) in matrix.iter_mut().enumerate() {
                if let Some(key) = &previous {
                    let residual = row[2 * K + i] - dot(state, key);
                    for (entry, k) in state.iter_mut().zip(key) {
                        *entry += residual * k;
                    }
                }
                reads.push(dot(state, &query));
                states.extend_from_slice(state);
            }
            previous = Some(bound(&row[..K]));
        }
    }
    (reads, states)
}

fn scan(length: usize) -> Result<()> {
    let batch = 3;
    let rows = batch * length;
    let cuda = Cuda::new()?;
    let kernels = associative::kernels::Kernels::load(&cuda)?;
    let mut rng = random::Random::new(797);
    let mut sample = |n: usize| {
        (0..n)
            .map(|_| rng.gauss(0., 0.61) as f32)
            .collect::<Vec<_>>()
    };
    let mut projected = sample(rows * P);
    for row in projected.as_chunks_mut::<P>().0.iter_mut().step_by(13) {
        row[..K].fill(0.);
    }
    let upstream = sample(rows * V);
    let packed = cuda.upload(&projected)?;
    let g = cuda.upload(&upstream)?;
    let inverse = cuda.alloc::<f32>(rows * 2)?;
    let states = cuda.alloc::<f32>(rows * K * V)?;
    let read = cuda.alloc::<f32>(rows * V)?;
    let grad = cuda.alloc::<f32>(rows * P)?;
    let per_row = Grid::new(rows as u32, 32);
    let per_seq = Grid::new(batch as u32, 32);
    unsafe {
        kernels.bound::<false>(per_row, packed.ptr(), grad.ptr(), inverse.ptr())?;
        kernels.forward(
            per_seq,
            read.ptr(),
            states.ptr(),
            packed.ptr(),
            length as u32,
        )?;
        kernels.backward(
            per_seq,
            grad.ptr(),
            g.ptr(),
            states.ptr(),
            packed.ptr(),
            length as u32,
        )?;
        kernels.bound::<true>(per_row, packed.ptr(), grad.ptr(), inverse.ptr())?;
    }
    let raw: Vec<_> = projected.into_iter().map(f64::from).collect();
    let cotangent: Vec<_> = upstream.into_iter().map(f64::from).collect();
    let (expected, memory) = reference(&raw, length);
    let forward = phase_checks::error(&read.read()?, &expected, 2e-5)?;
    let state = phase_checks::error(&states.read()?, &memory, 2e-5)?;
    let grad: Vec<_> = grad.read()?.into_iter().map(f64::from).collect();
    let objective = |x: &[f64]| dot(&reference(x, length).0, &cotangent);
    let mut worst = 0.0_f64;
    let mut check = |a: f64, b: f64| -> Result<()> {
        let err = (a - b).abs();
        if !err.is_finite() || err > 3e-5 * (1. + b.abs()) {
            return Err(
                format!("associative raw projection derivative {a} differs from {b}").into(),
            );
        }
        worst = worst.max(err);
        Ok(())
    };
    let step = 1e-5;
    if length <= 7 {
        let mut x = raw.clone();
        for (i, &analytic) in grad.iter().enumerate() {
            x[i] = raw[i] + step;
            let plus = objective(&x);
            x[i] = raw[i] - step;
            let minus = objective(&x);
            x[i] = raw[i];
            check(analytic, (plus - minus) / (2. * step))?;
        }
    }
    for _ in 0..7 {
        let direction: Vec<_> = (0..raw.len()).map(|_| rng.gauss(0., 0.4)).collect();
        let value = |h: f64| {
            objective(
                &raw.iter()
                    .zip(&direction)
                    .map(|(x, d)| x + h * d)
                    .collect::<Vec<_>>(),
            )
        };
        check(
            dot(&grad, &direction),
            (value(step) - value(-step)) / (2. * step),
        )?;
    }
    println!(
        "associative scan length={length} reads={} states={} max_read={forward:.3e} max_state={state:.3e} max_gradient={worst:.3e}",
        expected.len(),
        memory.len()
    );
    Ok(())
}

fn network(width: usize) -> Result<()> {
    let len = 6;
    let batch = 2;
    let mut cfg = config(batch)?;
    cfg.width = width;
    cfg.heads = width / 32;
    cfg.gain = cfg.heads as f32 / width as f32;
    cfg.vocab = 29;
    cfg.max_length = len;
    cfg.head = head::Head::AssociativeReflection;
    let mut plain_cfg = cfg.clone();
    plain_cfg.head = head::Head::ReflectionPhase;
    let start = layout::frequency_offset(&plain_cfg);
    let output = start + P * width;
    let end = layout::frequency_offset(&cfg);
    let mut theta = layout::initialize_model(&cfg, 17);
    let base_theta = layout::initialize_model(&plain_cfg, 17);
    if theta[..start] != base_theta[..start]
        || theta[end..] != base_theta[start..]
        || theta[output..end].iter().any(|&x| x != 0.)
    {
        return Err("associative initialization failed to retain mixed reflection weights".into());
    }
    let mut rng = random::Random::new(113);
    for w in &mut theta[architecture::memory_start(&cfg) + head::MEMORY_WIDTH * width..start] {
        *w = rng.gauss(0., 0.08) as f32;
    }
    let mixed = [&theta[..start], &theta[end..]].concat();
    let mut base = Engine::new(plain_cfg, &mixed)?;
    let mut e = Engine::new(cfg.clone(), &theta)?;
    let rows = batch * len;
    let x: Vec<_> = (0..rows * width)
        .map(|_| rng.gauss(0., 0.27) as f32)
        .collect();
    let g: Vec<_> = (0..rows * cfg.vocab)
        .map(|_| rng.gauss(0., 0.25) as f32)
        .collect();
    let plain = base.infer_embeddings(&x, batch, len)?;
    let init = e.infer_embeddings(&x, batch, len)?;
    if init.hidden != plain.hidden
        || init.logits != plain.logits
        || e.embedding_pullback(&x, batch, len, &g)?
            != base.embedding_pullback(&x, batch, len, &g)?
    {
        return Err(
            "zero associative output changed a mixed feature, score or input gradient".into(),
        );
    }
    let tokens = [1, 9, 4, 9, 5, 11, 3, 11, 2, 12, 5, 3];
    let labels = [u32::MAX, 5, 13, u32::MAX, 7, 2, u32::MAX, 1, 4, 6, 3, 5];
    e.forward(&tokens, &labels, batch, len)?;
    e.backward(batch, len)?;
    let gradient = e.gradients.read()?;
    if gradient[start..output].iter().any(|&g| g != 0.)
        || !gradient[output..end].iter().any(|g| g.abs() > 1e-6)
    {
        return Err("incorrect gradient through zero associative output".into());
    }
    for w in &mut theta[output..end] {
        *w = rng.gauss(0., 0.23) as f32;
    }
    e.parameters.write(&theta)?;
    let actual = e.infer_embeddings(&x, batch, len)?;
    let snapshot = e.diagnostic_snapshot(rows)?;
    let mut projected = Vec::new();
    for row in snapshot["embedding_normalized"].chunks_exact(width) {
        projected.extend(memory_checks::linear(
            &theta[start..output],
            &row.iter().copied().map(f64::from).collect::<Vec<_>>(),
        ));
    }
    let (reads, _) = reference(&projected, len);
    let mut expected = Vec::new();
    let mut active = 0.0_f64;
    for (h, row) in plain
        .hidden
        .chunks_exact(width)
        .zip(reads.as_chunks::<V>().0)
    {
        let add = memory_checks::linear(&theta[output..end], row);
        for (&v, a) in h.iter().zip(add) {
            active = active.max(a.abs());
            expected.push(f64::from(v) + a);
        }
    }
    let forward = phase_checks::error(&actual.hidden, &expected, 1e-5)?;
    let mut changed = x.clone();
    for sequence in changed.chunks_exact_mut(len * width) {
        for v in &mut sequence[3 * width..] {
            *v = rng.gauss(0., 1.) as f32;
        }
    }
    let other = e.infer_embeddings(&changed, batch, len)?;
    for (a, b) in actual
        .hidden
        .chunks_exact(len * width)
        .zip(other.hidden.chunks_exact(len * width))
    {
        if a[..3 * width] != b[..3 * width] || active < 0.02 {
            return Err("active associative residual failed off-table prefix test".into());
        }
    }
    println!(
        "associative network width={width} identical_mixed_features={} identical_mixed_input_gradient={} max_hidden={forward:.3e} max_residual={active:.3e}",
        plain.hidden.len(),
        x.len()
    );
    let router = layout::router_offset(&cfg, 0);
    memory_checks::check_directions(
        &mut e,
        &theta,
        &tokens,
        &labels,
        len,
        vec![
            ("tied embedding", (0..width * cfg.vocab).collect()),
            ("contextual router", (router..router + 96 * width).collect()),
            (
                "mixed projection",
                (architecture::memory_start(&cfg)
                    ..architecture::memory_start(&cfg) + head::MEMORY_WIDTH * width)
                    .collect(),
            ),
            ("associative key", (start..start + K * width).collect()),
            (
                "associative query",
                (start + K * width..start + 2 * K * width).collect(),
            ),
            (
                "associative value",
                (start + 2 * K * width..output).collect(),
            ),
            ("associative output", (output..end).collect()),
        ],
    )
}

pub fn run() -> Result<()> {
    if !precision::FP32_AFFINE_REFERENCE {
        return Err("associative checks require fp32-affine-reference".into());
    }
    for length in [1, 7, 257] {
        scan(length)?;
    }
    for width in [64, 128] {
        network(width)?;
    }
    Ok(())
}
