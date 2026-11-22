//! Independent direct complex recurrence and arbitrary-vector controls for
//! the GPU's moving-frame implementation of the learned affine phase read.
use super::*;
use head::MEMORY_WIDTH as M;
use kernite_cuda::{Cuda, Grid};

pub(super) fn scan(length: usize) -> Result<()> {
    let batch = 3;
    let size = batch * length * 2 * M;
    let cuda = Cuda::new()?;
    let kernels = affine_phase::kernels::Kernels::load(&cuda)?;
    let mut rng = random::Random::new(903);
    let mut samples = || {
        (0..size)
            .map(|_| rng.gauss(0., 0.31) as f32)
            .collect::<Vec<_>>()
    };
    let inputs = samples();
    let upstream = samples();
    let projected = cuda.upload(&inputs)?;
    let cotangent = cuda.upload(&upstream)?;
    let read = cuda.alloc::<f32>(size)?;
    let grad = cuda.alloc::<f32>(size)?;
    let grid = Grid::new(batch as u32, M as u32);
    // One warp owns one complete sequence with disjoint input and output buffers.
    unsafe {
        kernels.forward(grid, read.ptr(), projected.ptr(), length as u32)?;
        kernels.backward(
            grid,
            grad.ptr(),
            projected.ptr(),
            read.ptr(),
            cotangent.ptr(),
            length as u32,
        )?;
    }
    let mut expected = vec![0.; size];
    let mut gradient = vec![0.; size];
    for seq in 0..batch {
        for k in 0..M {
            let mut rotated = Vec::with_capacity(length);
            let (mut u, mut v) = (1., 0.);
            for t in 0..length {
                let i = (seq * length + t) * 2 * M + k;
                let (s, c) = f64::from(inputs[i]).sin_cos();
                let re = c * u - s * v;
                let im = s * u + c * v;
                rotated.push((re, im, s, c));
                (u, v) = (re + f64::from(inputs[i + M]), im);
                expected[i] = v;
                expected[i + M] = u - 1.;
                // Send each output independently back through the earlier
                // one-step rotations. No angle prefix or suffix sum is used.
                let (mut gr, mut gi) = (f64::from(upstream[i + M]), f64::from(upstream[i]));
                for (j, &(r, h, sin, cos)) in rotated.iter().enumerate().rev() {
                    let at = (seq * length + j) * 2 * M + k;
                    gradient[at] += gi * r - gr * h;
                    gradient[at + M] += gr;
                    (gr, gi) = (cos * gr + sin * gi, -sin * gr + cos * gi);
                }
            }
        }
    }
    let forward = phase_checks::error(&read.read()?, &expected, 2e-4)?;
    let backward = phase_checks::error(&grad.read()?, &gradient, 1.5e-3)?;
    println!(
        "affine phase scan length={length} reads={size} gradients={size} max_absolute_forward={forward:.3e} backward={backward:.3e}"
    );
    Ok(())
}

/// The two models share every learned matrix except the zero write map.
/// Preserve the trained output projections and all trailing rotary weights.
pub(super) fn zero_write(cfg: &Config, weights: &[f32], ordered: &mut Engine) -> Result<()> {
    let size = M * cfg.width;
    let writes = architecture::memory_start(cfg) + size;
    if weights[writes..writes + size].iter().any(|&x| x != 0.) {
        return Err("initial affine write matrix must be zero".into());
    }
    let mut base_cfg = cfg.clone();
    base_cfg.head = head::Head::InputPhase;
    let base_weights = [&weights[..writes], &weights[writes + size..]].concat();
    let mut base = Engine::new(base_cfg, &base_weights)?;
    ordered.parameters.write(weights)?;
    let rows = cfg.max_length * 2;
    let mut rng = random::Random::new(101);
    let mut samples = |n| {
        (0..n)
            .map(|_| rng.gauss(0., 0.2) as f32)
            .collect::<Vec<_>>()
    };
    let inputs = samples(rows * cfg.width);
    let g = samples(rows * cfg.vocab);
    let a = ordered.infer_embeddings(&inputs, 2, cfg.max_length)?;
    let b = base.infer_embeddings(&inputs, 2, cfg.max_length)?;
    let da = ordered.embedding_pullback(&inputs, 2, cfg.max_length, &g)?;
    let db = base.embedding_pullback(&inputs, 2, cfg.max_length, &g)?;
    let hidden = compact_checks::close("zero affine write hidden", &a.hidden, &b.hidden, 3e-6)?;
    let logits = compact_checks::close("zero affine write logits", &a.logits, &b.logits, 3e-6)?;
    let gradient = compact_checks::close("zero affine write input VJP", &da, &db, 1e-5)?;
    println!(
        "zero affine writes width={} hidden={hidden:.3e} logits={logits:.3e} input_gradient={gradient:.3e}",
        cfg.width
    );
    Ok(())
}
