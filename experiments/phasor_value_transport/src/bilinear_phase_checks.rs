//! FP64 pair sums and per-read derivatives for the bilinear phase scan.
use super::*;
use head::MEMORY_WIDTH as M;
use kernite_cuda::{Cuda, Grid};

pub(super) fn scan(length: usize) -> Result<()> {
    let batch = 3;
    let rows = batch * length;
    let cuda = Cuda::new()?;
    let kernels = bilinear_phase::kernels::Kernels::load(&cuda)?;
    let mut rng = random::Random::new(903);
    let mut samples = |n| {
        (0..n)
            .map(|_| rng.gauss(0., 0.27) as f32)
            .collect::<Vec<_>>()
    };
    let inputs = samples(rows * 4 * M);
    let upstream = samples(rows * 2 * M);
    let projected = cuda.upload(&inputs)?;
    let cotangent = cuda.upload(&upstream)?;
    let read = cuda.alloc::<f32>(upstream.len())?;
    let grad = cuda.alloc::<f32>(inputs.len())?;
    let grid = Grid::new(batch as u32, M as u32);
    // A warp reads and writes disjoint channels from its own sequence.
    unsafe {
        kernels.forward(grid, read.ptr(), projected.ptr(), length as u32)?;
        kernels.backward(
            grid,
            grad.ptr(),
            projected.ptr(),
            cotangent.ptr(),
            length as u32,
        )?;
    }
    let mut expected = vec![0.; upstream.len()];
    let mut gradient = vec![0.; inputs.len()];
    for seq in 0..batch {
        for channel in 0..M {
            let index = |t: usize, group: usize| ((seq * length + t) * 4 + group) * M + channel;
            let input = |t, group| f64::from(inputs[index(t, group)]);
            // Recompute every earlier left sum directly from the original
            // projections. Each read differentiates its own complete prefix.
            let earlier: Vec<f64> = (0..length)
                .map(|t| (0..t).map(|i| input(i, 1)).sum())
                .collect();
            for t in 0..length {
                let angle = input(t, 3)
                    + (0..=t)
                        .map(|i| input(i, 0) + earlier[i] * input(i, 2))
                        .sum::<f64>();
                let (sin, cos) = angle.sin_cos();
                let out = (seq * length + t) * 2 * M + channel;
                expected[out] = sin;
                expected[out + M] = cos - 1.;
                let g = f64::from(upstream[out]) * cos - f64::from(upstream[out + M]) * sin;
                gradient[index(t, 3)] += g;
                let mut later_right = 0.;
                for i in (0..=t).rev() {
                    gradient[index(i, 0)] += g;
                    gradient[index(i, 1)] += g * later_right;
                    gradient[index(i, 2)] += g * earlier[i];
                    later_right += input(i, 2);
                }
            }
        }
    }
    let forward = phase_checks::error(&read.read()?, &expected, 2e-4)?;
    let backward = phase_checks::error(&grad.read()?, &gradient, 2e-3)?;
    println!(
        "bilinear phase scan length={length} reads={} gradients={} max_absolute_forward={forward:.3e} backward={backward:.3e}",
        expected.len(),
        gradient.len()
    );
    Ok(())
}
