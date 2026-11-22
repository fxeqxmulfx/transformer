//! Direct signed FP64 sums, shared gate derivatives and duplicate insertions
//! for Transformer.Phasor.ReflectionPhaseRead.
use super::*;
use head::MEMORY_WIDTH as M;
use kernite_cuda::{Cuda, Grid};

pub(super) fn scan(length: usize) -> Result<()> {
    let batch = 3;
    let size = batch * length * M;
    let cuda = Cuda::new()?;
    let kernels = reflection_phase::kernels::Kernels::load(&cuda)?;
    let mut rng = random::Random::new(903);
    let mut samples = |n| {
        (0..n)
            .map(|_| rng.gauss(0., 0.31) as f32)
            .collect::<Vec<_>>()
    };
    let inputs = samples(size);
    let upstream = samples(2 * size);
    let gates: Vec<_> = samples(M).into_iter().map(|v| v + 0.5).collect();
    let rotations = cuda.upload(&inputs)?;
    let reflections = cuda.alloc::<f32>(size)?;
    let mixing = cuda.upload(&gates)?;
    let cotangent = cuda.upload(&upstream)?;
    let partial = cuda.alloc::<f32>(batch * M)?;
    let gate_grad = cuda.alloc::<f32>(M)?;
    let read = cuda.alloc::<f32>(2 * size)?;
    let grad = cuda.alloc::<f32>(size)?;
    let grid = Grid::new(batch as u32, M as u32);
    // One independent scan per sequence/channel; reduce gates in batch order.
    unsafe {
        kernels.forward(
            grid,
            read.ptr(),
            rotations.ptr(),
            reflections.ptr(),
            mixing.ptr(),
            length as u32,
        )?;
        kernels.backward(
            grid,
            grad.ptr(),
            partial.ptr(),
            rotations.ptr(),
            reflections.ptr(),
            cotangent.ptr(),
            mixing.ptr(),
            length as u32,
        )?;
        kernels.gate_sum(
            Grid::new(1, M as u32),
            gate_grad.ptr(),
            partial.ptr(),
            batch as u32,
        )?;
    }
    let mut expected = vec![0.; 2 * size];
    let mut gradient = vec![0.; size];
    let mut gate_derivatives = vec![0.; M];
    for seq in 0..batch {
        for k in 0..M {
            let gate = f64::from(gates[k]);
            let index = |j| (seq * length + j) * M + k;
            for t in 0..length {
                let sign = |j| if (t - j) % 2 == 0 { 1. } else { -1. };
                let plus: f64 = (0..=t).map(|j| f64::from(inputs[index(j)])).sum();
                let minus: f64 = (0..=t).map(|j| sign(j) * f64::from(inputs[index(j)])).sum();
                let (s, c) = plus.sin_cos();
                let (sr, cr) = minus.sin_cos();
                let out = (seq * length + t) * 2 * M + k;
                expected[out] = (1. - gate) * s + gate * sr;
                expected[out + M] = (1. - gate) * (c - 1.) + gate * (cr - 1.);
                let (gs, gc) = (f64::from(upstream[out]), f64::from(upstream[out + M]));
                gate_derivatives[k] += gs * (sr - s) + gc * (cr - c);
                let rotate_grad = (1. - gate) * (gs * c - gc * s);
                let reflect_grad = gate * (gs * cr - gc * sr);
                for j in 0..=t {
                    gradient[index(j)] += rotate_grad + sign(j) * reflect_grad;
                }
            }
        }
    }
    let forward = phase_checks::error(&read.read()?, &expected, 3e-5)?;
    let backward = phase_checks::error(&grad.read()?, &gradient, 1.5e-4)?;
    let shared = phase_checks::error(&gate_grad.read()?, &gate_derivatives, 2e-4)?;
    println!(
        "reflection phase scan length={length} reads={} increments={size} gates={M} max_absolute_forward={forward:.3e} backward={backward:.3e} gate_gradient={shared:.3e}",
        2 * size
    );
    Ok(())
}

/// Inserting two copies of an off-table vector in the middle of a prefix
/// preserves the reflection-only endpoint and the full VJP on the original
/// inputs. The two gradients for the shared inserted vector sum to zero.
pub(super) fn insert_pair(
    cfg: &Config,
    weights: &[f32],
    gates: usize,
    e: &mut Engine,
) -> Result<()> {
    let d = cfg.width;
    let mut active = weights.to_vec();
    active[gates..gates + M].fill(1.);
    e.parameters.write(&active)?;
    let mut base = phase_controls::stream_control(cfg, weights)?;
    let mut rng = random::Random::new(191);
    let mut sample = |n| {
        (0..n)
            .map(|_| rng.gauss(0., 0.31) as f32)
            .collect::<Vec<_>>()
    };
    let input = sample(4 * d);
    let twin = sample(d);
    let cotangent = sample(cfg.vocab);
    let expanded = [&input[..2 * d], &twin, &twin, &input[2 * d..]].concat();
    let (old, old_grad) = phase_controls::endpoint(e, &mut base, &input, &cotangent)?;
    let (new, new_grad) = phase_controls::endpoint(e, &mut base, &expanded, &cotangent)?;
    let surviving = [&new_grad[..2 * d], &new_grad[4 * d..]].concat();
    let shared: Vec<_> = new_grad[2 * d..3 * d]
        .iter()
        .zip(&new_grad[3 * d..4 * d])
        .map(|(x, y)| x + y)
        .collect();
    if !old.iter().any(|v| v.abs() > 0.01)
        || !new_grad[2 * d..3 * d].iter().any(|g| g.abs() > 0.001)
    {
        return Err("inserted-pair check requires active reads and individual gradients".into());
    }
    let hidden = compact_checks::close("inserted reflection pair hidden", &old, &new, 1e-5)?;
    let gradient = compact_checks::close(
        "surviving reflection gradients",
        &old_grad,
        &surviving,
        4e-5,
    )?;
    let cancelled = compact_checks::close(
        "shared inserted vector gradient",
        &shared,
        &vec![0.; d],
        4e-5,
    )?;
    println!(
        "reflection pair insertion width={d} hidden={hidden:.3e} original_gradient={gradient:.3e} shared_gradient={cancelled:.3e}"
    );
    Ok(())
}
