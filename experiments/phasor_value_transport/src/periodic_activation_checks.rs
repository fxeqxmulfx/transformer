//! FP64 activation, elementwise derivative, shared derivative and local error
//! checks off the embedding table, including zero strength and negative lifts.
use super::*;
use head::MEMORY_WIDTH as M;
use kernite_cuda::{Cuda, Grid};

fn activation(x: f64, strength: f64, center: f64) -> f64 {
    x - strength * (2. * (x - center)).sin() / 2.
}

pub(super) fn scan(length: usize) -> Result<()> {
    let rows = 3 * length;
    let mut rng = random::Random::new(701);
    let mut xs: Vec<_> = (0..rows * M).map(|_| rng.gauss(0., 2.) as f32).collect();
    let upstream: Vec<_> = (0..xs.len()).map(|_| rng.gauss(0., 0.7) as f32).collect();
    let mut weights: Vec<_> = (0..2 * M).map(|_| rng.gauss(0.2, 0.75) as f32).collect();
    weights[0] = 1.;
    weights[1] = 0.;
    let mut local = Vec::new();
    for (i, row) in xs.as_chunks_mut::<M>().0.iter_mut().enumerate() {
        let center = f64::from(weights[M]) + (i as i32 % 9 - 4) as f64 * std::f64::consts::PI;
        row[0] = (center + rng.random() - 0.5) as f32;
        let error = (f64::from(row[0]) - center).abs();
        local.push((center, 2. * error.powi(3) / 3. + 4. * error.powi(5) / 25.));
    }
    let cuda = Cuda::new()?;
    let kernels = periodic_activation::kernels::Kernels::load(&cuda)?;
    let values = cuda.upload(&xs)?;
    let raw = cuda.alloc::<f32>(xs.len())?;
    let params = cuda.upload(&weights)?;
    let grad = cuda.upload(&upstream)?;
    let dw = cuda.alloc::<f32>(2 * M)?;
    unsafe {
        kernels.forward(
            Grid::new(rows as u32, M as u32),
            values.ptr(),
            raw.ptr(),
            params.ptr(),
        )?;
        kernels.backward(
            Grid::new(M as u32, 32),
            grad.ptr(),
            dw.ptr(),
            raw.ptr(),
            params.ptr(),
            rows as u32,
        )?;
    }
    let output = values.read()?;
    let input_grad = grad.read()?;
    let param_grad = dw.read()?;
    if raw.read()? != xs {
        return Err("periodic activation failed to retain the original projection".into());
    }
    let mut expected = Vec::with_capacity(xs.len());
    let mut dx = Vec::with_capacity(xs.len());
    let mut shared = [0.; 2 * M];
    for (i, (&x, &g)) in xs.iter().zip(&upstream).enumerate() {
        let k = i % M;
        let (u, b) = (f64::from(weights[k]), f64::from(weights[M + k]));
        let (x, g) = (f64::from(x), f64::from(g));
        let (s, c) = (2. * (x - b)).sin_cos();
        expected.push(activation(x, u, b));
        dx.push((1. - u * c) * g);
        shared[k] -= g * s / 2.;
        shared[M + k] += g * u * c;
    }
    for (row, (center, budget)) in local.iter().enumerate() {
        if (f64::from(output[row * M]) - center).abs() > budget + 2e-6
            || output[row * M + 1] != xs[row * M + 1]
            || input_grad[row * M + 1] != upstream[row * M + 1]
        {
            return Err(
                "periodic activation failed a local cubic bound or its zero-strength identity"
                    .into(),
            );
        }
    }
    let forward = phase_checks::error(&output, &expected, 3e-6)?;
    let backward = phase_checks::error(&input_grad, &dx, 4e-6)?;
    let weight_error = phase_checks::error(&param_grad, &shared, 1e-4)?;
    let xdir: Vec<_> = (0..xs.len()).map(|_| rng.gauss(0., 0.3)).collect();
    let wdir: Vec<_> = (0..2 * M).map(|_| rng.gauss(0., 0.3)).collect();
    let objective = |h: f64| {
        xs.iter()
            .enumerate()
            .map(|(i, &x)| {
                let k = i % M;
                f64::from(upstream[i])
                    * activation(
                        f64::from(x) + h * xdir[i],
                        f64::from(weights[k]) + h * wdir[k],
                        f64::from(weights[M + k]) + h * wdir[M + k],
                    )
            })
            .sum::<f64>()
    };
    let analytic = input_grad
        .iter()
        .zip(&xdir)
        .chain(param_grad.iter().zip(&wdir))
        .map(|(&a, &b)| f64::from(a) * b)
        .sum::<f64>();
    let numeric = (objective(1e-5) - objective(-1e-5)) / 2e-5;
    let direction = (analytic - numeric).abs();
    if !direction.is_finite() || direction > 1e-4 {
        return Err(
            format!("periodic simultaneous derivative {analytic} differs from {numeric}").into(),
        );
    }
    println!(
        "periodic activation length={length} rows={rows} coordinates={} max_forward={forward:.3e} input_gradient={backward:.3e} shared_gradient={weight_error:.3e} direction={direction:.3e} cubic_bounds={rows}",
        xs.len()
    );
    Ok(())
}
