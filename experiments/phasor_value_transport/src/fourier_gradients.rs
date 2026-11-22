//! FP64 reference for non-vertex routes, their softmax VJP and the phase VJP.
use super::*;
use kernite_cuda::{Cuda, Grid};

fn compare(name: &str, actual: &[f32], expected: &[f64]) -> Result<f64> {
    if actual.len() != expected.len() {
        return Err(format!("{name} shape mismatch").into());
    }
    let mut worst = 0f64;
    for (&a, &b) in actual.iter().zip(expected) {
        let error = (f64::from(a) - b).abs();
        if !error.is_finite() || error > 7e-6 * (1. + b.abs()) {
            return Err(format!("Fourier {name}: device {a}, scalar {b}").into());
        }
        worst = worst.max(error);
    }
    Ok(worst)
}

pub(super) fn run() -> Result<()> {
    let (rows, length, m) = (6usize, 3usize, 5usize);
    let n = 2 * m + 1;
    let mut rng = random::Random::new(437);
    let mut sample = |n| {
        (0..n)
            .map(|_| rng.gauss(0., 0.54) as f32)
            .collect::<Vec<_>>()
    };
    let x = sample(rows * 2 * m);
    let ctrl = sample(rows * n);
    let bias = sample(n);
    let grad = sample(rows * 2 * m);
    let cuda = Cuda::new()?;
    let kernels = fourier::kernels::Kernels::load(&cuda)?;
    let xd = cuda.upload(&x)?;
    let controls = cuda.upload(&ctrl)?;
    let biases = cuda.upload(&bias)?;
    let g = cuda.upload(&grad)?;
    let routes = cuda.alloc(rows * (m + 1))?;
    let phases = cuda.alloc(rows * 2 * m)?;
    let y = cuda.alloc(rows * 2 * m)?;
    let dm = cuda.alloc(rows * 2 * m)?;
    let da = cuda.alloc(rows * m)?;
    let dx = cuda.alloc(rows * 2 * m)?;
    let dp = cuda.alloc(rows * (m + 1))?;
    let dc = cuda.alloc(rows * n)?;
    let launch = Grid::new(rows as u32, m as u32);
    unsafe {
        kernels.prepare(
            Grid::new(rows as u32, 32),
            routes.ptr(),
            phases.ptr(),
            controls.ptr(),
            biases.ptr(),
            m as u32,
            length as u32,
        )?;
        kernels.mix::<false>(launch, y.ptr(), xd.ptr(), routes.ptr(), phases.ptr())?;
        kernels.unrotate(launch, dm.ptr(), da.ptr(), g.ptr(), y.ptr(), phases.ptr())?;
        kernels.mix::<true>(launch, dx.ptr(), dm.ptr(), routes.ptr(), phases.ptr())?;
        kernels.route_back(
            Grid::new(rows as u32, (m + 1) as u32),
            dp.ptr(),
            dm.ptr(),
            xd.ptr(),
            m as u32,
        )?;
        kernels.controls_back(
            Grid::new(rows as u32, 32),
            dc.ptr(),
            dp.ptr(),
            routes.ptr(),
            da.ptr(),
            m as u32,
            length as u32,
        )?;
    }
    let mut weights = vec![0.; rows * (m + 1)];
    let mut rotations = vec![0.; rows * 2 * m];
    let mut result = vec![0.; rows * 2 * m];
    let mut dx_ref = vec![0.; rows * 2 * m];
    let mut dc_ref = vec![0.; rows * n];
    for r in 0..rows {
        let first = r % length == 0;
        let w = &mut weights[r * (m + 1)..(r + 1) * (m + 1)];
        if first {
            w[0] = 1.;
        } else {
            for s in 0..=m {
                w[s] = (f64::from(ctrl[(r - 1) * n + s]) + f64::from(bias[s])).exp();
            }
            let norm: f64 = w.iter().sum();
            for z in &mut *w {
                *z /= norm;
            }
        }
        let mut dp_ref = vec![0.; m + 1];
        for j in 0..m {
            let angle = if first {
                0.
            } else {
                f64::from(ctrl[(r - 1) * n + m + 1 + j]) + f64::from(bias[m + 1 + j])
            };
            let (s, c) = angle.sin_cos();
            let lo = r * 2 * m + j;
            let hi = lo + m;
            rotations[lo] = c;
            rotations[hi] = s;
            let (mut re, mut im) = (w[m], 0.);
            let (gr, gi) = (f64::from(grad[lo]), f64::from(grad[hi]));
            let (ur, ui) = (c * gr + s * gi, c * gi - s * gr);
            for shift in 0..m {
                let at = r * 2 * m + (j + shift) % m;
                let (xr, xi) = (f64::from(x[at]), f64::from(x[at + m]));
                re += w[shift] * xr;
                im += w[shift] * xi;
                dx_ref[at] += w[shift] * ur;
                dx_ref[at + m] += w[shift] * ui;
                dp_ref[shift] += ur * xr + ui * xi;
            }
            dp_ref[m] += ur;
            result[lo] = c * re - s * im;
            result[hi] = s * re + c * im;
            if !first {
                dc_ref[(r - 1) * n + m + 1 + j] = gi * result[lo] - gr * result[hi];
            }
        }
        if !first {
            let mean: f64 = dp_ref.iter().zip(&*w).map(|(a, b)| a * b).sum();
            for s in 0..=m {
                dc_ref[(r - 1) * n + s] = w[s] * (dp_ref[s] - mean);
            }
        }
    }
    let mut worst = 0f64;
    for (name, measured, expected) in [
        ("softmax", routes.read()?, weights),
        ("phase", phases.read()?, rotations),
        ("transport", y.read()?, result),
        ("value adjoint", dx.read()?, dx_ref),
        ("control adjoint", dc.read()?, dc_ref),
    ] {
        worst = worst.max(compare(name, &measured, &expected)?);
    }
    println!(
        "Fourier fractional routes and adjoints rows={rows} length={length} channels={m} max_error={worst:.3e}"
    );
    Ok(())
}
