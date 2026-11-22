//! Independent ideal-coordinate fixture. Only the reference knows residue arithmetic.
use super::*;
use fourier_geometry::{Measurements, character, conjugate, multiply};
use kernite_cuda::{Cuda, Grid};
use std::f64::consts::TAU;

pub fn run() -> Result<()> {
    fourier_gradients::run()?;
    for seed in 0..3 {
        let mut cfg = config(1)?;
        let original = layout::initialize_model(&cfg, seed);
        cfg.head = head::Head::FourierTransport;
        let fourier = layout::initialize_model(&cfg, seed);
        let lookup = cfg.vocab * cfg.width;
        if original[..lookup] != fourier[..lookup] {
            return Err("Fourier initialization changed the original random lookup".into());
        }
    }
    let (p, m) = (97usize, 96usize);
    let mut power = vec![1; m];
    for j in 1..m {
        power[j] = 5 * power[j - 1] % p;
    }
    let frequencies: Vec<_> = (0..m).map(|j| power[(13 * j + 17) % m]).collect();
    let mut inverse = vec![usize::MAX; p];
    for (j, &k) in frequencies.iter().enumerate() {
        if k == 0 || inverse[k] != usize::MAX {
            return Err("reference chart is not a bijection".into());
        }
        inverse[k] = j;
    }
    let origins: Vec<_> = (0..m)
        .map(|j| {
            let (s, c) = (TAU * ((17 * j + 2) % m) as f64 / m as f64).sin_cos();
            [c, s]
        })
        .collect();
    let coordinates: Vec<_> = (0..p)
        .flat_map(|b| {
            frequencies
                .iter()
                .enumerate()
                .map(|(j, &k)| multiply(origins[j], character(k, b, p)))
                .collect::<Vec<_>>()
        })
        .collect();
    let cuda = Cuda::new()?;
    let kernels = fourier::kernels::Kernels::load(&cuda)?;
    let mut data = Measurements {
        p,
        m,
        coordinates,
        phases: vec![[0.; 2]; p * m],
        routes: vec![0.; p * (m + 1)],
    };
    let mut inputs = vec![0f32; p * 2 * m];
    for b in 0..p {
        for j in 0..m {
            for t in 0..2 {
                inputs[b * 2 * m + t * m + j] = data.coordinates[b * m + j][t] as f32;
            }
        }
    }
    let input = cuda.upload(&inputs)?;
    let out = cuda.alloc::<f32>(inputs.len())?;
    let mut max_error = 0f64;
    for a in 0..p {
        let shift = if a == 0 {
            m
        } else {
            inverse[a * frequencies[0] % p]
        };
        data.routes[a * (m + 1) + shift] = 1.;
        let intercept = a * a % p * a % p;
        let mut phases = vec![0f32; 2 * m];
        for (j, &k) in frequencies.iter().enumerate() {
            let reference = k * a % p;
            if a != 0 && frequencies[(j + shift) % m] != reference {
                return Err(
                    "nonzero multiplication is not the scheduled cyclic permutation".into(),
                );
            }
            let src = if a == 0 {
                [1., 0.]
            } else {
                origins[inverse[reference]]
            };
            let ph = multiply(
                multiply(origins[j], conjugate(src)),
                character(k, intercept, p),
            );
            data.phases[a * m + j] = ph;
            phases[j] = ph[0] as f32;
            phases[m + j] = ph[1] as f32;
        }
        let mut r = vec![0f32; m + 1];
        r[shift] = 1.;
        let route = cuda.upload(&r.repeat(p))?;
        let phase = cuda.upload(&phases.repeat(p))?;
        unsafe {
            kernels.mix::<false>(
                Grid::new(p as u32, m as u32),
                out.ptr(),
                input.ptr(),
                route.ptr(),
                phase.ptr(),
            )?;
        }
        let actual = out.read()?;
        for b in 0..p {
            for (j, &k) in frequencies.iter().enumerate() {
                let expected = multiply(origins[j], character(k, (intercept + a * b) % p, p));
                for t in 0..2 {
                    let err = (f64::from(actual[b * 2 * m + t * m + j]) - expected[t]).abs();
                    if !err.is_finite() || err > 3e-6 {
                        return Err(
                            format!("Fourier vertex error a={a} b={b} j={j} error={err}").into(),
                        );
                    }
                    max_error = max_error.max(err);
                }
            }
        }
    }
    let aligned = fourier_geometry::analyze(&data)?;
    for val in [
        &aligned["channel_mean_squared_correlation"],
        &aligned["routing"]["nonzero_mean_correct_mass"],
        &aligned["phase"]["zero_mean_cosine"],
        &aligned["phase"]["nonzero_mean_cosine"],
        &aligned["transported_character_cosine"],
    ] {
        if (val.as_f64().ok_or("missing reference measurement")? - 1.).abs() > 1e-9 {
            return Err(format!(
                "ideal relabelled Fourier coordinates failed alignment: {aligned}"
            )
            .into());
        }
    }
    if aligned["frequency_by_learned_slot"] != json!(frequencies)
        || aligned["routing"]["nonzero_top1_correct"] != m * m
    {
        return Err("reference frequency assignment changed a channel".into());
    }
    println!(
        "Fourier vertices nonzero_permutations={m} constant_maps=1 complex_outputs={} maximum_error={max_error:.3e} matched_channels={m}",
        p * p * m
    );
    Ok(())
}
