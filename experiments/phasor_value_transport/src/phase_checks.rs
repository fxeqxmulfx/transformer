//! Independent FP64 prefix sums, reverse sums and masked classification
//! gradients for the learned phase memory from Transformer.Phasor.PhasePrefixRead.
use super::*;
use head::MEMORY_WIDTH as M;
use kernite_cuda::{Cuda, Grid};
use memory_checks::linear;

pub(super) fn error(native: &[f32], expected: &[f64], tolerance: f64) -> Result<f64> {
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

fn scan<const REFLECT: bool>(length: usize) -> Result<()> {
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
        kernel.forward::<REFLECT>(grid, read.ptr(), saved.ptr(), length as u32)?;
        kernel.backward::<REFLECT>(grid, grad.ptr(), saved.ptr(), g.ptr(), length as u32)?;
    }
    let mut features = vec![0.; upstream.len()];
    let mut derivative = vec![0.; increments.len()];
    // Each read contributes directly to all earlier inputs. This oracle
    // never carries an adjoint across rows and resets at batch boundaries.
    for row in 0..rows {
        let start = row - row % length;
        let sign = |i| {
            if REFLECT && (row - i) % 2 == 1 {
                -1.
            } else {
                1.
            }
        };
        for k in 0..M {
            let angle: f64 = (start..=row)
                .map(|i| sign(i) * f64::from(increments[i * M + k]))
                .sum();
            let (s, c) = angle.sin_cos();
            let offset = row * 2 * M + k;
            features[offset] = s;
            features[offset + M] = c - 1.;
            let g = f64::from(upstream[offset]) * c - f64::from(upstream[offset + M]) * s;
            for i in start..=row {
                derivative[i * M + k] += sign(i) * g;
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
    cfg.head = architecture::HEAD;
    let mut weights = layout::initialize_model(&cfg, 3);
    let first = architecture::start(&cfg);
    let increments = architecture::memory_start(&cfg);
    let affine = cfg.head == head::Head::AffinePhase;
    let bilinear = cfg.head == head::Head::BilinearPhase;
    let shared = cfg.head == head::Head::SharedBilinear;
    let channels = if shared { head::SHARED_LATENTS } else { M };
    let writes = increments + channels * width;
    let periodic = cfg.head == head::Head::PeriodicReflection;
    let reflecting = cfg.head == head::Head::ReflectionPhase || periodic;
    let single = cfg.head == head::Head::SingleReflection;
    let output = increments + cfg.head.phase_inputs() * width;
    let end = output + 2 * M * width;
    let mut e = Engine::new(cfg.clone(), &weights)?;
    e.forward(&tokens, &labels, 2, length)?;
    e.backward(2, length)?;
    let grad = e.gradients.read()?;
    let output_indices =
        |half: usize| (output..end).filter(move |&i| (i - output) % (2 * M) / M == half);
    if grad[increments..output].iter().any(|&g| g != 0.)
        || grad[end..end
            + cfg.head.phase_gates()
            + cfg.head.activation_parameters()
            + cfg.head.frequency_parameters()]
            .iter()
            .any(|&g| g != 0.)
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
    if shared {
        let inactive = writes + channels * width;
        if weights[inactive..output].iter().any(|&w| w != 0.) {
            return Err(
                "ordered right and local latent projections must initialize at zero".into(),
            );
        }
        for w in &mut weights[inactive..output] {
            *w = rng.gauss(0., 0.05) as f32;
        }
        shared_latent_permutation(&cfg, &weights, &mut e)?;
    }
    if affine || bilinear || reflecting {
        phase_controls::zero_extension(&cfg, &weights, &mut e)?;
        let inactive = if bilinear { writes + M * width } else { writes };
        for w in &mut weights[inactive..output] {
            *w = rng.gauss(0., 0.05) as f32;
        }
        if reflecting {
            for w in &mut weights[end..end + M] {
                *w = rng.gauss(0.5, 0.45) as f32;
            }
        }
        if periodic {
            for w in &mut weights[end + M..end + 2 * M] {
                *w = rng.gauss(1., 0.24) as f32;
            }
            for w in &mut weights[end + 2 * M..end + 3 * M] {
                *w = rng.gauss(0., 0.48) as f32;
            }
        }
    }
    e.parameters.write(&weights)?;
    if single {
        phase_controls::one_gate(&cfg, &weights, end, &mut e)?;
    }
    e.forward(&tokens, &labels, 2, length)?;
    let actual = e.logits.read()?;
    let hidden = e.diagnostic_snapshot(tokens.len())?;
    let mut expected = Vec::new();
    let source = if cfg.head.projects_embeddings() {
        "embedding_normalized"
    } else {
        "readout_input"
    };
    let contextual = hidden["readout_input"].chunks_exact(width * length);
    for (seq, source) in contextual.zip(hidden[source].chunks_exact(width * length)) {
        let mut angles = [0.; M];
        let mut ordered = [(1., 0.); M];
        let mut left_sum = [0.; M];
        let mut reflection = [0.; M];
        for (h, x) in seq.chunks_exact(width).zip(source.chunks_exact(width)) {
            let h: Vec<_> = h.iter().copied().map(f64::from).collect();
            let x: Vec<_> = x.iter().copied().map(f64::from).collect();
            let mut step = linear(&weights[increments..output], &x);
            if periodic {
                for (k, x) in step.iter_mut().enumerate() {
                    let strength = f64::from(weights[end + M + k]);
                    let center = f64::from(weights[end + 2 * M + k]);
                    *x -= strength * (2. * (*x - center)).sin() / 2.;
                }
            }
            let mut pair = [0.; 2 * M];
            if shared {
                let mut state = [0.0; head::SHARED_LATENTS];
                for (k, current) in state.iter_mut().enumerate() {
                    angles[k] += step[k] + left_sum[k] * step[2 * channels + k];
                    *current = angles[k] + step[3 * channels + k];
                    left_sum[k] += step[channels + k];
                }
                let mixed = linear(&weights[end..end + cfg.head.frequency_parameters()], &state);
                for (j, angle) in mixed.into_iter().enumerate() {
                    let (sine, cosine) = angle.sin_cos();
                    pair[j] = sine;
                    pair[M + j] = cosine - 1.;
                }
            } else {
                for k in 0..M {
                    if affine {
                        let (s, c) = step[k].sin_cos();
                        let (u, v) = ordered[k];
                        ordered[k] = (c * u - s * v + step[M + k], s * u + c * v);
                        pair[k] = ordered[k].1;
                        pair[M + k] = ordered[k].0 - 1.;
                    } else if single {
                        reflection[k] = step[k] - reflection[k];
                        let (s, c) = reflection[k].sin_cos();
                        pair[k] = s;
                        pair[M + k] = c - 1.;
                    } else {
                        angles[k] += step[k];
                        let mut angle = angles[k];
                        if bilinear {
                            angles[k] += left_sum[k] * step[2 * M + k];
                            angle = angles[k] + step[3 * M + k];
                            left_sum[k] += step[M + k];
                        }
                        let (s, c) = angle.sin_cos();
                        pair[k] = s;
                        pair[M + k] = c - 1.;
                        if reflecting {
                            reflection[k] = step[k] - reflection[k];
                            let (rs, rc) = reflection[k].sin_cos();
                            let gate = f64::from(weights[end + k]);
                            pair[k] += gate * (rs - s);
                            pair[M + k] += gate * (rc - c);
                        }
                    }
                }
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
    let mut groups = vec![
        ("tied embedding", (0..cfg.vocab * width).collect()),
        ("backbone router", (router..router + 96 * width).collect()),
        ("stream read", (first..increments).collect()),
        ("phase increment", (increments..writes).collect()),
        ("sine output", output_indices(0).collect()),
        ("cosine output", output_indices(1).collect()),
    ];
    if affine {
        groups.push(("affine phase write", (writes..output).collect()));
    } else if reflecting {
        groups.push(("rotation/reflection gate", (end..end + M).collect()));
        if periodic {
            groups.push(("periodic strength", (end + M..end + 2 * M).collect()));
            groups.push(("periodic center", (end + 2 * M..end + 3 * M).collect()));
        }
    } else if bilinear || shared {
        for (j, name) in ["ordered left", "ordered right", "local phase"]
            .into_iter()
            .enumerate()
        {
            let start = writes + j * channels * width;
            groups.push((name, (start..start + channels * width).collect()));
        }
        if shared {
            groups.push((
                "learned frequency matrix",
                (end..end + cfg.head.frequency_parameters()).collect(),
            ));
        }
    }
    println!("phase network width={width} max_absolute_logits={logits:.3e} prefix={prefix:.3e}");
    memory_checks::check_directions(&mut e, &weights, &tokens, &labels, length, groups)?;
    if cfg.head.projects_embeddings() {
        phase_controls::input_segment(&cfg, &weights, &mut e)?;
    }
    if reflecting || single {
        reflection_phase_checks::insert_pair(&cfg, &weights, end, &mut e)?;
    }
    Ok(())
}

// Relabel all recurrent coordinates while preserving every periodic observation.
fn shared_latent_permutation(cfg: &Config, weights: &[f32], engine: &mut Engine) -> Result<()> {
    let r = head::SHARED_LATENTS;
    let d = cfg.width;
    let input = architecture::memory_start(cfg);
    let frequency = layout::frequency_offset(cfg) - cfg.head.frequency_parameters();
    let mut renamed = weights.to_vec();
    for group in 0..4 {
        renamed[input + group * r * d..input + (group + 1) * r * d].rotate_left(d);
    }
    for row in renamed[frequency..frequency + M * r].chunks_exact_mut(r) {
        row.rotate_left(1);
    }
    engine.parameters.write(weights)?;
    let mut other = Engine::new(cfg.clone(), &renamed)?;
    phase_controls::compare_dense(cfg, engine, &mut other, "cyclic latent reindexing")
}

pub fn run() -> Result<()> {
    if !precision::FP32_AFFINE_REFERENCE {
        return Err("phase memory checks require --features fp32-affine-reference".into());
    }
    for length in [1, 7, 257] {
        if architecture::PERIODIC_REFLECTION {
            crate::periodic_activation_checks::scan(length)?;
            reflection_phase_checks::scan(length)?;
        } else if architecture::REFLECTION_PHASE {
            reflection_phase_checks::scan(length)?;
        } else if architecture::BILINEAR_PHASE || architecture::SHARED_BILINEAR {
            let channels = if architecture::SHARED_BILINEAR {
                head::SHARED_LATENTS
            } else {
                M
            };
            bilinear_phase_checks::scan::<{ architecture::SHARED_BILINEAR }>(length, channels)?;
        } else if architecture::AFFINE_PHASE {
            affine_phase_checks::scan(length)?;
        } else {
            scan::<{ architecture::SINGLE_REFLECTION }>(length)?;
        }
    }
    for width in [64, 128] {
        network(width)?;
    }
    Ok(())
}
