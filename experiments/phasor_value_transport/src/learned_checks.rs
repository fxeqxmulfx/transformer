//! Independent FP64 and vocabulary-permutation checks for the embedding-input models.
use super::*;
use kernite_cuda::{Cuda, Grid};

fn near(a: f32, reference: f64, error: &mut f64) -> Result<()> {
    let difference = (f64::from(a) - reference).abs();
    if !difference.is_finite() || difference > 3e-5 * (1. + reference.abs()) {
        return Err(format!("stream kernel produced {a}, expected {reference}").into());
    }
    *error = (*error).max(difference);
    Ok(())
}

fn scalar_oracle(rows: usize, width: usize) -> Result<()> {
    let cuda = Cuda::new()?;
    let kernels = learned_read::kernels::Kernels::load(&cuda)?;
    let mut rng = random::Random::new(113);
    let mut values = |n| {
        (0..n)
            .map(|_| rng.gauss(0., 0.4) as f32)
            .collect::<Vec<_>>()
    };
    let states = values(rows * 4 * width);
    let offsets = values(4 * width);
    let grad = values(rows * width);
    let (x, w, g) = (
        cuda.upload(&states)?,
        cuda.upload(&offsets)?,
        cuda.upload(&grad)?,
    );
    let y = cuda.alloc::<f32>(rows * width)?;
    let dx = cuda.alloc::<f32>(states.len())?;
    let dw = cuda.alloc::<f32>(offsets.len())?;
    let grid = Grid::new((rows * width).div_ceil(256) as u32, 256);
    // All buffers have precisely the shapes being tested.
    unsafe {
        kernels.mix::<false>(
            grid,
            y.ptr(),
            x.ptr(),
            w.ptr(),
            width as u32,
            (rows * width) as u32,
        )?;
        kernels.mix::<true>(
            grid,
            dx.ptr(),
            g.ptr(),
            w.ptr(),
            width as u32,
            (rows * width) as u32,
        )?;
        kernels.weights_back(
            Grid::new((4 * width) as u32, 32),
            dw.ptr(),
            x.ptr(),
            g.ptr(),
            width as u32,
            rows as u32,
        )?;
    }
    let (y, dx, dw) = (y.read()?, dx.read()?, dw.read()?);
    let mut expected_dw = vec![0.; 4 * width];
    let mut error: f64 = 0.;
    for (row, (state, upstream)) in states
        .chunks_exact(4 * width)
        .zip(grad.chunks_exact(width))
        .enumerate()
    {
        for c in 0..width {
            let mut expected_y = 0.;
            for s in 0..4 {
                let k = s * width + c;
                let weight = 0.25 + f64::from(offsets[k]);
                expected_y += weight * f64::from(state[k]);
                expected_dw[k] += f64::from(upstream[c]) * f64::from(state[k]);
                near(
                    dx[row * 4 * width + k],
                    f64::from(upstream[c]) * weight,
                    &mut error,
                )?;
            }
            near(y[row * width + c], expected_y, &mut error)?;
        }
    }
    for (&a, reference) in dw.iter().zip(expected_dw) {
        near(a, reference, &mut error)?;
    }
    println!(
        "stream oracle rows={rows} width={width} checked={} max_absolute_error={error:.3e}",
        y.len() + dx.len() + dw.len()
    );
    Ok(())
}

fn end_to_end(length: usize, width: usize) -> Result<()> {
    let mut cfg = config(3)?;
    cfg.width = width;
    cfg.heads = if architecture::V1_RESIDUAL { 4 } else { width / 32 };
    cfg.gain = cfg.heads as f32 / width as f32;
    cfg.max_length = length;
    cfg.vocab = 37;
    let mut rng = random::Random::new(179);
    let original_cfg = cfg.clone();
    cfg.head = architecture::HEAD;
    let mut parameters = layout::initialize_model(&cfg, 2);
    let mut control = if architecture::COMPACT {
        Engine::new(compact::wide_config(&cfg), &compact::pad(&cfg, &parameters))?
    } else {
        let plain = layout::initialize_model(&original_cfg, 2);
        Engine::new(original_cfg, &plain)?
    };
    let p = cfg.vocab;
    let rows = 3 * length;
    let tokens: Vec<_> = (0..rows)
        .map(|i| [3, 21, 3, 8, 0, 34, 21, 11, 36][i % 9])
        .collect();
    let labels: Vec<_> = (0..rows)
        .map(|i| {
            if i % 3 == 0 {
                u32::MAX
            } else {
                tokens[(i + 2) % rows]
            }
        })
        .collect();
    control.forward(&tokens, &labels, 3, length)?;
    let original = control.logits.read()?;
    drop(control);
    let start = architecture::start(&cfg);
    let end = start + architecture::parameters(&cfg);
    let mut e = Engine::new(cfg.clone(), &parameters)?;
    e.forward(&tokens, &labels, 3, length)?;
    let initial = e.logits.read()?;
    if architecture::COMPACT {
        compact_checks::close("initial padded logits", &initial, &original, 2e-5)?;
    } else if architecture::HEAD.replaces_context() && !architecture::HEAD.unit_read_gates() {
        if initial == original {
            return Err("an active replacement read reproduced the unrestricted stream".into());
        }
    } else if initial != original {
        return Err("zero read offsets failed the pinned stream mean comparison".into());
    }
    for w in &mut parameters[start..end] {
        *w = rng.gauss(0., 0.15) as f32;
    }
    if architecture::TRIPLE_TRANSPORT {
        crate::slot_checks::fixture(&cfg, &mut parameters);
    }
    e.parameters.write(&parameters)?;
    e.forward(&tokens, &labels, 3, length)?;
    let logits = e.logits.read()?;
    if architecture::HEAD.unit_read_gates() && logits == original {
        return Err("learned slot reads left the reference logits unchanged".into());
    }
    e.backward(3, length)?;
    let gradients = e.gradients.read()?;
    let unmasked = labels.iter().filter(|&&v| v != u32::MAX).count();
    let read_epsilon = if architecture::BOUNDED
        || architecture::INPUT_PRODUCT
        || architecture::PHASE_MEMORY
        || architecture::CONTEXT_ROTATION
        || architecture::HOUSEHOLDER
        || architecture::TOEPLITZ
        || architecture::TWO_FRAME
        || architecture::TRIPLE_TRANSPORT
        || architecture::FOURIER
    {
        0.0003
    } else {
        0.003
    };
    let mut ranges = vec![(0..p * width, 0.0002), (start..end, read_epsilon)];
    if architecture::V1_RESIDUAL {
        ranges.push((layout::score_offset(&cfg)..layout::total_count(&cfg), 0.0003));
    }
    for (range, epsilon) in ranges {
        if range.is_empty() {
            continue;
        }
        for trial in 0..3 {
            let signs: Vec<_> = range
                .clone()
                .map(|_| if rng.gauss(0., 1.) >= 0. { 1. } else { -1. })
                .collect();
            let adjoint: f64 = gradients[range.clone()]
                .iter()
                .zip(&signs)
                .map(|(&g, &s)| f64::from(g) * s)
                .sum();
            let mut difference = |step: f64| -> Result<f64> {
                let mut loss = [0.; 2];
                for (i, sign) in [-1., 1.].into_iter().enumerate() {
                    let mut changed = parameters.clone();
                    for (w, &s) in changed[range.clone()].iter_mut().zip(&signs) {
                        *w += (sign * step * s) as f32;
                    }
                    e.parameters.write(&changed)?;
                    e.forward(&tokens, &labels, 3, length)?;
                    loss[i] =
                        e.losses(rows)?.into_iter().map(f64::from).sum::<f64>() / unmasked as f64;
                }
                Ok((loss[1] - loss[0]) / (2. * step))
            };
            let coarse = difference(epsilon)?;
            let finite = if architecture::BILINEAR_PHASE
                || architecture::CROSSED_CONTEXT
                || architecture::SHARED_BILINEAR
                || architecture::SCALAR_PHASE
                || architecture::TWO_FRAME
                || architecture::HOUSEHOLDER
            {
                let fine = difference(epsilon / 2.)?;
                let finer = difference(epsilon / 4.)?;
                memory_checks::refined_difference(
                    &format!(
                        "stream width={width} length={length} start={} trial={trial}",
                        range.start
                    ),
                    adjoint,
                    coarse,
                    fine,
                    finer,
                )?
            } else {
                coarse
            };
            if !finite.is_finite()
                || !adjoint.is_finite()
                || (adjoint - finite).abs() > 0.012 * (1. + finite.abs())
            {
                return Err(format!("stream derivative {adjoint} differs from {finite}").into());
            }
            println!(
                "stream direction length={length} width={width} first={} trial={trial} analytic={adjoint:.6} numeric={finite:.6}",
                range.start
            );
        }
    }
    let mut permutation: Vec<_> = (0..p as u32).collect();
    rng.shuffle(&mut permutation);
    let mut renamed = parameters.clone();
    for (i, &j) in permutation.iter().enumerate() {
        let j = j as usize;
        renamed[j * width..(j + 1) * width]
            .copy_from_slice(&parameters[i * width..(i + 1) * width]);
    }
    let renamed_tokens: Vec<_> = tokens.iter().map(|&i| permutation[i as usize]).collect();
    e.parameters.write(&renamed)?;
    e.forward(&renamed_tokens, &vec![u32::MAX; rows], 3, length)?;
    let actual = e.logits.read()?;
    for row in 0..rows {
        for (v, &j) in permutation.iter().enumerate() {
            if logits[row * p + v].to_bits() != actual[row * p + j as usize].to_bits() {
                return Err("vocabulary permutation changed a contextual logit".into());
            }
        }
    }
    if architecture::parameters(&cfg) > 0 {
        e.parameters.write(&parameters)?;
        e.forward(&tokens, &labels, 3, length)?;
        e.backward(3, length)?;
        let mut optimizer = optimizer::Optimizer::new(&e, BUDGET)?;
        optimizer.step(&e, 0)?;
        // The pinned schedule starts with a zero learning rate. The moments
        // still update before the first nonzero-rate step.
        if e.parameters.read()?[start..end] != parameters[start..end]
            || e.first_moment.read()?[start..end].iter().all(|&x| x == 0.)
        {
            return Err("unexpected stream moments or weights at zero learning rate".into());
        }
        e.forward(&tokens, &labels, 3, length)?;
        e.backward(3, length)?;
        optimizer.step(&e, 1)?;
        if e.parameters.read()?[start..end] == parameters[start..end] {
            return Err("AdamW failed to update the learned readout".into());
        }
    }
    println!(
        "embedding model length={length} width={width} bit_identical_initial_logits={} permuted_logits={} additional_parameters={}",
        initial
            .iter()
            .zip(&original)
            .filter(|(x, y)| x.to_bits() == y.to_bits())
            .count(),
        rows * p,
        end - start
    );
    Ok(())
}

pub fn run(width: Option<usize>) -> Result<()> {
    if !precision::FP32_AFFINE_REFERENCE {
        return Err("the numerical check requires fp32-affine-reference".into());
    }
    if let Some(width) = width {
        if ![64, 128, 256].contains(&width) {
            return Err("shared checks support widths 64, 128 and 256".into());
        }
        if architecture::STREAM_READ {
            scalar_oracle(35, width)?;
        }
        return end_to_end(9, width);
    }
    if architecture::STREAM_READ {
        for (rows, width) in [(1, 16), (7, 32), (35, 128)] {
            scalar_oracle(rows, width)?;
        }
    }
    for (length, width) in [(1, 256), (2, 128), (9, 64)] {
        end_to_end(length, width)?;
    }
    if architecture::V1_RESIDUAL {
        return Ok(());
    }
    if architecture::COMPACT {
        compact_checks::run()?;
    }
    if architecture::ASSOCIATIVE_REFLECTION {
        crate::associative_checks::run()?;
    } else if architecture::SCALAR_PHASE {
        crate::scalar_phase_checks::run()?;
    } else if architecture::TOEPLITZ {
        crate::toeplitz_checks::run()?;
    } else if architecture::TWO_FRAME {
        crate::two_frame_checks::run()?;
    } else if architecture::TRIPLE_TRANSPORT {
        crate::slot_gradients::run()?;
    } else if architecture::FOURIER {
        crate::fourier_checks::run()?;
    } else if architecture::PHASE_MEMORY {
        crate::phase_checks::run()?;
    }
    if architecture::CONTEXT_ROTATION {
        crate::context_rotation_checks::run()?;
    }
    if architecture::HOUSEHOLDER {
        crate::householder_checks::run()?;
    }
    crate::embedding_checks::run()
}
