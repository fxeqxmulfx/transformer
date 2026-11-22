//! Matched backbone and arbitrary-vector controls for the learned phase reads.
use super::*;
use head::MEMORY_WIDTH as M;

/// Remove the extension projections and compare with the input-phase model.
/// Keep active output maps and all trailing rotary weights in both models.
pub(super) fn zero_extension(cfg: &Config, weights: &[f32], ordered: &mut Engine) -> Result<()> {
    let size = M * cfg.width;
    let first = architecture::memory_start(cfg);
    let end = first + cfg.head.phase_inputs() * cfg.width;
    let inactive = match cfg.head {
        head::Head::AffinePhase => first + size,
        head::Head::BilinearPhase => first + 2 * size,
        _ => return Err("expected an ordered phase extension".into()),
    };
    if weights[inactive..end].iter().any(|&x| x != 0.) {
        return Err("initial ordered contributions must be zero".into());
    }
    let mut base_cfg = cfg.clone();
    base_cfg.head = head::Head::InputPhase;
    let base_weights = [&weights[..first + size], &weights[end..]].concat();
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
    let hidden = compact_checks::close("initial ordered hidden", &a.hidden, &b.hidden, 3e-6)?;
    let logits = compact_checks::close("initial ordered logits", &a.logits, &b.logits, 3e-6)?;
    let gradient = compact_checks::close("initial ordered input VJP", &da, &db, 1e-5)?;
    println!(
        "zero ordered contribution width={} hidden={hidden:.3e} logits={logits:.3e} input_gradient={gradient:.3e}",
        cfg.width
    );
    Ok(())
}

/// Isolate the residual and its input VJP by subtracting a model with the
/// same trained backbone and classifier. Swap unequal off-table segments.
pub(super) fn input_segment(cfg: &Config, weights: &[f32], e: &mut Engine) -> Result<()> {
    let mut control = cfg.clone();
    control.head = head::Head::Stream;
    let memory = architecture::memory_start(cfg);
    if layout::frequency_offset(&control) != memory {
        return Err("input phase parameters overlap the stream backbone".into());
    }
    // The native layout stores the learned rotary frequencies after the readout.
    let base_weights = [
        &weights[..memory],
        &weights[layout::frequency_offset(cfg)..],
    ]
    .concat();
    let mut base = Engine::new(control, &base_weights)?;
    let (width, length, vocab) = (cfg.width, cfg.max_length, cfg.vocab);
    let mut rng = random::Random::new(717);
    let mut values: Vec<_> = (0..length * width)
        .map(|_| rng.gauss(0., 0.28) as f32)
        .collect();
    let mut cotangent = vec![0.; length * vocab];
    for g in &mut cotangent[(length - 1) * vocab..] {
        *g = rng.gauss(0., 0.2) as f32;
    }
    let mut residual = |input: &[f32]| -> Result<(Vec<f32>, Vec<f32>)> {
        let a = e.infer_embeddings(input, 1, length)?;
        let b = base.infer_embeddings(input, 1, length)?;
        let g = e.embedding_pullback(input, 1, length, &cotangent)?;
        let h = base.embedding_pullback(input, 1, length, &cotangent)?;
        let read = a.hidden[(length - 1) * width..]
            .iter()
            .zip(&b.hidden[(length - 1) * width..])
            .map(|(x, y)| x - y)
            .collect();
        Ok((read, g.iter().zip(h).map(|(x, y)| x - y).collect()))
    };
    let (ab, mut gab) = residual(&values)?;
    values.rotate_left(2 * width);
    let (ba, gba) = residual(&values)?;
    gab.rotate_left(2 * width);
    if !ab.iter().any(|x| x.abs() > 0.01) || !gba[..width].iter().any(|x| x.abs() > 0.001) {
        return Err("segment check requires active phase outputs and early input gradients".into());
    }
    if matches!(
        cfg.head,
        head::Head::AffinePhase | head::Head::BilinearPhase
    ) {
        let changed = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0_f32, f32::max)
        };
        let forward = changed(&ab, &ba);
        let gradient = changed(&gab, &gba);
        if forward < 0.01 || gradient < 0.001 {
            return Err(
                "learned phase read failed to distinguish ordered off-table segments".into(),
            );
        }
        println!(
            "ordered phase segment width={width} hidden_difference={forward:.3e} input_gradient_difference={gradient:.3e}"
        );
        return Ok(());
    }
    let forward = compact_checks::close("phase segment exchange", &ab, &ba, 8e-6)?;
    let grad = compact_checks::close("phase segment VJP exchange", &gab, &gba, 3e-5)?;
    println!(
        "input phase segment width={width} max_absolute_hidden={forward:.3e} input_grad={grad:.3e}"
    );
    Ok(())
}
