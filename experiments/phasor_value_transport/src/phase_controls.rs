//! Matched backbone and arbitrary-vector controls for the learned phase reads.
use super::*;
use head::MEMORY_WIDTH as M;

/// Remove the extension projections and compare with the input-phase model.
/// Keep active output maps and all trailing rotary weights in both models.
pub(super) fn zero_extension(cfg: &Config, weights: &[f32], ordered: &mut Engine) -> Result<()> {
    let size = M * cfg.width;
    let first = architecture::memory_start(cfg);
    let end = first + cfg.head.phase_inputs() * cfg.width;
    let mut controlled = weights.to_vec();
    let removed = match cfg.head {
        head::Head::PeriodicReflection => {
            let after = end + 2 * size + M;
            if weights[after..after + M].iter().any(|&x| x != 1.)
                || weights[after + M..after + 2 * M].iter().any(|&x| x != 0.)
            {
                return Err(
                    "periodic activation must start at unit strength and zero center".into(),
                );
            }
            controlled[after..after + M].fill(0.);
            after..after + 2 * M
        }
        head::Head::ReflectionPhase => {
            let gates = end + 2 * size..end + 2 * size + M;
            if weights[gates.clone()].iter().any(|&x| x != 0.5) {
                return Err("reflection gates must initialize at one half".into());
            }
            controlled[gates.clone()].fill(0.);
            gates
        }
        head::Head::AffinePhase | head::Head::BilinearPhase => {
            let inactive = if cfg.head == head::Head::AffinePhase {
                first + size
            } else {
                first + 2 * size
            };
            if weights[inactive..end].iter().any(|&x| x != 0.) {
                return Err("initial ordered contributions must be zero".into());
            }
            first + size..end
        }
        _ => return Err("expected an ordered phase extension".into()),
    };
    let mut base_cfg = cfg.clone();
    base_cfg.head = if cfg.head == head::Head::PeriodicReflection {
        head::Head::ReflectionPhase
    } else {
        head::Head::InputPhase
    };
    let base_weights = [&weights[..removed.start], &weights[removed.end..]].concat();
    let mut base = Engine::new(base_cfg, &base_weights)?;
    ordered.parameters.write(&controlled)?;
    compare_dense(cfg, ordered, &mut base, "zero ordered contribution")
}

/// With the extra gates fixed to one, the mixed model supplies a separately
/// implemented reference for the whole single-state solve and input VJP.
pub(super) fn one_gate(
    cfg: &Config,
    weights: &[f32],
    gates: usize,
    single: &mut Engine,
) -> Result<()> {
    let mut mixed = cfg.clone();
    mixed.head = head::Head::ReflectionPhase;
    let mixed_weights = [&weights[..gates], &[1.; M], &weights[gates..]].concat();
    let mut reference = Engine::new(mixed, &mixed_weights)?;
    compare_dense(
        cfg,
        single,
        &mut reference,
        "single reflection against unit gates",
    )
}

fn compare_dense(cfg: &Config, ordered: &mut Engine, base: &mut Engine, name: &str) -> Result<()> {
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
        "{name} width={} hidden={hidden:.3e} logits={logits:.3e} input_gradient={gradient:.3e}",
        cfg.width
    );
    Ok(())
}

/// Retain exactly the same contextual backbone, classifier and rotary frequencies.
pub(super) fn stream_control(cfg: &Config, weights: &[f32]) -> Result<Engine> {
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
    Engine::new(control, &base_weights)
}

/// Subtract the control's hidden vector and input gradient at the endpoint.
pub(super) fn endpoint(
    e: &mut Engine,
    base: &mut Engine,
    input: &[f32],
    g: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let width = e.config.width;
    let length = input.len() / width;
    let mut cotangent = vec![0.; length * g.len()];
    cotangent[(length - 1) * g.len()..].copy_from_slice(g);
    let a = e.infer_embeddings(input, 1, length)?;
    let b = base.infer_embeddings(input, 1, length)?;
    let da = e.embedding_pullback(input, 1, length, &cotangent)?;
    let db = base.embedding_pullback(input, 1, length, &cotangent)?;
    let hidden = a.hidden[(length - 1) * width..]
        .iter()
        .zip(&b.hidden[(length - 1) * width..])
        .map(|(x, y)| x - y)
        .collect();
    Ok((hidden, da.iter().zip(db).map(|(x, y)| x - y).collect()))
}

/// Swap two unequal off-table segments after isolating the phase contribution.
pub(super) fn input_segment(cfg: &Config, weights: &[f32], e: &mut Engine) -> Result<()> {
    let mut base = stream_control(cfg, weights)?;
    let (width, length, vocab) = (cfg.width, cfg.max_length, cfg.vocab);
    let mut rng = random::Random::new(717);
    let mut values: Vec<_> = (0..length * width)
        .map(|_| rng.gauss(0., 0.28) as f32)
        .collect();
    let cotangent: Vec<_> = (0..vocab).map(|_| rng.gauss(0., 0.2) as f32).collect();
    let (ab, mut gab) = endpoint(e, &mut base, &values, &cotangent)?;
    let shift = if matches!(
        cfg.head,
        head::Head::ReflectionPhase | head::Head::SingleReflection | head::Head::PeriodicReflection
    ) {
        let mut even = values.clone();
        even.rotate_left(2 * width);
        let (read, grad) = endpoint(e, &mut base, &even, &cotangent)?;
        let mut old_gradient = gab.clone();
        old_gradient.rotate_left(2 * width);
        let hidden = compact_checks::close("even reflection segment exchange", &ab, &read, 1e-5)?;
        let derivative =
            compact_checks::close("even reflection exchange VJP", &old_gradient, &grad, 4e-5)?;
        println!(
            "even reflection segment width={width} hidden={hidden:.3e} input_gradient={derivative:.3e}"
        );
        width
    } else {
        2 * width
    };
    values.rotate_left(shift);
    let (ba, gba) = endpoint(e, &mut base, &values, &cotangent)?;
    gab.rotate_left(shift);
    if !ab.iter().any(|x| x.abs() > 0.01) || !gba[..width].iter().any(|x| x.abs() > 0.001) {
        return Err("segment check requires active phase outputs and early input gradients".into());
    }
    if matches!(
        cfg.head,
        head::Head::AffinePhase
            | head::Head::BilinearPhase
            | head::Head::ReflectionPhase
            | head::Head::SingleReflection
            | head::Head::PeriodicReflection
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
